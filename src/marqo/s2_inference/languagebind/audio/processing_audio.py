import cv2
import numpy as np
import torch
import torchaudio
from torchvision import transforms
from transformers import ProcessorMixin, BatchEncoding
from transformers.image_processing_utils import BatchFeature
from torch.nn import functional as F


def make_list_of_images(x):
    if not isinstance(x, list):
        return [x]
    return x


torchaudio.set_audio_backend("soundfile")

def torchaudio_loader(path):
    return torchaudio.load(path)

def int16_to_float32_torch(x):
    return (x / 32767.0).type(torch.float32)

def float32_to_int16_torch(x):
    x = torch.clamp(x, min=-1., max=1.)
    return (x * 32767.).type(torch.int16)

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10

class AudioTransform:
    def __init__(self, args):
        self.sample_rate = args.audio_sample_rate
        self.num_mel_bins = args.num_mel_bins
        self.target_length = args.target_length
        self.audio_mean = args.audio_mean
        self.audio_std = args.audio_std
        self.mean = []
        self.std = []
        # mean=-4.2677393
        # std=4.5689974
        # self.norm = transforms.Normalize(mean=self.audio_mean, std=self.audio_std)


    def __call__(self, audio_data_and_origin_sr):
        audio_data, origin_sr = audio_data_and_origin_sr
        if self.sample_rate != origin_sr:
            # print(audio_data.shape, origin_sr)
            audio_data = torchaudio.functional.resample(audio_data, orig_freq=origin_sr, new_freq=self.sample_rate)
        waveform_melspec = self.waveform2melspec(audio_data)
        return waveform_melspec


    def waveform2melspec(self, audio_data):
        mel = self.get_mel(audio_data)
        if mel.shape[0] > self.target_length:
            # split to three parts
            chunk_frames = self.target_length
            total_frames = mel.shape[0]
            ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
            # print('total_frames-chunk_frames:', total_frames-chunk_frames,
            #       'len(audio_data):', len(audio_data),
            #       'chunk_frames:', chunk_frames,
            #       'total_frames:', total_frames)
            if len(ranges[1]) == 0:  # if the audio is too short, we just use the first chunk
                ranges[1] = [0]
            if len(ranges[2]) == 0:  # if the audio is too short, we just use the first chunk
                ranges[2] = [0]
            # randomly choose index for each part
            idx_front = np.random.choice(ranges[0])
            idx_middle = np.random.choice(ranges[1])
            idx_back = np.random.choice(ranges[2])
            # idx_front = ranges[0][0]  # fixed
            # idx_middle = ranges[1][0]
            # idx_back = ranges[2][0]
            # select mel
            mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
            mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
            mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]
            # print(total_frames, idx_front, idx_front + chunk_frames, idx_middle, idx_middle + chunk_frames, idx_back, idx_back + chunk_frames)
            # stack
            mel_fusion = torch.stack([mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
        elif mel.shape[0] < self.target_length:  # padding if too short
            n_repeat = int(self.target_length / mel.shape[0]) + 1
            # print(self.target_length, mel.shape[0], n_repeat)
            mel = mel.repeat(n_repeat, 1)[:self.target_length, :]
            mel_fusion = torch.stack([mel, mel, mel], dim=0)
        else:  # if equal
            mel_fusion = torch.stack([mel, mel, mel], dim=0)
        mel_fusion = mel_fusion.transpose(1, 2)  # [3, target_length, mel_bins] -> [3, mel_bins, target_length]

        # self.mean.append(mel_fusion.mean())
        # self.std.append(mel_fusion.std())
        mel_fusion = (mel_fusion - self.audio_mean) / (self.audio_std * 2)
        return mel_fusion

    def get_mel(self, audio_data):
        # mel shape: (n_mels, T)
        audio_data -= audio_data.mean()
        mel = torchaudio.compliance.kaldi.fbank(
            audio_data,
            htk_compat=True,
            sample_frequency=self.sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.num_mel_bins,
            dither=0.0,
            frame_length=25,
            frame_shift=DEFAULT_AUDIO_FRAME_SHIFT_MS,
        )
        return mel  # (T, n_mels)

def get_audio_transform(config):
    config = config.vision_config
    return AudioTransform(config)


def load_and_transform_audio(
    audio_path,
    transform,
):
    waveform_and_sr = torchaudio_loader(audio_path)
    audio_outputs = transform(waveform_and_sr)

    return audio_outputs

class LanguageBindAudioProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("LanguageBindAudioTokenizer")

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.transform = get_audio_transform(config)
        self.image_processor = load_and_transform_audio
        self.tokenizer = tokenizer

    def __call__(self, images=None, text=None, context_length=77, return_tensors=None, **kwargs):
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, max_length=context_length, padding='max_length',
                                      truncation=True, return_tensors=return_tensors, **kwargs)

        if images is not None:
            images = make_list_of_images(images)
            image_features = [self.image_processor(image, self.transform) for image in images]
            image_features = torch.stack(image_features)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features
            return encoding
        elif text is not None:
            return encoding
        else:
            return {"pixel_values": image_features}

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)
