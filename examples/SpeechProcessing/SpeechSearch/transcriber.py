from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from pyannote.audio import Pipeline
from tqdm import tqdm
import librosa
import numpy as np
import os

from typing import Set, List, Tuple, Dict, Any


class AudioTranscriber:
    def __init__(
        self, hf_token: str, device: str = "cpu", transcription_batch_size: int = 4
    ):

        self.device = device
        self.sample_rate = 16000

        self.transcription_batch_size = transcription_batch_size

        self._model_size = "medium"

        self.transcription_model = Speech2TextForConditionalGeneration.from_pretrained(
            f"facebook/s2t-{self._model_size}-librispeech-asr"
        )
        self.transcription_model.to(self.device)
        self.transcription_processor = Speech2TextProcessor.from_pretrained(
            f"facebook/s2t-{self._model_size}-librispeech-asr"
        )
        self.annotation_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1", use_auth_token=hf_token
        )

    def annotate(self, file: str) -> List[Tuple[float, float, Set[str]]]:
        diarization = self.annotation_pipeline(file)
        speaker_times = []
        for t in diarization.get_timeline():
            start, end = t.start, t.end
            # reduce to 30 second chunks in case of long segments
            while end - start > 0:
                speaker_times.append(
                    (start, min(start + 30, end), diarization.get_labels(t))
                )
                start += 30

        return speaker_times

    def transcribe(self, datas: List[np.ndarray], samplerate: int = 16000) -> List[str]:
        batches = []
        batch = []
        i = 0
        for data in datas:
            # pad short audio
            if data.shape[0] < 400:
                data = np.pad(data, [(0, 400)], mode="constant")

            batch.append(data)
            i += 1
            if i > self.transcription_batch_size:
                batches.append(batch)
                i = 0
                batch = []
        if batch:
            batches.append(batch)

        transcriptions = []
        for batch in tqdm(
            batches, desc=f"Processing with batch size {self.transcription_batch_size}"
        ):
            inputs = self.transcription_processor(
                batch, sampling_rate=samplerate, return_tensors="pt", padding=True
            )
            generated_ids = self.transcription_model.generate(
                inputs["input_features"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
            )
            transcription_batch = self.transcription_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            transcriptions += transcription_batch

        return transcriptions

    def _create_id(self, file: str, i: int):
        return os.path.basename(file) + f"_{i}"

    def process_audio(self, file: str) -> Dict[str, Any]:
        speaker_times = self.annotate(file)
        audio_data, samplerate = librosa.load(file, sr=self.sample_rate)

        datas = []
        for start, end, _ in speaker_times:
            datas.append(audio_data[int(start * samplerate) : int(end * samplerate)])

        transcriptions = self.transcribe(datas, samplerate)

        annotated_transcriptions = []
        for i in range(len(transcriptions)):
            annotated_transcriptions.append(
                {
                    "_id": self._create_id(file, i),
                    "speaker": [*speaker_times[i][2]],
                    "start": speaker_times[i][0],
                    "end": speaker_times[i][1],
                    "transcription": transcriptions[i],
                    "samplerate": samplerate,
                    "file": file,
                }
            )

        return annotated_transcriptions
