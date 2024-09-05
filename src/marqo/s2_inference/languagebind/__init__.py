from torch import nn

from .audio.configuration_audio import LanguageBindAudioConfig
from .audio.modeling_audio import LanguageBindAudio
from .audio.processing_audio import LanguageBindAudioProcessor
from .audio.tokenization_audio import LanguageBindAudioTokenizer
from .image.configuration_image import LanguageBindImageConfig
from .image.modeling_image import LanguageBindImage
from .image.processing_image import LanguageBindImageProcessor
from .image.tokenization_image import LanguageBindImageTokenizer
from .video.configuration_video import LanguageBindVideoConfig
from .video.modeling_video import LanguageBindVideo
from .video.processing_video import LanguageBindVideoProcessor
from .video.tokenization_video import LanguageBindVideoTokenizer

config_dict = {
    'image': LanguageBindImageConfig,
    'video': LanguageBindVideoConfig,
    'audio': LanguageBindAudioConfig
}
model_dict = {
    'image': LanguageBindImage,
    'video': LanguageBindVideo,
    'audio': LanguageBindAudio
}
transform_dict = {
    'video': LanguageBindVideoProcessor,
    'audio': LanguageBindAudioProcessor,
    'image': LanguageBindImageProcessor,
}


class LanguageBind(nn.Module):
    def __init__(self, clip_type, use_temp=True, cache_dir='./cache_dir'):
        super(LanguageBind, self).__init__()
        self.use_temp = use_temp
        self.modality_encoder = {}
        self.modality_proj = {}
        self.modality_scale = {}
        self.modality_config = {}
        for k, v in clip_type.items():
            pretrained_ckpt = f'LanguageBind/{v}'
            model = model_dict[k].from_pretrained(pretrained_ckpt, cache_dir=cache_dir)
            self.modality_encoder[k] = model.vision_model
            self.modality_proj[k] = model.visual_projection
            self.modality_scale[k] = model.logit_scale
            self.modality_config[k] = model.config
        self.modality_encoder['language'] = model.text_model
        self.modality_proj['language'] = model.text_projection

        self.modality_encoder = nn.ModuleDict(self.modality_encoder)
        self.modality_proj = nn.ModuleDict(self.modality_proj)

    def forward(self, inputs):
        outputs = {}
        for key, value in inputs.items():
            value = self.modality_encoder[key](value)[1]
            value = self.modality_proj[key](value)
            value = value / value.norm(p=2, dim=-1, keepdim=True)
            if self.use_temp:
                if key != 'language':
                    value = value * self.modality_scale[key].exp()
            outputs[key] = value
        return outputs


def to_device(x, device):
    out_dict = {k: v.to(device) for k, v in x.items()}
    return out_dict
