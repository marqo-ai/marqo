import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import ProcessorMixin, BatchEncoding
from transformers.image_processing_utils import BatchFeature

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def make_list_of_images(x):
    if not isinstance(x, list):
        return [x]
    return x

def opencv_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype('float32')


class DepthNorm(nn.Module):
    def __init__(
        self,
        max_depth=0,
        min_depth=0.01,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.scale = 1000.0  # nyuv2 abs.depth

    def forward(self, image):
        # image = np.array(image)
        depth_img = image / self.scale  # (H, W)   in meters
        depth_img = depth_img.clip(min=self.min_depth)
        if self.max_depth != 0:
            depth_img = depth_img.clip(max=self.max_depth)
            depth_img /= self.max_depth   #  0-1
        else:
            depth_img /= depth_img.max()
        depth_img = torch.from_numpy(depth_img).unsqueeze(0).repeat(3, 1, 1)  # assume image
        return depth_img.to(torch.get_default_dtype())

def get_depth_transform(config):
    config = config.vision_config
    transform = transforms.Compose(
        [
            DepthNorm(max_depth=config.max_depth),
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),  # assume image
            # transforms.Normalize((0.5, ), (0.5, ))  # 0-1 to norm distribution
            # transforms.Normalize((0.0418, ), (0.0295, ))  # sun rgb-d  imagebind
            # transforms.Normalize((0.02, ), (0.00295, ))  # nyuv2
        ]
    )
    return transform

def load_and_transform_depth(depth_path, transform):
    depth = opencv_loader(depth_path)
    depth_outputs = transform(depth)
    return depth_outputs

class LanguageBindDepthProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("LanguageBindDepthTokenizer")

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.transform = get_depth_transform(config)
        self.image_processor = load_and_transform_depth
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
