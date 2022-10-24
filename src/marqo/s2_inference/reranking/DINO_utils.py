import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
import cv2

import vision_transformer as vits

# def _vit_registry():

#     vit_models = dict()

#     vit_models[("vit_small", 16)] = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
#     vit_models[("vit_small", 8)]

#     vit_models[("vit_base", 8)]
#     vit_models[("vit_small", 16)]


#     url = None
#     if arch == "vit_small" and patch_size == 16:
#         url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
#     elif arch == "vit_small" and args.patch_size == 8:
#         url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
#     elif arch == "vit_base" and args.patch_size == 16:
#         url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
#     elif arch == "vit_base" and args.patch_size == 8:
#         url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"


def _load_DINO_model(arch, patch_size, device):
    # arch -> ['vit_tiny', 'vit_small', 'vit_base']
    # build model

    allowed_archs = ('vit_small', 'vit_base')
    allowed_patches = (8, 16)
    if arch not in allowed_archs:
        raise KeyError(f"{arch} not found in {allowed_archs}")

    if patch_size not in allowed_patches:
        raise KeyError(f"{patch_size} not found in {allowed_patches}")

    if arch == "vit_small":
        # TODO add onnx support
        if patch_size == 8:
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')

        if patch_size == 16:
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

    elif arch == "vit_base":
        if patch_size == 8:
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')

        if patch_size == 16:
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

    else:
        raise RuntimeError("wrong")

    model.eval()
    model.to(device)

    return model

def _get_DINO_transform(image_size=(224, 224)):
    return pth_transforms.Compose([
    pth_transforms.Resize(image_size),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def DINO_inference(model, img, patch_size, image_size=(224,224), device="cpu"):

    transform = _get_DINO_transform(image_size=image_size)

    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    with torch.no_grad():
        attentions = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    return attentions


def PIL_to_cv2(pil_image):
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    return open_cv_image[:, :, ::-1]

def _rescale_image(image):
    
    image /= image.max()
    image *= 255
    image = image.astype(np.uint8)
    #dtype=np.uint8
    return image

def combine_boxes(boxes):
    pass

def attention_to_bboxs(image):
    
    image = _rescale_image(image)
    backtorgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    gray = cv2.cvtColor(backtorgb, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    bboxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        x1 = x
        x2 = x + w
        y1 = y        
        y2 = y + h
        box = (x1, y1, x2, y2)
        bboxes.append(box)

    return bboxes

if __name__ == "__main__":
    image_name = "/Users/jesseclark/Downloads/IMG_20210507_195425.jpg"
    patch_size = 8

    model = _load_DINO_model("vit_small", patch_size, "cpu")

    image = Image.open(image_name) #PIL_to_cv2()

    attentions = DINO_inference(model, image, patch_size, image_size=(224,224), device="cpu")

    bboxes = attention_to_bboxs(np.abs(attentions).mean(0))

image_box = np.array(image.resize((224,224)))
for box in bboxes:
    
    x1, y1, x2, y2 = box
    cv2.rectangle(image_box, (x1, y1), (x2, y2), (36,255,12), 2)

#cv2.imshow('thresh', thresh)
cv2.imshow('image', image)
# cv2.waitKey()