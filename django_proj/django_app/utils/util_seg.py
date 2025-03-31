import torch
import torchvision.transforms as T
from ..apps import DjangoAppConfig
from ..model_src.seg.utils import get_transforms, adjust_ratio_and_convert_to_numpy, colour_image
from pathlib import Path
import random
import torch.nn.functional as F
import numpy as np

def segment(images):
    model = DjangoAppConfig.seg_model
    transforms = get_transforms(False)
    images = [
        transforms(image=adjust_ratio_and_convert_to_numpy(img))["image"] for img in images
    ]
    
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        outputs = F.softmax(outputs, dim=1) # N, C, H, W
        outputs = outputs.argmax(1) # N, H, W
        outputs = outputs.cpu().numpy().astype(np.uint8) # N, H, W
    outputs = [colour_image(out, DjangoAppConfig.labelmap_path) for out in outputs]
    return outputs
    