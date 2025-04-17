from django.apps import AppConfig
from pathlib import Path
from .model_src.gen.ltn_model import CustomDDPM
from .model_src.gen.utils import get_transforms
from .model_src.seg.ltn_model import SPRSegmentModel
import yaml

import torch

class DjangoAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "django_app"

    # --------- when invoking inside 'django_proj' folder ---------
    # load a model
    gen_ckpt_path = "./django_app/model_ckpt/gen/epoch=2999-step=33000-train_loss=0.0021_last.ckpt"
    plate_dict_path = "./django_app/model_src/gen/plate_dict.json"
        
    # # --------- for debugging ---------
    # ckpt_path = "./django_proj/django_app/model_ckpt/gen/last.ckpt"
    # hparams_path = "./django_proj/django_app/model_ckpt/gen/hparams.yaml"
    # plate_dict_path = "./django_proj/django_app/model_src/gen/plate_dict.json"
    
    
    gen_model = CustomDDPM.load_from_checkpoint(
        checkpoint_path=gen_ckpt_path,
    )
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    gen_model.to(DEVICE)
    
    gen_transforms = get_transforms(
        height=480,
        width=640,
        plate_dict_path=plate_dict_path
    )
    print("gen_model loaded!!!!")

    # Segmentation model
    seg_ckpt_path = "./django_app/model_ckpt/seg/seg.ckpt"
    seg_model = SPRSegmentModel.load_from_checkpoint(
        checkpoint_path=seg_ckpt_path
    )
    print("seg model loaded!!!")
    
    labelmap_path = "./django_app/model_src/seg/labelmap.txt"