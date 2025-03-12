from django.apps import AppConfig
from pathlib import Path
from .model_src.gen.ltn_model import CustomDDPM
from .model_src.gen.utils import get_transforms
import yaml

import torch

class DjangoAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "django_app"

    # --------- when invoking inside 'django_proj' folder ---------
    # load a model
    ckpt_path = "./django_app/model_ckpt/gen/last.ckpt"
    hparams_path = "./django_app/model_ckpt/gen/hparams.yaml"
    plate_dict_path = "./django_app/model_src/gen/plate_dict.json"
    
    # # --------- for debugging ---------
    # ckpt_path = "./django_proj/django_app/model_ckpt/gen/last.ckpt"
    # hparams_path = "./django_proj/django_app/model_ckpt/gen/hparams.yaml"
    # plate_dict_path = "./django_proj/django_app/model_src/gen/plate_dict.json"
    
    with open(hparams_path) as stream:
        hp = yaml.safe_load(stream)
    print("========================", hp, "========================")
    gen_model = CustomDDPM.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        multi_class_nums=hp["multi_class_nums"],
        num_continuous_class_embeds=hp["num_continuous_class_embeds"],
        train_num_steps=hp["train_num_steps"],
        train_batch_size=hp["train_batch_size"],
        unet_sample_size=hp["unet_sample_size"],
        unet_block_out_channels=hp["unet_block_out_channels"],
        train_scheduler_name=hp["train_scheduler_name"],
        inference_scheduler_name=hp["inference_scheduler_name"],
        inference_num_steps=hp["inference_num_steps"],
        inference_batch_size=hp["inference_batch_size"],
        inference_height=hp["inference_height"],
        inference_width=hp["inference_width"],
        lr=hp["lr"],
        is_train=False,
        strict=False
    )
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    gen_model.to(DEVICE)
    
    transforms = get_transforms(
        height=480,
        width=640,
        plate_dict_path=plate_dict_path
    )
    print("gen_model loaded!!!!")
