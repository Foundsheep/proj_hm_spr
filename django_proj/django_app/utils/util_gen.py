from ..apps import DjangoAppConfig

import pandas as pd
import torch
from PIL import Image
import io
import base64
import numpy as np

class GenOptionProducer:
    def __init__(self, path):
        self.df = self.get_dummy_df(path)
        self.rivet = "rivet"
        self.die = "die"
        self.upper_type = "upper_type"
        self.upper_thickness = "upper_thickness"
        self.middle_type = "middle_type"
        self.middle_thickness = "middle_thickness"
        self.lower_type = "lower_type"
        self.lower_thickness = "lower_thickness"
        self.is_javascript = False
        self.method = self._give_unique_list_by_key if self.is_javascript else self._give_unique_tupled_list_by_key
        
    def get_condition_options_rivet(self):
        return self.method(self.rivet)
        
    def get_condition_options_die(self):
        return self.method(self.die)

    def get_condition_options_upper_type(self):
        return self.method(self.upper_type)
        
    def get_condition_options_middle_type(self):
        return self.method(self.middle_type)

    def get_condition_options_lower_type(self):
        return self.method(self.lower_type)
        
    def get_dummy_df(self, path):
        return pd.read_csv(path)

    def _give_unique_list_by_key(self, key):
        return self.df[key].dropna().unique().tolist()
    
    def _give_unique_dict_by_key(self, key):
        return {c: c for c in self.df[key].dropna().unique()}
    
    def _give_unique_tupled_list_by_key(self, key):
        return sorted([(c, c) for c in self.df[key].dropna().unique()])
    
    
def generate_image(conds):
    model = DjangoAppConfig.gen_model
    transforms = DjangoAppConfig.transforms
    

    plate_count = transforms["plate_count"](int(conds["plate_count"]))
    rivet = transforms["rivet"](conds["rivet"])
    die = transforms["die"](conds["die"])
    upper_type = transforms["upper_type"](conds["upper_type"])
    upper_thickness = transforms["upper_thickness"](conds["upper_thickness"])
    middle_type = transforms["middle_type"](conds["middle_type"])
    middle_thickness = transforms["middle_thickness"](conds["middle_thickness"])
    lower_type = transforms["lower_type"](conds["lower_type"])
    lower_thickness = transforms["lower_thickness"](conds["lower_thickness"])
    head_height = transforms["head_height"](conds["head_height"])
    
    categorical_conds = (
        torch.stack([
            rivet, die, upper_type, middle_type, lower_type
        ]).to(device=DjangoAppConfig.DEVICE)
    )
    
    continuous_conds = (
        torch.stack([
            plate_count, upper_thickness, middle_thickness, lower_thickness, head_height
        ]).to(device=DjangoAppConfig.DEVICE)
    )
    try:
        with torch.no_grad():
            model.eval()
            out = model(
                batch_size=1,
                categorical_conds=categorical_conds,
                continuous_conds=continuous_conds
            )
    except Exception as e:
        print(e)
        out = torch.randn((3, 300, 400))
    print("************* DONE *************")
    return out


def convert_image_to_base64(img):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif isinstance(img, torch.Tensor):
        # TODO: range should be between [0, 1]
        img = img.permute(1, 2, 0).numpy() * 255
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        pass
    else:
        raise TypeError(f"image type is {type(img)}") 
    raw_bytes = io.BytesIO()
    img.save(raw_bytes, "PNG")
    raw_bytes.seek(0)
    return base64.b64encode(raw_bytes.read()).decode()