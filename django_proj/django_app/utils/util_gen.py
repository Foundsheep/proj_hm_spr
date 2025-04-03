from ..apps import DjangoAppConfig

import pandas as pd
import torch
from PIL import Image
import io
import base64
import numpy as np
import traceback

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
        self.is_javascript = True
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
        return sorted(self.df[key].dropna().unique().tolist())
    
    def _give_unique_dict_by_key(self, key):
        return sorted({c: c for c in self.df[key].dropna().unique()})
    
    def _give_unique_tupled_list_by_key(self, key):
        return sorted([(c, c) for c in self.df[key].dropna().unique()])
    
    
def generate_image(conds):
    model = DjangoAppConfig.gen_model
    transforms = DjangoAppConfig.gen_transforms
    conds = conds.dict()
    if "middle_type" not in conds.keys():
        conds["middle_type"] = None
        conds["middle_thickness"] = None
    
    
    batch_size = int(conds["number_to_generate"])
    plate_count = transforms["plate_count"](int(conds["plate_count"]))
    rivet = transforms["rivet"](conds["rivet"])
    die = transforms["die"](conds["die"])
    upper_type = transforms["upper_type"](conds["upper_type"])
    upper_thickness = transforms["upper_thickness"](float(conds["upper_thickness"]))
    middle_type = transforms["middle_type"](conds["middle_type"])
    middle_thickness = transforms["middle_thickness"](float(conds["middle_thickness"]) if conds["middle_thickness"] is not None else None)
    lower_type = transforms["lower_type"](conds["lower_type"])
    lower_thickness = transforms["lower_thickness"](float(conds["lower_thickness"]))
    head_height = transforms["head_height"](float(conds["head_height"]))
    
    categorical_conds = (
        torch.stack([rivet, die, upper_type, lower_type, middle_type], dim=0)
        .to(DjangoAppConfig.DEVICE)
    )
    # continuous_conds = (
    #     torch.stack([plate_count, upper_thickness, middle_thickness, lower_thickness, head_height])
    #     .to(device="cuda" if torch.cuda.is_available() else "cpu")
    # )
    continuous_conds = (
        torch.stack([upper_thickness, lower_thickness, middle_thickness, head_height], dim=1)
        .to(DjangoAppConfig.DEVICE)
    )

    try:
        with torch.no_grad():
            model.eval()
            outs = model(
                batch_size=batch_size,
                categorical_conds=categorical_conds,
                continuous_conds=continuous_conds,
                do_post_process=True,
                do_save_fig=False,
            )
    except Exception as e:
        traceback.print_exc()
        outs = torch.randn((batch_size, 3, 300, 400))
    print("************* DONE *************")
    return outs


def convert_image_to_base64(img: np.ndarray | Image.Image | torch.Tensor) -> str:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    elif isinstance(img, torch.Tensor):
        # TODO: range should be between [0, 1]
        img = img.permute(1, 2, 0).cpu().numpy()
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