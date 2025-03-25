import json


def read_json(path: str) -> dict:
    with open(path, "r") as j:
        data = json.load(j)
    return data

def get_plate_name_list_in_dict(path: str) -> dict:
    data = read_json(path)
    plate_name_dict = data["plate_name_list"]
    plate_name_list = list(plate_name_dict.keys())
    
    return_dict = {}
    return_dict["steel_coated"] = sorted([name for name in plate_name_list if name.lower().startswith("s")])
    return_dict["steel_uncoated"] = sorted([name for name in plate_name_list if name.lower().startswith("sa")])
    return_dict["aluminum"] = sorted([name for name in plate_name_list if name.lower().startswith("a")])
    return return_dict

def get_plate_thickness_list(path: str) -> list:
    data = read_json(path)
    plate_thickness_list = data["plate_thickness_list"]
    return sorted(plate_thickness_list)

def recommend_attaching_method(plate_conditions:dict) -> dict:
    if _is_steel_spot_welding(plate_conditions):
        return _build_recommendation_dict("ssw", "ssw", "ssw")
    # return _recommendation_model_output(plate_conditions)
    return _get_dummy_recommendation(plate_conditions)

def _get_dummy_recommendation(plate_conditions):
    return _build_recommendation_dict("spr", "ssw", "ssw")

def _recommendation_model_output(plate_conditions:dict) -> dict:
    # model logic here   
    pass

def _is_steel_spot_welding(plate_conditions:dict) -> bool:
    
    no_value = "no-value"
    
    top_name = plate_conditions["top_name"]
    middle_name = plate_conditions["middle_name"]
    bottom_name = plate_conditions["bottom_name"]
        
    is_top_steel = top_name.lower().startswith("s")
    is_middle_steel = middle_name.lower().startswith("s")
    is_bottom_steel = bottom_name.lower().startswith("s")
    
    return (
        is_top_steel and is_bottom_steel 
        if middle_name == no_value else
        is_top_steel and is_middle_steel and is_bottom_steel
    )
    
def _build_recommendation_dict(first:str, second:str, third:str) -> dict:
    return {
        "first": first,
        "second": second,
        "third": third
    }