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