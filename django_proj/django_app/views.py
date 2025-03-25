from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponse
from .apps import DjangoAppConfig
from .models import *
from .utils.util_gen import generate_image, convert_image_to_base64
from .utils.util_index import get_plate_name_list_in_dict, get_plate_thickness_list, recommend_attaching_method
import torch

from PIL import Image
import numpy as np
import traceback
import json

def page_index(req):
    context = {}
    
    # update the context
    plate_name_dict = get_plate_name_list_in_dict(DjangoAppConfig.PLATE_DICT_PATH)    
    plate_thickness_list = get_plate_thickness_list(DjangoAppConfig.PLATE_DICT_PATH)
    context.update({"plate_name_dict": json.dumps(plate_name_dict)})
    context.update({"plate_thickness_list": plate_thickness_list})
    return render(req, "index.html", context)

def page_index_result(req):
    context = {}
    if req.method == "POST":
        form_data = req.POST
        recommendation_dict = recommend_attaching_method(form_data)
        
        # context update
        context.update(recommendation_dict)
        context.update({"previous_data": form_data})
    return render(req, "method-result.html", context)

def page_ssw_main(req):
    print(req.POST)
    context = {}
    return render(req, "steel-spot-welding.html", context)

def page_ssw_detail(req):
    context = {}
    print(req.POST)
    return render(req, "steel-spot-welding-detail.html", context)
    
@csrf_exempt
def page_seg_main(req):
    context = {}
    if req.method == "POST":
        if "images" not in req.FILES:
            return JsonResponse({"error": "No images uploaded"}, status=400)
        try:
            images = req.FILES.getlist("images")
            images = [Image.open(img).convert("RGB") for img in images]
            
            shapes = []
            shapes.append(len(images))            
            context.update({"shapes": shapes})
            # print("here")
            # shapes = [np.array(img).shape for img in images]
            img_bytes = convert_image_to_base64(images[0])
            context.update({"first_image": img_bytes})
            print(img_bytes)
            return render(req, "seg.html", context)
        except Exception as e:
            print("error occured")
            print(e)
            return JsonResponse({"code": "failure"})

    return render(req, "seg.html", context)

def page_gen_main(req):
    context = {
        "rivet": OPTION_RIVET,
        "die": OPTION_DIE,
        "upper_type": OPTION_UPPER_TYPE,
        "middle_type": OPTION_MIDDLE_TYPE,
        "lower_type": OPTION_LOWER_TYPE,
    }
    
    if req.method == "POST":
        form = GenCondtidionForm(req.POST)
        if form.is_valid():
            data = form.cleaned_data
            print(data)
            generated_images = generate_image(data)
            converted_image = convert_image_to_base64(generated_images)
            context.update({"generated_images": converted_image})
            flag = True
            context.update({"flag": flag})
    else:
        form = GenCondtidionForm()
        
    context.update({"form": form})
    return render(req, "gen.html", context)

def api_process_segmentation(req):
    context = {}
    if req.method == "POST":
        if "images" not in req.FILES:
            return JsonResponse({"result": "No images uploaded"}, status=400)
        try:
            images = req.FILES.getlist("images")
            images = [Image.open(img).convert("RGB") for img in images]
            
            shapes = []
            shapes.append(len(images))            
            context.update({"shapes": shapes})
            # shapes = [np.array(img).shape for img in images]
            img_bytes = convert_image_to_base64(images[0])
            context.update({"first_image": img_bytes})
            print(img_bytes)
            # return render(req, "seg.html", context)
        except Exception as e:
            print("error occured")
            print(e)
            img_bytes = "failure"
    return JsonResponse({"result": img_bytes})
    # return render(req, "seg.html", context)
