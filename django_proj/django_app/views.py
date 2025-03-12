from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponse
from .apps import DjangoAppConfig
from .models import *
from .utils.util_gen import generate_image, convert_image_to_base64
import torch

from PIL import Image
import numpy as np
import traceback

def index(req):
    context = {}
    return render(req, "index.html", context)

def index_result(req):
    context = {}
    return render(req, "method-result.html", context)

def steel_spot_welding(req):
    context = {}
    return render(req, "steel-spot-welding.html", context)

def steel_spot_welding_detail(req):
    context = {}
    return render(req, "steel-spot-welding-detail.html", context)
    
# @csrf_exempt
def seg(req):
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

def gen(req):
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

def process_segmentation(req):
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
