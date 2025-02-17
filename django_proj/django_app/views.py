from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponse
from .apps import DjangoAppConfig
from .models import *
import torch

from PIL import Image
import numpy as np
import traceback

def index(req):
    context = {}
    return render(req, "index.html", context)

def seg(req):
    context = {}
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
    else:
        form = GenCondtidionForm()
        
    context.update({"form": form})
    return render(req, "gen.html", context)

@csrf_exempt
def api_process_segmentation(request):
    if request.method == "POST":
        if "images" not in request.FILES:
            return JsonResponse({"error": "No images uploaded"}, status=400)
        
        try:
            images = request.FILES.getlist("images")
            images = [Image.open(img).convert("RGB") for img in images]
            
            print(np.array(images).shape)
        except Exception as e:
            print("error occured")
            traceback.print_tb(e)
            return JsonResponse({"code": "failure"})    
    return JsonResponse({"code": "success"})


def api_process_generation(req):
    print(req)
    if req.method == "POST":
        # model = DjangoAppConfig.gen_model
        # transforms = DjangoAppConfig.transforms
        
        # plate_count = transforms["plate_count"](3)
        # rivet = transforms["rivet"]("AB5.5X5.5")
        # die = transforms["die"]("D1020")
        # upper_type = transforms["upper_type"]("SABC1470")
        # upper_thickness = transforms["upper_thickness"](1.)
        # middle_type = transforms["middle_type"]("SGAFC590")
        # middle_thickness = transforms["middle_thickness"](0.7)
        # lower_type = transforms["lower_type"]("A6N01")
        # lower_thickness = transforms["lower_thickness"](3.)
        # head_height = transforms["head_height"](0.230122)
        
        # categorical_conds = (
        #     torch.stack([
        #         rivet, die, upper_type, middle_type, lower_type
        #     ]).to(device=DjangoAppConfig.DEVICE)
        # )
        
        # continuous_conds = (
        #     torch.stack([
        #         plate_count, upper_thickness, middle_thickness, lower_thickness, head_height
        #     ]).to(device=DjangoAppConfig.DEVICE)
        # )
        
        # with torch.no_grad():
        #     model.eval()
        #     out = model(
        #         batch_size=1,
        #         categorical_conds=categorical_conds,
        #         continuous_conds=continuous_conds
        #     )
        print("************* DONE *************")
    
    return JsonResponse({"result": "success"})
        