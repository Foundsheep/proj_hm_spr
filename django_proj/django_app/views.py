from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

from PIL import Image
import numpy as np
import traceback

from .utils import util_gen

# TODO: PATH to be derived from env
PATH = r"C:\Users\msi\Desktop\workspace\001_HM_SPR\00_src\proj_hm_spr\django_proj\django_app\utils\metadata.csv"
op = util_gen.GenOptionProducer(PATH)
OPTION_UPPER_TYPE = op.get_condition_options_upper_type()
print("upper_type")

def index(req):
    context = {}
    return render(req, "index.html", context)

def seg(req):
    context = {}
    return render(req, "seg.html", context)

def gen(req):
    context = {"upper_type": OPTION_UPPER_TYPE}
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