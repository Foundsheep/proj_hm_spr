from django.shortcuts import render

# Create your views here.
def index(req):
    context = {}
    return render(req, "index.html", context)

def seg(req):
    context = {}
    return render(req, "seg.html", context)

def gen(req):
    context = {}
    return render(req, "gen.html", context)