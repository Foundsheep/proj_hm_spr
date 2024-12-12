from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("seg/", views.seg, name="seg"),
    path("gen/", views.gen, name="gen"),
    path("api/process-segmentation/", views.api_process_segmentation, name="process_segmentation")
]