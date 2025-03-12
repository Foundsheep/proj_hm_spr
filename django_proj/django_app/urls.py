from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("method-result/", views.index_result, name="index-result"),
    path("seg/", views.seg, name="seg"),
    path("gen/", views.gen, name="gen"),
    path("steel-spot-welding/", views.steel_spot_welding, name="steel-spot_welding"),
    path("steel-spot-welding-detail/", views.steel_spot_welding_detail, name="steel-spot_welding_detail"),
    path("process-segmentation/", views.process_segmentation, name="process_segmentation"),
]