from django.urls import path

from . import views

urlpatterns = [
    path("", views.page_index, name="index"),
    path("method-result/", views.page_index_result, name="index-result"),
    path("seg/", views.page_seg_main, name="seg"),
    path("gen/", views.page_gen_main, name="gen"),
    path("steel-spot-welding/", views.page_ssw_main, name="steel-spot_welding"),
    path("steel-spot-welding-detail/", views.page_ssw_detail, name="steel-spot_welding_detail"),
    path("api/segment/", views.api_process_segmentation, name="process_segmentation"),
]