from django.db import models
from django import forms

from .utils import util_gen

# TODO: PATH to be derived from env
PATH = "./django_app/utils/metadata.csv"
OP = util_gen.GenOptionProducer(PATH)

OPTION_RIVET = OP.get_condition_options_rivet()
OPTION_DIE = OP.get_condition_options_die()
OPTION_UPPER_TYPE = OP.get_condition_options_upper_type()
OPTION_MIDDLE_TYPE = OP.get_condition_options_middle_type()
OPTION_LOWER_TYPE = OP.get_condition_options_lower_type()
    

class GenCondtidionForm(forms.Form):

    plate_count = forms.ChoiceField(widget=forms.RadioSelect, choices=[("2겹", "2겹"), ("3겹", "3겹")])
    이미지_장수 = forms.IntegerField()
    rivet = forms.ChoiceField(choices=OPTION_RIVET)
    die = forms.ChoiceField(choices=OPTION_DIE)
    head_height = forms.FloatField()
    upper_type = forms.ChoiceField(choices=OPTION_UPPER_TYPE)
    upper_thickness = forms.FloatField()
    middle_type = forms.ChoiceField(choices=OPTION_MIDDLE_TYPE)
    middle_thickness = forms.FloatField()
    lower_type = forms.ChoiceField(choices=OPTION_LOWER_TYPE)
    lower_thickness = forms.FloatField()