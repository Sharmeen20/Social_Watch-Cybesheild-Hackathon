from django import forms

class PredictionForm(forms.Form):
    feature1 = forms.CharField(label="Enter Text")