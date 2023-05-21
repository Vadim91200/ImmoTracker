from django import forms

class DepartementsForm(forms.Form):
    departement = forms.CharField(label='departement', max_length=100)