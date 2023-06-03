from django.forms import forms, IntegerField, CharField, Textarea


class TestForm(forms.Form):
    text = CharField(widget=Textarea(attrs={'rows': 5}))
