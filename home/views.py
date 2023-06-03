from django.http import HttpRequest
from django.shortcuts import render

from home.forms import TestForm


# Create your views here.


def home_index(req: HttpRequest):
    form = TestForm(req.POST or None)
    result = {}
    if req.method == 'POST':
        print("isop")
        result['data'] = '100%'
    return render(req, 'home/index.html', {
        'form': form,
        'result': result
    })


def model_reinstall(req: HttpRequest):
    if req.method == 'POST':
        print("post bosilib ketdi")

    return render(req, 'home/reinstall.html')
