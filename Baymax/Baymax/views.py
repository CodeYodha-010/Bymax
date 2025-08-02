from django.shortcuts import render

def baymax_view(request):
    return render(request, 'baymax.html')
