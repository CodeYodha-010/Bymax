from django import forms

class UploadFileForm(forms.Form):
    file = forms.FileField(label='Upload CSV file')

class AskQuestionForm(forms.Form):
    question = forms.CharField(label='Ask a question about the table')