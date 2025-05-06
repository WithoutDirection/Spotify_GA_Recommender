from django import forms
from .models import UserRating, Track

class TrackRatingForm(forms.ModelForm):
    class Meta:
        model = UserRating
        fields = ['rating']
        widgets = {
            'rating': forms.RadioSelect(choices=[(1, '1'), (2, '2'), (3, '3'), (4, '4'), (5, '5')])
        }

class InitialRatingForm(forms.Form):
    """Form for initial track ratings"""
    
    def __init__(self, *args, tracks=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # If tracks were provided, create a field for each track
        if tracks:
            for track in tracks:
                field_name = f'track_{track.track_id}'
                self.fields[field_name] = forms.ChoiceField(
                    label=f"{track.artist_name} - {track.track_name}",
                    choices=[('1', '1'), ('2', '2'), ('3', '3'), ('4', '4'), ('5', '5')],
                    widget=forms.RadioSelect(),
                    required=True
                )
        
        # Add debugging to see what's happening with form validation
        print(f"Form initialized with fields: {list(self.fields.keys())}")
    
    def clean(self):
        cleaned_data = super().clean()
        print(f"Form data submitted: {self.data}")
        print(f"Form errors: {self.errors}")
        return cleaned_data

class GenerationRatingForm(forms.Form):
    def __init__(self, *args, **kwargs):
        recommendations = kwargs.pop('recommendations', None)
        super().__init__(*args, **kwargs)
        
        if recommendations:
            for i, rec in enumerate(recommendations):
                self.fields[f'rec_{i}'] = forms.ChoiceField(
                    label=f"{rec['artist_name']} - {rec['track_name']}",
                    choices=[(1, '1'), (2, '2'), (3, '3'), (4, '4'), (5, '5')],
                    widget=forms.RadioSelect,
                    required=True
                )
        
        self.fields['satisfied'] = forms.BooleanField(
            label="Are you satisfied with these recommendations?",
            required=False
        )