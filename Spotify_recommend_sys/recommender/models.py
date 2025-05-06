from django.db import models
from django.contrib.auth.models import User

class Track(models.Model):
    track_id = models.CharField(max_length=255, primary_key=True)
    artist_name = models.CharField(max_length=255)
    track_name = models.CharField(max_length=255)
    acousticness = models.FloatField()
    danceability = models.FloatField()
    energy = models.FloatField()
    instrumentalness = models.FloatField()
    key = models.FloatField()
    liveness = models.FloatField()
    loudness = models.FloatField()
    mode = models.FloatField()
    speechiness = models.FloatField()
    tempo = models.FloatField()
    time_signature = models.FloatField()
    valence = models.FloatField()
    cluster = models.IntegerField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.artist_name} - {self.track_name}"


class UserRating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE) # ForeignKey to User model
    track = models.ForeignKey(Track, on_delete=models.CASCADE) # ForeignKey to Track model, on_delete=models.CASCADE to delete ratings if the track is deleted
    rating = models.FloatField(choices=[(1, '1'), (2, '2'), (3, '3'), (4, '4'), (5, '5')])
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('user', 'track')
    
    def __str__(self):
        return f"{self.user.username} - {self.track} - {self.rating}"

class Recommendation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    track = models.ForeignKey(Track, on_delete=models.CASCADE)
    score = models.FloatField(null=True, blank=True)
    recommended_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('user', 'track', 'recommended_at')
    
    def __str__(self):
        return f"{self.user.username} - {self.track}"

