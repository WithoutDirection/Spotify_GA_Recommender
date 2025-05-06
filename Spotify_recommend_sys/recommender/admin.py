from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Track, UserRating, Recommendation

@admin.register(Track)
class TrackAdmin(admin.ModelAdmin):
    list_display = ('track_name', 'artist_name', 'cluster')
    search_fields = ('track_name', 'artist_name', 'track_id')
    list_filter = ('cluster',)

@admin.register(UserRating)
class UserRatingAdmin(admin.ModelAdmin):
    list_display = ('user', 'track', 'rating', 'created_at')
    list_filter = ('rating', 'created_at')
    search_fields = ('user__username', 'track__track_name')

@admin.register(Recommendation)
class RecommendationAdmin(admin.ModelAdmin):
    list_display = ('user', 'track', 'score', 'recommended_at')
    list_filter = ('recommended_at',)
    search_fields = ('user__username', 'track__track_name')

