from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.http import JsonResponse
from django.contrib import messages
from django.db.models import Avg
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.views import generic
import random
import json
import logging

from .models import Track, UserRating, Recommendation
from .forms import TrackRatingForm, InitialRatingForm, GenerationRatingForm
from .recommender_system import SpotifyRecommender

# Set up logging
logger = logging.getLogger(__name__)

# Initialize the recommender system as a singleton
recommender = None

class SignUpView(generic.CreateView):
    """
    View for user registration.
    Uses Django's built-in UserCreationForm for user registration.
    """
    form_class = UserCreationForm
    success_url = reverse_lazy('login')
    template_name = 'registration/signup.html'
    
    def form_valid(self, form):
        """
        Override form_valid to automatically log in user after successful registration.
        """
        valid = super().form_valid(form)
        # Auto-login after registration
        username = form.cleaned_data.get('username')
        password = form.cleaned_data.get('password1')
        user = authenticate(username=username, password=password)
        if user is not None:
            login(self.request, user)
            messages.success(self.request, f"Welcome, {username}! Your account has been created successfully.")
            return redirect('home')
        return valid

def logout_view(request):
    """
    View for handling user logout.
    """
    logout(request)
    messages.success(request, "You have been successfully logged out.")
    return redirect('home')

def get_recommender():
    """
    Get or initialize the SpotifyRecommender singleton.
    
    Returns:
        SpotifyRecommender: The recommender instance
    """
    global recommender
    if recommender is None:
        recommender = SpotifyRecommender()
        # Make sure tracks are clustered
        if not Track.objects.filter(cluster__isnull=False).exists():
            recommender.cluster_tracks()
    return recommender


def home(request):
    """Home page view"""
    get_recommender()
    return render(request, 'recommender/home.html')

def guest_page(request):
    """
    View for users who haven't registered yet.
    Provides more information about the music recommendation system.
    """
    return render(request, 'recommender/guest.html')

@login_required
def initial_rating(request):
    """View for the initial rating of sample tracks"""
    recommender = get_recommender()
    
    if request.method == 'POST':
        return _process_initial_ratings(request, recommender)
    else:
        return _display_sample_tracks(request, recommender)


def _process_initial_ratings(request, recommender):
    """
    Process submitted initial ratings.
    
    Args:
        request: The HTTP request
        recommender: The recommender instance
    
    Returns:
        HttpResponse: Redirect to next step or error response
    """
    # Get track IDs that were displayed in the form
    track_ids = request.session.get('sample_track_ids', [])
    
    # Get the actual Track objects
    sample_tracks = Track.objects.filter(track_id__in=track_ids)
    
    logger.debug("Processing initial ratings POST data")
    
    errors = {}
    user_ratings = {}
    
    # Process ratings for each track
    for track in sample_tracks:
        field_name = f'track_{track.track_id}'
        rating_value = request.POST.get(field_name)
        
        if not rating_value:
            errors[field_name] = ["This rating is required."]
        else:
            try:
                rating_float = float(rating_value)
                if rating_float < 1 or rating_float > 5:
                    errors[field_name] = ["Rating must be between 1 and 5."]
                else:
                    # Valid rating, save to user_ratings and database
                    user_ratings[track.track_id] = rating_float
                    UserRating.objects.update_or_create(
                        user=request.user,
                        track=track,
                        defaults={'rating': rating_float}
                    )
            except ValueError:
                errors[field_name] = ["Rating must be a number."]
    
    # If no errors, proceed to next step
    if not errors:
        logger.info(f"User {request.user.id} submitted {len(user_ratings)} valid ratings")
        
        # Store ratings in session for next step
        request.session['user_ratings'] = user_ratings
        
        # Clear sample track IDs from session
        request.session.pop('sample_track_ids', None)
        request.session.modified = True
        
        return redirect('start_recommendation')
    else:
        # Show errors for invalid ratings
        logger.warning(f"User {request.user.id} submitted ratings with errors: {errors}")
        for field_name, error_list in errors.items():
            messages.error(request, f"{field_name}: {error_list[0]}")
        
        # Re-display the form with errors
        return render(request, 'recommender/rate_tracks.html', {
            'tracks': sample_tracks,
            'step': 'initial'
        })


def _display_sample_tracks(request, recommender):
    """
    Display sample tracks for initial rating.
    
    Args:
        request: The HTTP request
        recommender: The recommender instance
    
    Returns:
        HttpResponse: Rendered template with sample tracks
    """
    # Get one sample track from each cluster
    sample_tracks_dict = recommender.get_sample_tracks(n_per_cluster=1)
    
    # Convert to Track objects and save track IDs
    sample_tracks = []
    track_ids = []
    
    for track_dict in sample_tracks_dict:
        track_id = track_dict['track_id']
        track, created = Track.objects.get_or_create(
            track_id=track_id,
            defaults={
                'artist_name': track_dict.get('artist_name', ''),
                'track_name': track_dict.get('track_name', ''),
                'acousticness': track_dict.get('acousticness', 0),
                'danceability': track_dict.get('danceability', 0),
                'energy': track_dict.get('energy', 0),
                'instrumentalness': track_dict.get('instrumentalness', 0),
                'key': track_dict.get('key', 0),
                'liveness': track_dict.get('liveness', 0),
                'loudness': track_dict.get('loudness', 0),
                'mode': track_dict.get('mode', 0),
                'speechiness': track_dict.get('speechiness', 0),
                'tempo': track_dict.get('tempo', 0),
                'time_signature': track_dict.get('time_signature', 0),
                'valence': track_dict.get('valence', 0),
                'cluster': track_dict.get('cluster')
            }
        )
        sample_tracks.append(track)
        track_ids.append(track_id)
    
    # Save track IDs in session for consistency between GET and POST
    request.session['sample_track_ids'] = track_ids
    request.session.modified = True
    
    logger.info(f"Displaying {len(sample_tracks)} sample tracks for user {request.user.id}")
    
    return render(request, 'recommender/rate_tracks.html', {
        'tracks': sample_tracks,
        'step': 'initial'
    })


@login_required
def start_recommendation(request):
    """Start the recommendation process"""
    recommender = get_recommender()
    
    # Get user ratings from session or database
    user_ratings = request.session.get('user_ratings', {})
    
    if not user_ratings:
        # Get from database if not in session
        user_ratings_db = UserRating.objects.filter(user=request.user)
        user_ratings = {rating.track.track_id: rating.rating for rating in user_ratings_db}
        logger.info(f"Retrieved {len(user_ratings)} ratings from database for user {request.user.id}")
    
    if not user_ratings:
        messages.error(request, "No ratings found. Please rate some tracks first.")
        return redirect('initial_rating')
    
    # Store current generation in session
    request.session['current_generation'] = 0
    
    try:
        # Get initial recommendations using FCB-RS
        initial_recommendations = recommender.fcb_rs(user_ratings, top_n=recommender.population_size)
        logger.info(f"Generated {len(initial_recommendations)} initial recommendations for user {request.user.id}")
        
        # Store recommendations in session
        request.session['current_recommendations'] = initial_recommendations
        request.session.modified = True
        
        return redirect('rate_recommendations')
    except Exception as e:
        logger.error(f"Error generating recommendations for user {request.user.id}: {str(e)}")
        messages.error(request, f"Error generating recommendations: {str(e)}")
        return redirect('initial_rating')


@login_required
def rate_recommendations(request):
    """View for rating recommendations during the IGA process"""
    recommender = get_recommender()
    
    # Get current generation and recommendations from session
    current_generation = request.session.get('current_generation', 0)
    current_recommendations = request.session.get('current_recommendations', [])
    
    if not current_recommendations:
        messages.warning(request, "No recommendations available. Please start over.")
        return redirect('initial_rating')
    
    logger.debug(f"Generation {current_generation}: {len(current_recommendations)} recommendations")
    
    if request.method == 'POST':
        return _process_recommendation_ratings(request, recommender, current_generation, current_recommendations)
    else:
        # Display form for rating current recommendations
        form = GenerationRatingForm(recommendations=current_recommendations)
        
        return render(request, 'recommender/rate_recommendations.html', {
            'form': form,
            'recommendations': current_recommendations,
            'current_generation': current_generation + 1,
            'total_generations': recommender.generations
        })


def _process_recommendation_ratings(request, recommender, current_generation, current_recommendations):
    """
    Process ratings for the current generation of recommendations.
    
    Args:
        request: The HTTP request
        recommender: The recommender instance
        current_generation: Current generation number
        current_recommendations: Current list of recommendations
    
    Returns:
        HttpResponse: Redirect to next step or error response
    """
    form = GenerationRatingForm(request.POST, recommendations=current_recommendations)
    
    if not form.is_valid():
        messages.error(request, "Please provide all ratings.")
        return render(request, 'recommender/rate_recommendations.html', {
            'form': form,
            'recommendations': current_recommendations,
            'current_generation': current_generation + 1,
            'total_generations': recommender.generations
        })
    
    # Process ratings
    user_ratings = {}
    satisfied = form.cleaned_data.get('satisfied', False)
    
    # Extract and save ratings
    for field_name, rating_value in form.cleaned_data.items():
        if field_name.startswith('rec_'):
            idx = int(field_name.replace('rec_', ''))
            if idx < len(current_recommendations):
                rec = current_recommendations[idx]
                _save_track_rating(request.user, rec, rating_value)
                user_ratings[idx] = float(rating_value)
    
    # If user is satisfied or reached max generations, finalize recommendations
    if satisfied or current_generation >= recommender.generations - 1:
        return _finalize_recommendations(request, current_recommendations)
    
    # Otherwise, generate next generation
    try:
        next_recommendations = _generate_next_generation(recommender, current_recommendations, user_ratings)
        
        # Update session with next generation
        request.session['current_recommendations'] = next_recommendations
        request.session['current_generation'] = current_generation + 1
        request.session.modified = True
        
        logger.info(f"Generated generation {current_generation + 1} for user {request.user.id}")
        
        return redirect('rate_recommendations')
    except Exception as e:
        logger.error(f"Error generating next generation: {str(e)}")
        messages.error(request, f"Error generating next generation: {str(e)}")
        return redirect('final_recommendations')


def _save_track_rating(user, recommendation, rating_value):
    """
    Save a user's rating for a track.
    
    Args:
        user: User object
        recommendation: Recommendation dictionary
        rating_value: Rating value
    """
    track_id = recommendation['track_id']
    
    # Get or create track
    track, created = Track.objects.get_or_create(
        track_id=track_id,
        defaults={
            'artist_name': recommendation.get('artist_name', ''),
            'track_name': recommendation.get('track_name', ''),
            'acousticness': recommendation.get('acousticness', 0),
            'danceability': recommendation.get('danceability', 0),
            'energy': recommendation.get('energy', 0),
            'instrumentalness': recommendation.get('instrumentalness', 0),
            'key': recommendation.get('key', 0),
            'liveness': recommendation.get('liveness', 0),
            'loudness': recommendation.get('loudness', 0),
            'mode': recommendation.get('mode', 0),
            'speechiness': recommendation.get('speechiness', 0),
            'tempo': recommendation.get('tempo', 0),
            'time_signature': recommendation.get('time_signature', 0),
            'valence': recommendation.get('valence', 0)
        }
    )
    
    # Save user rating
    UserRating.objects.update_or_create(
        user=user,
        track=track,
        defaults={'rating': float(rating_value)}
    )


def _finalize_recommendations(request, recommendations):
    """
    Save final recommendations and clean up session.
    
    Args:
        request: The HTTP request
        recommendations: List of recommendation dictionaries
    
    Returns:
        HttpResponse: Redirect to final recommendations page
    """
    # Save recommendations to database
    for rec in recommendations:
        track, created = Track.objects.get_or_create(
            track_id=rec['track_id'],
            defaults={
                'artist_name': rec.get('artist_name', ''),
                'track_name': rec.get('track_name', ''),
                'acousticness': rec.get('acousticness', 0),
                'danceability': rec.get('danceability', 0),
                'energy': rec.get('energy', 0),
                'instrumentalness': rec.get('instrumentalness', 0),
                'key': rec.get('key', 0),
                'liveness': rec.get('liveness', 0),
                'loudness': rec.get('loudness', 0),
                'mode': rec.get('mode', 0),
                'speechiness': rec.get('speechiness', 0),
                'tempo': rec.get('tempo', 0),
                'time_signature': rec.get('time_signature', 0),
                'valence': rec.get('valence', 0)
            }
        )
        
        Recommendation.objects.create(
            user=request.user,
            track=track,
            score=rec.get('score', None)
        )
    
    # Clear session data
    for key in ['user_ratings', 'current_generation', 'current_recommendations', 'evolution_history']:
        request.session.pop(key, None)
    request.session.modified = True
    
    logger.info(f"Finalized {len(recommendations)} recommendations for user {request.user.id}")
    
    return redirect('final_recommendations')


def _generate_next_generation(recommender, current_recommendations, user_ratings):
    """
    Generate the next generation of recommendations using the genetic algorithm.
    
    Args:
        recommender: The recommender instance
        current_recommendations: Current list of recommendations
        user_ratings: Dictionary of user ratings
    
    Returns:
        List of recommendation dictionaries for the next generation
    """
    # Get initial population from current recommendations
    initial_population = []
    for rec in current_recommendations:
        track_data = recommender.data[recommender.data['track_id'] == rec['track_id']]
        if not track_data.empty:
            track = track_data.iloc[0].to_dict()
            initial_population.append(track)
        else:
            # If track not found in data, use what we have
            initial_population.append(rec)
    
    # Apply selection - get elite tracks
    sorted_indices = sorted(
        range(len(initial_population)), 
        key=lambda i: user_ratings.get(i, 0), 
        reverse=True
    )
    elite_size = min(recommender.elite_size, len(initial_population))
    elite = [initial_population[i] for i in sorted_indices[:elite_size]]
    
    # Create chromosomes from current population
    chromosomes = [recommender.chromosome_representation(track) for track in initial_population]
    
    # Generate offspring through crossover
    offspring = []
    while len(offspring) < recommender.population_size - elite_size:
        # Select parents from top half of population
        top_half = max(2, len(sorted_indices) // 2)
        parent1_idx = random.randint(0, top_half - 1)
        parent2_idx = random.randint(0, top_half - 1)
        
        # Ensure different parents
        while parent2_idx == parent1_idx:
            parent2_idx = random.randint(0, top_half - 1)
        
        parent1 = chromosomes[sorted_indices[parent1_idx]]
        parent2 = chromosomes[sorted_indices[parent2_idx]]
        
        # Apply crossover with probability
        if random.random() < recommender.crossover_rate:
            child_chromosome = recommender.blx_alpha_crossover(parent1, parent2)
            
            # Find the most similar actual track
            child_track = recommender.find_similar_track(child_chromosome)
            if child_track is not None:
                offspring.append(child_track)
    
    # Create new population with elite and offspring
    next_population = elite + offspring
    
    # Convert to recommendations format
    next_recommendations = []
    for track in next_population:
        next_recommendations.append({
            'track_id': track['track_id'],
            'artist_name': track.get('artist_name', ''),
            'track_name': track.get('track_name', ''),
            'score': None  # Score not relevant in IGA generations
        })
    
    return next_recommendations


@login_required
def final_recommendations(request):
    """View for showing final recommendations"""
    # Get latest recommendations for user
    recommendations = Recommendation.objects.filter(user=request.user).order_by('-recommended_at')[:10]
    
    return render(request, 'recommender/final_recommendations.html', {
        'recommendations': recommendations
    })


@login_required
def user_history(request):
    """View for showing user's rating history"""
    user_ratings = UserRating.objects.filter(user=request.user).order_by('-created_at')
    
    return render(request, 'recommender/user_history.html', {
        'user_ratings': user_ratings
    })