from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.http import JsonResponse
from django.contrib import messages
from django.db.models import Avg
import random
import json
import logging

from .models import Track, UserRating, Recommendation
from .forms import TrackRatingForm, InitialRatingForm, GenerationRatingForm
from .recommender_system import SpotifyRecommender

# Initialize the recommender system
recommender = None

def initialize_recommender():
    global recommender
    if recommender is None:
        recommender = SpotifyRecommender()
        # Make sure tracks are clustered
        if not Track.objects.filter(cluster__isnull=False).exists():
            recommender.cluster_tracks()
    return recommender

def home(request):
    initialize_recommender()
    return render(request, 'recommender/home.html')


@login_required
def initial_rating(request):
    """View for initial rating of tracks"""
    initialize_recommender()
    
    if request.method == 'POST':
        # Get track IDs from session that were displayed in the form
        track_ids = request.session.get('sample_track_ids', [])
        
        # Get the actual Track objects based on those IDs
        sample_tracks = Track.objects.filter(track_id__in=track_ids)
        
        # Debug: Print all POST data
        print("POST data received:")
        for key, value in request.POST.items():
            print(f"{key}: {value}")
        
        errors = {}
        user_ratings = {}
        
        # Process ratings for each track
        for track in sample_tracks:
            field_name = f'track_{track.track_id}'
            rating_value = request.POST.get(field_name)
            print(f"Rating for {field_name}: {rating_value}")
            
            if not rating_value:
                errors[field_name] = ["This rating is required."]
            else:
                try:
                    # Validate the rating is a number between 1 and 5
                    rating_float = float(rating_value)
                    if rating_float < 1 or rating_float > 5:
                        errors[field_name] = ["Rating must be between 1 and 5."]
                    else:
                        # Valid rating, add to user_ratings
                        user_ratings[track.track_id] = rating_float
                        
                        # Also save to database
                        UserRating.objects.update_or_create(
                            user=request.user,
                            track=track,
                            defaults={'rating': rating_float}
                        )
                except ValueError:
                    errors[field_name] = ["Rating must be a number."]
        
        # If no errors, proceed to next step
        if not errors:
            print(f"All ratings valid. User ratings: {user_ratings}")
            
            # Store ratings in session for next step
            request.session['user_ratings'] = user_ratings
            request.session.modified = True
            
            # Clear the sample track IDs from session to prevent reuse
            if 'sample_track_ids' in request.session:
                del request.session['sample_track_ids']
            
            # Redirect to recommendations page
            return redirect('start_recommendation')
        else:
            print(f"Rating errors: {errors}")
            for field_name, error_list in errors.items():
                messages.error(request, f"{field_name}: {error_list[0]}")
    else:
        # For GET requests, generate new sample tracks
        # Get one sample track from each cluster
        sample_tracks_dict = recommender.get_sample_tracks(n_per_cluster=1)
        
        # Convert to Track objects
        sample_tracks = []
        track_ids = []  # Store track IDs to save in session
        
        for track_dict in sample_tracks_dict:
            track, created = Track.objects.get_or_create(
                track_id=track_dict['track_id'],
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
            track_ids.append(track.track_id)
        
        # Save track IDs in session to ensure consistency between GET and POST
        request.session['sample_track_ids'] = track_ids
        request.session.modified = True
        
        print(f'sample_tracks: {track_ids}')
    
    # Render the form with sample tracks
    return render(request, 'recommender/rate_tracks.html', {
        'tracks': sample_tracks,
        'step': 'initial'
    })
@login_required
def start_recommendation(request):
    """Start the recommendation process"""
    initialize_recommender()
    
    # Get user ratings from session or database
    user_ratings = request.session.get('user_ratings', {})
    print(f"Retrieved user ratings from session: {user_ratings}")
    
    if not user_ratings:
        # Get from database if not in session
        print("No ratings in session, trying to get from database...")
        user_ratings_db = UserRating.objects.filter(user=request.user)
        user_ratings = {rating.track.track_id: rating.rating for rating in user_ratings_db}
        print(f"Retrieved {len(user_ratings)} ratings from database")
    
    if not user_ratings:
        messages.error(request, "No ratings found. Please rate some tracks first.")
        return redirect('initial_rating')
    
    # Store current generation in session
    request.session['current_generation'] = 0
    
    try:
        # Get initial recommendations using FCB-RS
        initial_recommendations = recommender.fcb_rs(user_ratings, top_n=recommender.population_size)
        print(f'Generated initial recommendations: {len(initial_recommendations)}')
        
        # Store recommendations in session
        request.session['current_recommendations'] = initial_recommendations
        request.session.modified = True
        
        # Redirect to rate recommendations
        return redirect('rate_recommendations')
    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")
        messages.error(request, f"Error generating recommendations: {str(e)}")
        return redirect('initial_rating')

@login_required
def rate_recommendations(request):
    """View for rating recommendations (IGA)"""
    initialize_recommender()
    
    # Get current generation and recommendations from session
    current_generation = request.session.get('current_generation', 0)
    current_recommendations = request.session.get('current_recommendations', [])
    
    print(f"Current generation: {current_generation}")
    print(f"Number of recommendations: {len(current_recommendations)}")
    
    if not current_recommendations:
        messages.warning(request, "No recommendations available. Please start over.")
        return redirect('initial_rating')
    
    if request.method == 'POST':
        form = GenerationRatingForm(request.POST, recommendations=current_recommendations)
        print(f"Form data submitted: {request.POST}")
        if form.is_valid():
            # Process ratings
            user_ratings = {}
            satisfied = form.cleaned_data.get('satisfied', False)
            
            for field_name, rating_value in form.cleaned_data.items():
                if field_name.startswith('rec_'):
                    # Extract index from field name
                    idx = int(field_name.replace('rec_', ''))
                    if idx < len(current_recommendations):
                        rec = current_recommendations[idx]
                        track_id = rec['track_id']
                        
                        # Get or create track
                        track, created = Track.objects.get_or_create(
                            track_id=track_id,
                            defaults={
                                'artist_name': rec.get('artist_name', ''),
                                'track_name': rec.get('track_name', ''),
                                # Add default values for features if not in rec
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
                        
                        # Save rating
                        UserRating.objects.update_or_create(
                            user=request.user,
                            track=track,
                            defaults={'rating': float(rating_value)}
                        )
                        
                        # Add to user_ratings dict for IGA
                        user_ratings[idx] = float(rating_value)
            
            # If user is satisfied or reached max generations, show final recommendations
            if satisfied or current_generation >= recommender.generations - 1:
                # Save these as final recommendations
                for rec in current_recommendations:
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
                if 'user_ratings' in request.session:
                    del request.session['user_ratings']
                if 'current_generation' in request.session:
                    del request.session['current_generation']
                if 'current_recommendations' in request.session:
                    del request.session['current_recommendations']
                if 'evolution_history' in request.session:
                    del request.session['evolution_history']
                
                # Save changes to session
                request.session.modified = True
                
                return redirect('final_recommendations')
            
            # Otherwise, continue with next generation
            try:
                # Simulate user feedback for IGA
                def user_feedback(population, gen):
                    return user_ratings
                
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
                
                # Get next generation from IGA
                # (This is a simplified version since we already have the ratings)
                
                # Select parents for next generation (truncated selection)
                sorted_indices = sorted(range(len(initial_population)), 
                                       key=lambda i: user_ratings.get(i, 0), 
                                       reverse=True)
                
                elite = [initial_population[i] for i in sorted_indices[:recommender.elite_size]]
                
                # Generate offspring through crossover
                offspring = []
                
                # Create chromosomes from current population
                chromosomes = [recommender.chromosome_representation(track) for track in initial_population]
                
                while len(offspring) < recommender.population_size - recommender.elite_size:
                    # Select two parents randomly from top half
                    parent1_idx = random.randint(0, max(1, len(sorted_indices)//2) - 1)
                    parent2_idx = random.randint(0, max(1, len(sorted_indices)//2) - 1)
                    
                    # Avoid selecting the same parent twice
                    while parent2_idx == parent1_idx:
                        parent2_idx = random.randint(0, max(1, len(sorted_indices)//2) - 1)
                    
                    parent1 = chromosomes[sorted_indices[parent1_idx]]
                    parent2 = chromosomes[sorted_indices[parent2_idx]]
                    
                    # Apply crossover
                    if random.random() < recommender.crossover_rate:
                        child_chromosome = recommender.blx_alpha_crossover(parent1, parent2)
                        
                        # Find the most similar actual track to this chromosome
                        child_track = recommender.find_similar_track(child_chromosome)
                        if child_track:
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
                
                # Update session with next generation recommendations
                request.session['current_recommendations'] = next_recommendations
                request.session['current_generation'] = current_generation + 1
                request.session.modified = True
                
                print(f"Generated next generation: {current_generation + 1}")
                
                # Redirect to rate the next generation
                return redirect('rate_recommendations')
            except Exception as e:
                print(f"Error generating next generation: {str(e)}")
                messages.error(request, f"Error generating next generation: {str(e)}")
                return redirect('final_recommendations')
        else:
            print(f"Form errors: {form.errors}")
            messages.error(request, "Please provide all ratings.")
    
    else:
        form = GenerationRatingForm(recommendations=current_recommendations)
    
    return render(request, 'recommender/rate_recommendations.html', {
        'form': form,
        'recommendations': current_recommendations,
        'current_generation': current_generation + 1,
        'total_generations': recommender.generations
    })

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