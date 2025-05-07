import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import random
import logging
from .models import Track

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SpotifyRecommender:
    def __init__(self):
        """Initialize the recommender system with the Spotify dataset"""
        self.features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                         'key', 'liveness', 'loudness', 'mode', 'speechiness', 
                         'tempo', 'time_signature', 'valence']
        
        # Normalize the features
        self.scaler = MinMaxScaler()
        
        # Initialize the number of clusters for K-means
        self.n_clusters = 10
        self.cluster_model = None
        self.clusters = {}
        
        # Parameters for IGA
        self.population_size = 10
        self.generations = 3
        self.crossover_rate = 0.6
        self.elite_size = 5
        
        # Get data from database
        self._load_data_from_db()
    
    def _load_data_from_db(self):
        """Load track data from the database"""
        tracks = Track.objects.all().values(
            'track_id', 'artist_name', 'track_name', 'acousticness', 'danceability',
            'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
            'speechiness', 'tempo', 'time_signature', 'valence', 'cluster'
        )
        
        self.data = pd.DataFrame(list(tracks))
        
        if not self.data.empty:
            # Scale the features
            self.data[self.features] = self.scaler.fit_transform(self.data[self.features])
            
            # Check if clusters already exist
            if 'cluster' in self.data.columns and self.data['cluster'].notna().all():
                # Load existing clusters
                for i in range(self.n_clusters):
                    self.clusters[i] = self.data[self.data['cluster'] == i]
            else:
                # Cluster the tracks
                self.cluster_tracks()
    
    def fuzzy_similarity(self, track1, track2):
        """Compute fuzzy similarity between two tracks"""
        feature_values1 = [track1[feature] for feature in self.features]
        feature_values2 = [track2[feature] for feature in self.features]
        
        # Compute the similarity
        sum_abs_diff = sum(abs(feature_values1[i] - feature_values2[i]) for i in range(len(self.features)))
        similarity = 1 - (sum_abs_diff / len(self.features))
        
        return similarity
    
    def recommendation_score(self, track, liked_tracks):
        """Compute recommendation score"""
        sum_sim_rating = 0
        sum_sim = 0
        
        for liked_track in liked_tracks:
            sim = self.fuzzy_similarity(track, liked_track['track'])
            rating = liked_track['rating']
            sum_sim_rating += sim * rating
            sum_sim += abs(sim)
        
        if sum_sim == 0:
            return 0
            
        return sum_sim_rating / sum_sim
    
    def cluster_tracks(self):
        """Cluster tracks using K-means for better computational efficiency"""
        # Apply K-means clustering
        features_array = self.data[self.features].values
        self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.data['cluster'] = self.cluster_model.fit_predict(features_array)
        
        # Store tracks by cluster
        self.clusters = {}
        for i in range(self.n_clusters):
            self.clusters[i] = self.data[self.data['cluster'] == i]
        
        # Update cluster values in the database
        for index, row in self.data.iterrows():
            Track.objects.filter(track_id=row['track_id']).update(cluster=row['cluster'])
        
        logging.info("Track clustering completed")
        
        # Log cluster statistics
        for i in range(self.n_clusters):
            logging.info(f"Cluster {i}: {len(self.clusters[i])} tracks")
        
        return self.clusters
    
    def fcb_rs(self, user_ratings, top_n=10):
        """
        Fuzzy Content-Based Recommender System (FCB-RS)
        
        Args:
            user_ratings: dict where keys are track_ids and values are ratings (1-5)
            top_n: number of recommendations to return
            
        Returns:
            Top-N recommended tracks
        """
        # Get liked tracks (rated above average)
        user_avg_rating = sum(user_ratings.values()) / len(user_ratings)
        liked_tracks = []
        
        for track_id, rating in user_ratings.items():
            track_data = self.data[self.data['track_id'] == track_id]
            if not track_data.empty:
                track = track_data.iloc[0].to_dict()
                if rating >= user_avg_rating:
                    liked_tracks.append({'track': track, 'rating': rating})
        
        # Get unrated tracks
        unrated_tracks = self.data[~self.data['track_id'].isin(user_ratings.keys())]
        
        # Calculate recommendation scores for unrated tracks
        recommendations = []
        
        for _, track in unrated_tracks.iterrows():
            track_dict = track.to_dict()
            score = self.recommendation_score(track_dict, liked_tracks)
            recommendations.append({
                'track_id': track_dict['track_id'],
                'artist_name': track_dict['artist_name'],
                'track_name': track_dict['track_name'],
                'score': score
            })
        
        # Sort by score and get top N
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_n]
    
    def chromosome_representation(self, track):
        """Represent a track as a chromosome for IGA"""
        return np.array([track[feature] for feature in self.features])
    
    def blx_alpha_crossover(self, parent1, parent2, alpha=0.5):
        """BLX-alpha crossover operator for real-valued chromosomes"""
        child = []
        for i in range(len(parent1)):
            # Calculate range with alpha extension
            min_val = min(parent1[i], parent2[i]) - alpha * abs(parent1[i] - parent2[i])
            max_val = max(parent1[i], parent2[i]) + alpha * abs(parent1[i] - parent2[i])
            
            # Keep values in [0,1] range after normalization
            min_val = max(0, min_val)
            max_val = min(1, max_val)
            
            # Generate random value in the range
            child.append(random.uniform(min_val, max_val))
            
        return np.array(child)
    
    def distance(self, s, t):
        """Calculate Euclidean distance between two feature vectors"""
        dis = 0.0
        for i in range(len(s)):
            dis += (s[i] - t[i]) ** 2
        return (dis / len(s)) ** 0.5
    
    def find_similar_track(self, chromosome):
        """Find the most similar actual track to a chromosome"""
        try:
            # First, verify cluster_model exists
            if self.cluster_model is None:
                # Initialize the cluster model if it doesn't exist
                # This assumes features are already normalized/processed
                from sklearn.cluster import KMeans
                
                # Get all feature data
                feature_data = self.data[self.features].values
                
                # Use a reasonable number of clusters (adjust as needed)
                n_clusters = min(20, len(self.data) // 50 + 1)  # Rule of thumb
                
                # Create and fit the model
                self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
                self.cluster_model.fit(feature_data)
                
                # Pre-compute clusters to avoid repeated predictions
                cluster_labels = self.cluster_model.predict(feature_data)
                self.clusters = {}
                for i, label in enumerate(cluster_labels):
                    if label not in self.clusters:
                        self.clusters[label] = self.data.iloc[[i]]
                    else:
                        self.clusters[label] = pd.concat([self.clusters[label], self.data.iloc[[i]]])
            
            # Get the cluster that would contain this chromosome
            chromosome_df = pd.DataFrame([chromosome], columns=self.features)
            cluster_id = self.cluster_model.predict(chromosome_df)[0]
            
            # Check if the cluster_id exists in self.clusters
            if cluster_id not in self.clusters:
                # Fall back to finding the most similar track across all data
                return self._find_similar_track_full_search(chromosome)
            
            cluster_tracks = self.clusters[cluster_id]
            
            # Find most similar track in the cluster
            min_distance = float('inf')
            most_similar_track = None
            
            for _, track in cluster_tracks.iterrows():
                track_chromosome = self.chromosome_representation(track)
                distance = self.distance(chromosome, track_chromosome)
                
                if distance < min_distance:
                    min_distance = distance
                    most_similar_track = track.to_dict()
            
            return most_similar_track
        
        except Exception as e:
            print(f"Error in find_similar_track: {str(e)}")
            # Fallback method when clustering fails
            return self._find_similar_track_full_search(chromosome)
        
    def _find_similar_track_full_search(self, chromosome):
        """Fallback method to find similar track without using clustering"""
        min_distance = float('inf')
        most_similar_track = None
        
        # Sample up to 1000 tracks from the dataset to avoid performance issues
        sample_size = min(1000, len(self.data))
        sampled_data = self.data.sample(n=sample_size, random_state=42)
        
        for _, track in sampled_data.iterrows():
            track_chromosome = self.chromosome_representation(track)
            distance = self.distance(chromosome, track_chromosome)
            
            if distance < min_distance:
                min_distance = distance
                most_similar_track = track.to_dict()
        
        return most_similar_track
    
    def interactive_genetic_algorithm(self, initial_population, user_feedback, generations=5):
        """
        Interactive Genetic Algorithm for adapting recommendations to user feedback
        
        Args:
            initial_population: Initial set of tracks
            user_feedback: A function that takes a population and returns ratings
            generations: Number of generations to evolve
            
        Returns:
            Final population of tracks
        """
        current_population = initial_population
        all_generations = [current_population]
        
        for gen in range(generations):
            # Get user ratings for current population
            user_ratings = user_feedback(current_population, gen)
            
            # Check if user is satisfied
            if user_ratings.get('satisfied', False):
                return current_population, all_generations
            
            # Select parents for next generation (truncated selection)
            sorted_indices = sorted(range(len(current_population)), 
                                   key=lambda i: user_ratings.get(i, 0), 
                                   reverse=True)
            
            elite = [current_population[i] for i in sorted_indices[:self.elite_size]]
            
            # Generate offspring through crossover
            offspring = []
            
            # Create chromosomes from current population
            chromosomes = [self.chromosome_representation(track) for track in current_population]
            
            while len(offspring) < self.population_size - self.elite_size:
                # Select two parents randomly from top half
                parent1_idx = random.randint(0, len(sorted_indices)//2 - 1)
                parent2_idx = random.randint(0, len(sorted_indices)//2 - 1)
                
                # Avoid selecting the same parent twice
                while parent2_idx == parent1_idx:
                    parent2_idx = random.randint(0, len(sorted_indices)//2 - 1)
                
                parent1 = chromosomes[sorted_indices[parent1_idx]]
                parent2 = chromosomes[sorted_indices[parent2_idx]]
                
                # Apply crossover
                if random.random() < self.crossover_rate:
                    child_chromosome = self.blx_alpha_crossover(parent1, parent2)
                    
                    # Find the most similar actual track to this chromosome
                    child_track = self.find_similar_track(child_chromosome)
                    if child_track is not None:
                        offspring.append(child_track)
            
            # Create new population with elite and offspring
            current_population = elite + offspring
            all_generations.append(current_population)
        
        return current_population, all_generations
    
    def get_sample_tracks(self, n_per_cluster=1):
        """Get a sample of tracks from each cluster"""
        sample_tracks = []
        
        for i in range(self.n_clusters):
            if i in self.clusters and not self.clusters[i].empty:
                cluster_sample = self.clusters[i].sample(n_per_cluster)
                for _, track in cluster_sample.iterrows():
                    sample_tracks.append(track.to_dict())
        
        return sample_tracks
    
    def ucb_rs(self, user_ratings, user_feedback_function, top_n=10):
        """
        User-oriented Content-Based Recommender System (UCB-RS)
        
        Args:
            user_ratings: dict where keys are track_ids and values are ratings (1-5)
            user_feedback_function: Function to get user feedback during IGA
            top_n: number of recommendations to return
            
        Returns:
            Top-N recommended tracks and evolution history
        """
        # Get initial recommendations using FCB-RS
        initial_recommendations = self.fcb_rs(user_ratings, top_n=self.population_size)
        
        # Convert recommendations to tracks
        initial_population = []
        for rec in initial_recommendations:
            track_data = self.data[self.data['track_id'] == rec['track_id']]
            if not track_data.empty:
                track = track_data.iloc[0].to_dict()
                initial_population.append(track)
        
        # Apply IGA to adapt to user preferences
        final_population, evolution_history = self.interactive_genetic_algorithm(
            initial_population, 
            user_feedback_function,
            self.generations
        )
        
        # Convert final population to recommendations format
        recommendations = []
        for track in final_population[:top_n]:
            recommendations.append({
                'track_id': track['track_id'],
                'artist_name': track['artist_name'],
                'track_name': track['track_name']
            })
        
        return recommendations, evolution_history