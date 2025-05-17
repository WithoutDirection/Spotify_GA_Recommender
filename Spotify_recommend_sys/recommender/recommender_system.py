import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import random
import logging
from .models import Track

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpotifyRecommender:
    """
    A recommender system for Spotify tracks using clustering and genetic algorithms.
    
    This class implements both a Fuzzy Content-Based Recommender System (FCB-RS)
    and a User-oriented Content-Based Recommender System (UCB-RS) approach.
    """
    
    # Track features used for recommendations
    FEATURES = [
        'acousticness', 'danceability', 'energy', 'instrumentalness', 
        'key', 'liveness', 'loudness', 'mode', 'speechiness', 
        'tempo', 'time_signature', 'valence'
    ]
    
    def __init__(self, n_clusters=10, population_size=10, generations=3):
        """
        Initialize the recommender system with parameters.
        
        Args:
            n_clusters: Number of clusters for K-means
            population_size: Size of population for interactive genetic algorithm
            generations: Number of generations for genetic algorithm
        """
        # Initialize the feature scaler
        self.scaler = MinMaxScaler()
        
        # Clustering parameters
        self.n_clusters = n_clusters
        self.cluster_model = None
        self.clusters = {}
        
        # Parameters for Interactive Genetic Algorithm
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = 0.6
        self.elite_size = min(5, population_size // 2)  # Ensure elite size is reasonable
        
        # Load data from database
        self.data = None
        self._load_data_from_db()
    
    def _load_data_from_db(self):
        """Load track data from the database and prepare it for recommendations."""
        try:
            # Get all tracks from database
            tracks = Track.objects.all().values(
                'track_id', 'artist_name', 'track_name', *self.FEATURES, 'cluster'
            )
            
            self.data = pd.DataFrame(list(tracks))
            
            if self.data.empty:
                logger.warning("No tracks found in database")
                return
                
            # Scale the features
            self.data[self.FEATURES] = self.scaler.fit_transform(self.data[self.FEATURES])
            
            # Check if clusters already exist in database
            if 'cluster' in self.data.columns and self.data['cluster'].notna().all():
                # Load existing clusters
                for i in range(self.n_clusters):
                    cluster_data = self.data[self.data['cluster'] == i]
                    if not cluster_data.empty:
                        self.clusters[i] = cluster_data
                logger.info(f"Loaded {len(self.clusters)} existing clusters from database")
            else:
                # Cluster the tracks
                self.cluster_tracks()
                
        except Exception as e:
            logger.error(f"Error loading data from database: {str(e)}")
    
    def cluster_tracks(self):
        """
        Cluster tracks using K-means for better computational efficiency.
        
        Returns:
            Dictionary of clusters containing track data
        """
        if self.data is None or self.data.empty:
            logger.warning("No data available for clustering")
            return {}
            
        try:
            # Apply K-means clustering
            features_array = self.data[self.FEATURES].values
            self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42)
            self.data['cluster'] = self.cluster_model.fit_predict(features_array)
            
            # Store tracks by cluster
            self.clusters = {}
            for i in range(self.n_clusters):
                cluster_data = self.data[self.data['cluster'] == i]
                if not cluster_data.empty:
                    self.clusters[i] = cluster_data
            
            # Update cluster values in the database
            for index, row in self.data.iterrows():
                Track.objects.filter(track_id=row['track_id']).update(cluster=row['cluster'])
            
            # Log cluster statistics
            logger.info(f"Track clustering completed: {len(self.clusters)} clusters created")
            for i in sorted(self.clusters.keys()):
                logger.info(f"Cluster {i}: {len(self.clusters[i])} tracks")
            
            return self.clusters
            
        except Exception as e:
            logger.error(f"Error during clustering: {str(e)}")
            return {}
    
    def fuzzy_similarity(self, track1, track2):
        """
        Compute fuzzy similarity between two tracks based on their features.
        
        Args:
            track1: First track dictionary with feature values
            track2: Second track dictionary with feature values
            
        Returns:
            Similarity score between 0 and 1
        """
        feature_values1 = [track1[feature] for feature in self.FEATURES]
        feature_values2 = [track2[feature] for feature in self.FEATURES]
        
        # Compute the average absolute difference
        sum_abs_diff = sum(abs(feature_values1[i] - feature_values2[i]) 
                          for i in range(len(self.FEATURES)))
        
        # Convert to similarity (1 = identical, 0 = completely different)
        similarity = 1 - (sum_abs_diff / len(self.FEATURES))
        
        return similarity
    
    def recommendation_score(self, track, liked_tracks):
        """
        Compute recommendation score for a track based on user's liked tracks.
        
        Args:
            track: Track to evaluate
            liked_tracks: List of dictionaries containing {'track': track_dict, 'rating': user_rating}
            
        Returns:
            Recommendation score
        """
        if not liked_tracks:
            return 0
            
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
    
    def fcb_rs(self, user_ratings, top_n=10):
        """
        Fuzzy Content-Based Recommender System (FCB-RS)
        
        Args:
            user_ratings: dict where keys are track_ids and values are ratings (1-5)
            top_n: number of recommendations to return
            
        Returns:
            Top-N recommended tracks
        """
        if not user_ratings or self.data is None or self.data.empty:
            logger.warning("Cannot generate recommendations: missing data or ratings")
            return []
            
        # Get liked tracks (rated above average)
        user_avg_rating = sum(user_ratings.values()) / len(user_ratings)
        liked_tracks = []
        
        for track_id, rating in user_ratings.items():
            track_data = self.data[self.data['track_id'] == track_id]
            if not track_data.empty:
                track = track_data.iloc[0].to_dict()
                if rating >= user_avg_rating:
                    liked_tracks.append({'track': track, 'rating': rating})
        
        if not liked_tracks:
            logger.warning("No liked tracks found above average rating")
            return []
            
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
    
    def get_sample_tracks(self, n_per_cluster=1):
        """
        Get a sample of tracks from each cluster for initial recommendations.
        
        Args:
            n_per_cluster: Number of tracks to sample from each cluster
            
        Returns:
            List of track dictionaries
        """
        sample_tracks = []
        
        for cluster_id, cluster_data in self.clusters.items():
            if not cluster_data.empty:
                # Take at most n_per_cluster samples (or all if fewer available)
                sample_size = min(n_per_cluster, len(cluster_data))
                cluster_sample = cluster_data.sample(sample_size)
                
                for _, track in cluster_sample.iterrows():
                    sample_tracks.append(track.to_dict())
        
        return sample_tracks
    
    # --- Interactive Genetic Algorithm methods ---
    
    def chromosome_representation(self, track):
        """
        Represent a track as a chromosome (feature vector) for genetic algorithm.
        
        Args:
            track: Track dictionary
            
        Returns:
            Numpy array of feature values
        """
        return np.array([track[feature] for feature in self.FEATURES])
    
    def blx_alpha_crossover(self, parent1, parent2, alpha=0.5):
        """
        BLX-alpha crossover operator for real-valued chromosomes.
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            alpha: Blending parameter (0.5 is standard)
            
        Returns:
            Child chromosome
        """
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
    
    def euclidean_distance(self, vector1, vector2):
        """
        Calculate normalized Euclidean distance between two feature vectors.
        
        Args:
            vector1: First feature vector
            vector2: Second feature vector
            
        Returns:
            Normalized distance value
        """
        if len(vector1) != len(vector2):
            raise ValueError("Vectors must have the same length")
            
        squared_diff_sum = sum((vector1[i] - vector2[i]) ** 2 for i in range(len(vector1)))
        return (squared_diff_sum / len(vector1)) ** 0.5
    
    def find_similar_track(self, chromosome):
        """
        Find the most similar actual track to a synthetic chromosome.
        
        Args:
            chromosome: Feature vector representing a synthetic track
            
        Returns:
            Most similar real track from the database
        """
        try:
            # Verify cluster_model exists
            if self.cluster_model is None or not self.clusters:
                return self._find_similar_track_full_search(chromosome)
            
            # Get the cluster that would contain this chromosome
            chromosome_df = pd.DataFrame([chromosome], columns=self.FEATURES)
            cluster_id = self.cluster_model.predict(chromosome_df)[0]
            
            # Check if the cluster_id exists
            if cluster_id not in self.clusters:
                return self._find_similar_track_full_search(chromosome)
            
            # Find most similar track in the predicted cluster
            cluster_tracks = self.clusters[cluster_id]
            min_distance = float('inf')
            most_similar_track = None
            
            for _, track in cluster_tracks.iterrows():
                track_chromosome = self.chromosome_representation(track)
                distance = self.euclidean_distance(chromosome, track_chromosome)
                
                if distance < min_distance:
                    min_distance = distance
                    most_similar_track = track.to_dict()
            
            return most_similar_track
            
        except Exception as e:
            logger.error(f"Error finding similar track: {str(e)}")
            return self._find_similar_track_full_search(chromosome)
    
    def _find_similar_track_full_search(self, chromosome):
        """
        Fallback method to find similar track by searching a sample of all tracks.
        
        Args:
            chromosome: Feature vector representing a synthetic track
            
        Returns:
            Most similar real track from a sample of the database
        """
        if self.data is None or self.data.empty:
            logger.warning("No data available for similarity search")
            return None
            
        min_distance = float('inf')
        most_similar_track = None
        
        # Sample up to 1000 tracks from the dataset to avoid performance issues
        sample_size = min(1000, len(self.data))
        sampled_data = self.data.sample(n=sample_size, random_state=42)
        
        for _, track in sampled_data.iterrows():
            track_chromosome = self.chromosome_representation(track)
            distance = self.euclidean_distance(chromosome, track_chromosome)
            
            if distance < min_distance:
                min_distance = distance
                most_similar_track = track.to_dict()
        
        return most_similar_track
    
    def interactive_genetic_algorithm(self, initial_population, user_feedback, generations=None):
        """
        Interactive Genetic Algorithm for adapting recommendations to user feedback.
        
        Args:
            initial_population: Initial set of tracks
            user_feedback: A function that takes a population and returns ratings
            generations: Number of generations to evolve (default: self.generations)
            
        Returns:
            Tuple of (final population of tracks, history of all generations)
        """
        if generations is None:
            generations = self.generations
            
        if not initial_population:
            logger.warning("Empty initial population for genetic algorithm")
            return [], []
        
        current_population = initial_population
        all_generations = [current_population]
        
        for gen in range(generations):
            # Get user ratings for current population
            user_ratings = user_feedback(current_population, gen)
            
            # Check if user is satisfied
            if user_ratings.get('satisfied', False):
                logger.info(f"User satisfied after {gen+1} generations")
                return current_population, all_generations
            
            # Select parents for next generation (truncated selection)
            sorted_indices = sorted(range(len(current_population)), 
                                   key=lambda i: user_ratings.get(i, 0), 
                                   reverse=True)
            
            # Keep elite tracks
            elite_size = min(self.elite_size, len(current_population))
            elite = [current_population[i] for i in sorted_indices[:elite_size]]
            
            # Generate offspring through crossover
            offspring = []
            
            # Create chromosomes from current population
            chromosomes = [self.chromosome_representation(track) for track in current_population]
            
            # Create enough offspring to maintain population size
            while len(offspring) < self.population_size - len(elite):
                # Select two parents randomly from top half
                top_half = max(2, len(sorted_indices) // 2)
                parent1_idx = random.randint(0, top_half - 1)
                parent2_idx = random.randint(0, top_half - 1)
                
                # Avoid selecting the same parent twice
                while parent2_idx == parent1_idx:
                    parent2_idx = random.randint(0, top_half - 1)
                
                parent1 = chromosomes[sorted_indices[parent1_idx]]
                parent2 = chromosomes[sorted_indices[parent2_idx]]
                
                # Apply crossover
                if random.random() < self.crossover_rate:
                    child_chromosome = self.blx_alpha_crossover(parent1, parent2)
                    
                    # Find the most similar actual track to this chromosome
                    child_track = self.find_similar_track(child_chromosome)
                    if child_track is not None:
                        # Check if child is not already in offspring or elite
                        track_id = child_track['track_id']
                        if (track_id not in [t['track_id'] for t in offspring] and 
                            track_id not in [t['track_id'] for t in elite]):
                            offspring.append(child_track)
            
            # Create new population with elite and offspring
            current_population = elite + offspring
            all_generations.append(current_population)
            
            logger.info(f"Generation {gen+1}: {len(current_population)} tracks in population")
        
        return current_population, all_generations
    
    def ucb_rs(self, user_ratings, user_feedback_function, top_n=10):
        """
        User-oriented Content-Based Recommender System (UCB-RS) using genetic algorithm.
        
        Args:
            user_ratings: dict where keys are track_ids and values are ratings (1-5)
            user_feedback_function: Function to get user feedback during IGA
            top_n: number of recommendations to return
            
        Returns:
            Tuple of (top-N recommended tracks, evolution history)
        """
        # Get initial recommendations using FCB-RS
        initial_recommendations = self.fcb_rs(user_ratings, top_n=self.population_size)
        
        if not initial_recommendations:
            logger.warning("Could not generate initial recommendations")
            return [], []
        
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