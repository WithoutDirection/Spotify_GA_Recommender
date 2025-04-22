import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import random
import matplotlib.pyplot as plt
from tabulate import tabulate
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().setLevel(logging.DEBUG)

data_path = "music_pool.csv"

class SpotifyRecommender:
    def __init__(self, data_path):
        """Initialize the recommender system with the Spotify dataset"""
        self.data = pd.read_csv(data_path)
        self.features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                         'key', 'liveness', 'loudness', 'mode', 'speechiness', 
                         'tempo', 'time_signature', 'valence']
        
        # Normalize the features
        self.scaler = MinMaxScaler()
        self.data[self.features] = self.scaler.fit_transform(self.data[self.features])
        
        # Initialize the number of clusters for K-means
        self.n_clusters = 10
        self.cluster_model = None
        self.clusters = None
        
        # Parameters for IGA
        self.population_size = 20
        self.generations = 6
        self.crossover_rate = 0.6
        self.elite_size = 5
        
    def fuzzy_similarity(self, track1, track2):
        """Compute fuzzy similarity between two tracks using equation (2) from the paper"""
        feature_values1 = track1[self.features].values
        feature_values2 = track2[self.features].values
        
        # Compute the similarity as per equation 
        sum_abs_diff = sum(abs(feature_values1[i] - feature_values2[i]) for i in range(len(self.features)))
        similarity = 1 - (sum_abs_diff / len(self.features))
        
        return similarity
    
    def recommendation_score(self, track, liked_tracks):
        """
        Compute recommendation score using formula: similarity = 1 - (sum of absolute differences between features) / (total number of features)
        """
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
        self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.data['cluster'] = self.cluster_model.fit_predict(self.data[self.features]) # Add cluster labels to the data
        
        # Store tracks by cluster
        self.clusters = {}
        for i in range(self.n_clusters):
            self.clusters[i] = self.data[self.data['cluster'] == i]
        # Output the music num that belongs to each cluster
        for i in range(self.n_clusters):
            logging.debug(f"Cluster {i}:")
            logging.debug(f"Number of tracks: {len(self.clusters[i])}")
            # print(f"Tracks: {self.clusters[i]['track_name'].tolist()}")
            
        return self.clusters
    
    def fcb_rs(self, user_ratings, top_n=10):
        """
        Fuzzy Content-Based Recommender System (FCB-RS): Recommendation Score
        
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
            if rating >= user_avg_rating:
                track = self.data[self.data['track_id'] == track_id].iloc[0]
                liked_tracks.append({'track': track, 'rating': rating})
        
        # Get unrated tracks
        unrated_tracks = self.data[~self.data['track_id'].isin(user_ratings.keys())]
        
        # Calculate recommendation scores for unrated tracks
        recommendations = []
        
        for _, track in unrated_tracks.iterrows():
            # print(f"Calculating recommendation score for track: {track['track_name']}")
            score = self.recommendation_score(track, liked_tracks)
            recommendations.append({
                'track_id': track['track_id'],
                'artist_name': track['artist_name'],
                'track_name': track['track_name'],
                'score': score
            })
        
        # Sort by score and get top N
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_n]
    
    def chromosome_representation(self, track):
        """Represent a track as a chromosome for IGA"""
        return track[self.features].values
    
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
        dis = 0.0
        for i in range(len(s)):
            dis += (s[i] - t[i]) ** 2
        return (dis / len(s)) ** 0.5
    def find_similar_track(self, chromosome):
        """Find the most similar actual track to a chromosome"""
        # Get the cluster that would contain this chromosome
        chromosome_df = pd.DataFrame([chromosome], columns=self.features)
        cluster_id = self.cluster_model.predict(chromosome_df)[0]
        cluster_tracks = self.clusters[cluster_id]
        
        # Find most similar track in the cluster
        min_distance = float('inf')
        most_similar_track = None
        
        for _, track in cluster_tracks.iterrows():
            track_chromosome = self.chromosome_representation(track)
            # print(f'calculated distance of {chromosome} and {track_chromosome}')
            distance = self.distance(chromosome, track_chromosome) 
            
            if distance < min_distance:
                min_distance = distance
                most_similar_track = track
        
        return most_similar_track
    
    def interactive_genetic_algorithm(self, initial_population, generations=5):
        """
        Interactive Genetic Algorithm for adapting recommendations to user feedback
        
        Args:
            initial_population: Initial set of tracks as chromosomes
            generations: Number of generations to evolve
            
        Returns:
            Final population of tracks
        """
        current_population = initial_population

        # inside interactive_genetic_algorithm
        for gen in range(generations):
            logging.debug(f'-' * 35)
            logging.debug(f'Generation {gen+1}')
            logging.debug(f'-' * 35)

            table = []
            user_ratings = {}
            for i, track in enumerate(current_population):
                table.append([
                    i+1,
                    track['artist_name'],
                    track['track_name']
                ])
            print(tabulate(table, headers=["#", "Artist", "Track"], tablefmt="pretty"))

            for i in range(len(current_population)):
                while True:
                    try:
                        rating = float(input(f"Rate track #{i+1} (1-5): "))
                        if 1 <= rating <= 5:
                            break
                        else:
                            print("Rating must be between 1 and 5.")
                    except ValueError:
                        print("Invalid input. Enter a number between 1 and 5.")
                user_ratings[i] = rating

            satisfied = input("\nAre you satisfied with these recommendations? (y/n): ").lower()
            if satisfied == 'y':
                return current_population

            
            # Select parents for next generation (truncated selection)
            sorted_indices = sorted(range(len(current_population)), key=lambda i: user_ratings[i], reverse=True)
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
                    offspring.append(child_track)
            
            # Create new population with elite and offspring
            current_population = elite + offspring
            
        return current_population
    
    def ucb_rs(self, user_ratings, top_n=10):
        """
        User-oriented Content-Based Recommender System (UCB-RS)
        
        Args:
            user_ratings: dict where keys are track_ids and values are ratings (1-5)
            top_n: number of recommendations to return
            
        Returns:
            Top-N recommended tracks
        """
        
        # Get initial recommendations using FCB-RS
        initial_recommendations = self.fcb_rs(user_ratings, top_n=self.population_size)
        print("\nInitial recommendations:")
        for rec in initial_recommendations:
            print(f"{rec['artist_name']} - {rec['track_name']}")
        initial_population = []
        
        for rec in initial_recommendations:
            track = self.data[self.data['track_id'] == rec['track_id']].iloc[0] # Get the actual track data
            initial_population.append(track)
        print("Initial population:")
        for track in initial_population:
            print(f"{track['artist_name']} - {track['track_name']}")
        
        # Apply IGA to adapt to user preferences
        final_population = self.interactive_genetic_algorithm(initial_population, self.generations)
        
        # Convert final population to recommendations format
        recommendations = []
        for track in final_population:
            recommendations.append({
                'track_id': track['track_id'],
                'artist_name': track['artist_name'],
                'track_name': track['track_name']
            })
        
        return recommendations[:top_n]

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    recommender = SpotifyRecommender(data_path)
    clusters = recommender.cluster_tracks()
    print("\n" + "="*35)
    logging.info("🎧 Spotify Recommender System Initialized 🎧")
    print("\n" + "="*35)
    # list 10 tracks in each cluster as ucb_rs's initial population
    user_ratings = dict()
    for i in range(recommender.n_clusters):
        track_sample = recommender.clusters[i].sample(1, random_state=42)
        track_id = track_sample['track_id'].values[0]
        logging.info(f"Sampled track from cluster {i}: {track_sample['track_name'].values[0]}; track_id: {track_id}")
        logging.info(f'Please rate this track (1-5): ')
        while True:
            try:
                rating = float(input(f"Rate track {track_sample['track_name'].values[0]} (1-5): "))
                if 1 <= rating <= 5:
                    break
                else:
                    print("Rating must be between 1 and 5.")
            except ValueError:
                print("Invalid input. Enter a number between 1 and 5.")
        user_ratings[track_id] = rating
    
    # Get recommendations
    recommendations = recommender.ucb_rs(user_ratings)
    
    print("\n" + "="*35)
    print("🎧 Final Recommendations 🎧")
    print("="*35)

    final_table = [
        [i+1, rec['artist_name'], rec['track_name']]
        for i, rec in enumerate(recommendations)
    ]

    print(tabulate(final_table, headers=["Rank", "Artist", "Track"], tablefmt="grid"))
