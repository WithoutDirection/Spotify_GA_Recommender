import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import os
import json
import requests
import base64

# Load the data
# music_pool = pd.read_csv("music_pool.csv")
# column_show = ["artist_name", "track_name", "album_name", "duration_ms",  "track_uri"] 
# # show the first 100 rows without column
# print(music_pool[column_show].info)

client_id = "981d8a0341f041dbbe997218d9672475"
client_secret = "09c3fac64e244785bdf3b8204dd83245"
redirect_uri = "http://localhost:8888/callback" 
# Authenticate with Spotify API
access_token = None

def create_spotify_oauth():
    return SpotifyOAuth (
        client_id = client_id,
        client_secret = client_secret,
        redirect_uri=redirect_uri,
        scope="user-library-read user-top-read"
    )  

def get_token():
    client_credentials = f"{client_id}:{client_secret}"
    client_credentials_base64 = base64.b64encode(client_credentials.encode())
    token_url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Authorization': f'Basic {client_credentials_base64.decode()}'
    }
    data = {
        'grant_type': 'client_credentials'
    }
    response = requests.post(token_url, data=data, headers=headers)

    if response.status_code == 200:
        global access_token
        print("Access token obtained successfully.")
        return response.json()['access_token']
        access_token = response.json()['access_token']
        
    else:
        print("Error obtaining access token.")
        exit()
def get_features(track_id, access_token):
    print(f'access_token: {access_token}')
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    url = f'https://api.spotify.com/v1/audio-features/{track_id}'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        # Parse the JSON response
        features = response.json()
        return features
    else:
        print(f"Error: {response.status_code}")
        return None

if __name__ == "__main__":
    # Authenticate with Spotify API
    sp = spotipy.Spotify(auth_manager=create_spotify_oauth())
    # Get the access token
    access_token = get_token()
    # Get the features for a specific track ID
    track_id = "4iV5Isq9ZsYc0y2X1a3v7d"  # Replace with your track ID
    features = get_features(track_id, access_token)
    print(features)

