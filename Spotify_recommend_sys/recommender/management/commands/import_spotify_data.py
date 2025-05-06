from django.core.management.base import BaseCommand
import pandas as pd
from recommender.models import Track
import os
import django

class Command(BaseCommand):
    help = 'Import Spotify track data from CSV'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='Path to the CSV file')

    def handle(self, *args, **options):
        csv_file = options['csv_file']
        self.stdout.write(f"Importing data from {csv_file}...")
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Count for feedback
            created_count = 0
            updated_count = 0
            
            # Process each row
            for _, row in df.iterrows():
                track, created = Track.objects.update_or_create(
                    track_id=row['track_id'],
                    defaults={
                        'artist_name': row['artist_name'],
                        'track_name': row['track_name'],
                        'acousticness': row['acousticness'],
                        'danceability': row['danceability'],
                        'energy': row['energy'],
                        'instrumentalness': row['instrumentalness'],
                        'key': row['key'],
                        'liveness': row['liveness'],
                        'loudness': row['loudness'],
                        'mode': row['mode'],
                        'speechiness': row['speechiness'],
                        'tempo': row['tempo'],
                        'time_signature': row['time_signature'],
                        'valence': row['valence'],
                        'cluster': row.get('cluster', None)
                    }
                )
                
                if created:
                    created_count += 1
                else:
                    updated_count += 1
                
                # Show progress
                if (created_count + updated_count) % 10000 == 0:
                    self.stdout.write(f"Processed {created_count + updated_count} tracks...")
            
            self.stdout.write(self.style.SUCCESS(
                f"Successfully imported data: {created_count} tracks created, {updated_count} tracks updated"
            ))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error importing data: {str(e)}"))