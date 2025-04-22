import json, os, csv

dir_path = "spotify_million_playlist_dataset/data/"
feature_dir = "archive/"
test_file = os.path.join(dir_path, "mpd.slice.0-999.json")

music_pool = {}

def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def write_csv(data, file_path, column_names):
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=column_names)
        writer.writeheader()
        for key, row in data.items():
            row_with_key = {"track_id": key}
            row_with_key.update(row)
            writer.writerow(row_with_key)


if __name__ == "__main__":
    for feature_csv in os.listdir(feature_dir):
        feature_csv_path = os.path.join(feature_dir, feature_csv)
        print(f'Reading file: {feature_csv_path} and adding it to the music pool')
        with open(feature_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # reading the track_id from the csv file
                track_id = row["track_id"]
                if track_id not in music_pool:
                    # adding the features to the music pool
                    music_pool[track_id] = {}
                    for key, value in row.items():
                        if key != "track_id":
                            music_pool[track_id][key] = value
        cloumns = ["track_id"] + list(music_pool[list(music_pool.keys())[0]].keys())
        output_csv_file = "music_pool.csv"
        write_csv(music_pool, output_csv_file, cloumns)
        
    if not os.path.exists("music_pool.csv"):
        print("music_pool.csv does not exist, creating it...")
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            print(f'Reading file: {file_path} and turning it into a csv file')
            data = read_json(file_path)
            playlists = data["playlists"]
            for playlist in playlists:
                tracks = playlist["tracks"]
                for track in tracks:
                    artist_name = track["artist_name"]
                    track_id = track["track_uri"].split(":")[-1]  # Extract just the track ID
                    artist_uri = track["artist_uri"]
                    track_name = track["track_name"]
                    album_uri = track["album_uri"]
                    duration_ms = track["duration_ms"]
                    album_name = track["album_name"]

                    if track_id not in music_pool:
                        music_pool[track_id] = {
                            "artist_name": artist_name,
                            "artist_uri": artist_uri,
                            "track_name": track_name,
                            "album_uri": album_uri,
                            "duration_ms": duration_ms,
                            "album_name": album_name
                        }
            # break # Only read the first file for now
        # reading csv file
        for feature_csv in os.listdir(feature_dir):
            feature_csv_path = os.path.join(feature_dir, feature_csv)
            print(f'Reading file: {feature_csv_path} and adding it to the music pool')
            with open(feature_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # reading the track_id from the csv file
                    track_id = row["track_id"]
                    if track_id in music_pool:
                        # adding the features to the music pool
                        for key, value in row.items():
                            if key != "track_id":
                                music_pool[track_id][key] = value
        column_names = ["track_id"] + list(music_pool[list(music_pool.keys())[0]].keys())
        output_csv_file = "music_pool.csv"
        # print(f'music_pool: {music_pool}')
        write_csv(music_pool, output_csv_file, column_names)
        