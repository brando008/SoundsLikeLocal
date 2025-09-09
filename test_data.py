import pandas as pd
from data_utils import load_data, scale_data

df = load_data("data_test/clean_songs_reduced.csv")
# df_sampled = df.sample(n=10000, random_state=42)
# df_sampled.to_csv("data_test/clean_songs_reduced.csv", index=False)

df_numeric = df[['Positiveness', 'Danceability', 'Energy', 'Popularity',
        'Liveness', 'Acousticness', 'Instrumentalness']].copy()
df_numeric.to_csv('data_test/numeric_data_reduced.csv')

df_song = df[['Artist(s)', 'Song', 'Emotion', 'Genre']].copy()
df_song.to_csv('data_test/song_data_reduced.csv', index=True)

scale_data('data_test/numeric_data_reduced.csv',
           index=True, 
           save_path='data_test/scaled_data_reduced.csv'
)