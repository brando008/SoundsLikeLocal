## Project overview

SoundsLike is a music recommendation system that takes your natural language prompts to suggest songs you might like. It combines a specified song, artist, and/or mood to find new music with similar characteristics.

Simply enter a prompt, adjust the number of recommendations you want, and get new songs that sound like your input!

**Example**: _"Chill songs like Moon by Kanye"_

![](media/SoundsLikeShowcase.gif)

## How it works

SoundsLike utilizes Natural Language Processing (NLP) for understanding user queries and Machine Learning (ML) techniques, such as embeddings, K-Nearest Neighbors (KNN), and vectorization, to find and visualize similar songs.

- It first uses Natural Language Processing (NLP) with a fine-tuned Named Entity Recognition (NER) model to understand your prompt and identify entities like the song, artist, or mood.
- The system then creates a temporary vector that numerically represents your input. This is done by finding a vector for the input song and artist (using a SentenceTransformer model) and combining it with a vector for the input mood.
- It then uses K-Nearest Neighbors (KNN) to find songs in the dataframe that have a vector closest to your input vector.
- Finally, the system visualizes the results by displaying the qualities of your input song versus the recommended songs in a radar chart and a 2D plot.

<p align="center">
  <img src="media/KNN_Graph.png" alt="KNN Graph" width="500"/>
  <img src="media/Vector.png" alt="Vector Graph" width="300"/>
</p>

## Dataset

![Dataset - 🎧 500K+ Spotify Songs with Lyrics,Emotions & More](https://www.kaggle.com/datasets/devdope/900k-spotify)
<a href="https://www.kaggle.com/datasets/devdope/900k-spotify" target="_blank">Dataset - 🎧 500K+ Spotify Songs with Lyrics,Emotions & More</a>

## Team/Contributors

- [![Brandon Aguilar](https://img.shields.io/static/v1?label=Brandon%20Aguilar&message=&color=1DB954&logo=github&logoColor=white&style=flat)](https://github.com/brando008)
- [![Yves Velasquez](https://img.shields.io/static/v1?label=Yves%20Velasquez&message=&color=1DB954&logo=github&logoColor=white&style=flat)](https://github.com/HallowsYves)
- [![May Chan](https://img.shields.io/static/v1?label=May%20Chan&message=&color=1DB954&logo=github&logoColor=white&style=flat)](https://github.com/mchan78)

## Online Repo

This was a rework of the original repo to !<a href="https://github.com/HallowsYves/soundslike" target="_blank">soundslike</a>
for personal exploration. We have our full commit history and version of this app over there.
