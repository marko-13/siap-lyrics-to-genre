import data_processing.lyrics_scraping as scraping
import pandas as pd


def run_scraping():
    df = pd.read_csv('../data/music_genres/basic_genres.csv', delimiter=";")

    for index, row in df.iterrows():

        scraping.scrape_extract_save_working(artist_name=row['artist'].split(',')[0],
                                             song_name=row['name'].split('-')[0],
                                             genre_name=row['genre_name'],
                                             index=index)


run_scraping()
