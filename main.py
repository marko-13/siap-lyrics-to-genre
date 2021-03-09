import data_processing.lyrics_scraping as scraping
import lyricsgenius as lg

if __name__ == "__main__":
    # song_url = scraping.request_song_url(artist_name='Daft Punk',
    #                                      song_name='One more time',
    #                                      song_cap=1,
    #                                      print_response=True)
    genius = lg.Genius('Qc5VC8btnP5JS4GLgoR6j1SnGWqJafxshVymsRHNff_Gdmk-CO5sJuefbtFCfTlO',
                       skip_non_songs=True,
                       excluded_terms=["(Remix)", "(Live)"],
                       remove_section_headers=True)

    song = (genius.search_song('Call Me maybe', 'Carly Rae Jepsen'))
    song_lyrics = song.lyrics
    print(song)
