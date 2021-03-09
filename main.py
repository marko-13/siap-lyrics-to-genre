import lyricsgenius as lg

if __name__ == "__main__":
    genius = lg.Genius('Qc5VC8btnP5JS4GLgoR6j1SnGWqJafxshVymsRHNff_Gdmk-CO5sJuefbtFCfTlO',
                       skip_non_songs=True,
                       excluded_terms=["(Remix)", "(Live)"],
                       remove_section_headers=True)

    song = (genius.search_song('Call Me maybe', 'Carly Rae Jepsen'))
    song_lyrics = song.lyrics
    print(song)
