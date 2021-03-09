import lyricsgenius as lg
from data.credentials import myAccessToken


if __name__ == "__main__":
    genius = lg.Genius(myAccessToken,
                       skip_non_songs=True,
                       excluded_terms=["(Remix)", "(Live)"],
                       remove_section_headers=True)

    song = (genius.search_song('Call Me maybe', 'Carly Rae Jepsen'))
    song_lyrics = song.lyrics
    print(song)
