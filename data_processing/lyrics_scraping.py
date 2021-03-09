from bs4 import BeautifulSoup
import requests
import os
import re
import lyricsgenius as lg

# GENIUS CREDENTIALS
myClientSecret = 'oGRhL_Be59r5sNJ6MRuQ_KCqmYCQ0RYhWo87ukY0etmN0v6qOIPs6LQO9xgAXyGSB8gnULruEAkbtlk7fir4aQ'
myClientID = 'fXN02nxU6NwMc9wKtfE-Zltz40tvzlMAMVo63BFt4QeXz7uC5sMvPRcpWpVgkmJY'
myAccessToken = 'Qc5VC8btnP5JS4GLgoR6j1SnGWqJafxshVymsRHNff_Gdmk-CO5sJuefbtFCfTlO'


def request_query_info(query, page):
    base_url = 'https://api.genius.com'
    headers = {'Authorization': 'Bearer ' + myAccessToken}
    search_url = base_url + '/search?per_page=10&page=' + str(page)
    data = {'q': query}
    response = requests.get(search_url, data=data, headers=headers)

    return response


def request_song_url(artist_name, song_name, song_cap, print_response=False):
    query = artist_name + ' ' + song_name
    page = 1
    song_url = []

    response = request_query_info(query, page)
    json = response.json()

    if print_response:
        print(json)

    # Collect up to song_cap number of songs from query result
    song_info = []

    for hit in json['response']['hits']:
        if artist_name.lower() in hit['result']['primary_artist']['name'].lower():
            song_info.append(hit)

    # Collect song urls from song objects
    for song in song_info:
        if len(song_url) < song_cap:
            url = song['result']['url']
            song_url.append(url)

    print('Found song named {} by {}'.format(song_name, artist_name))
    print(song_url)
    return song_url


# Get song lyrics from song url
def scrape_song_lyrics(url):
    page = requests.get(url)
    html = BeautifulSoup(page.text, 'html.parser')

    # print(html)
    try:
        print(html.find_all('br').get_text())
        lyrics = html.find('div', class_='lyrics').get_text()
        # remove indentifiers like chorus, verse, ...
        lyrics = re.sub(r'[\(\[].*?[\)\]]', '', lyrics)
        # remove empty lines
        lyrics = os.linesep.join([s for s in lyrics.splitlines() if s])
        return lyrics
    finally:
        print("Could not find lyrics for {}".format(url))
        return ""


# Save lyrics to file
def save_lyrics_to_file(artist_name, song_name, genre_name, lyrics, index):
    f = open('../data/lyrics/' + genre_name.lower() + '.txt', 'ab+')
    # Move read cursor to the start of file

    f.write("\n\n\n\n\n".encode('utf-8'))
    text_to_write = artist_name + '\n' + song_name + '\n' + lyrics
    f.write(text_to_write.encode('utf-8'))
    f.close()

    print("Added lyrics for song {} by {}".format(song_name, artist_name))
    print("Finished {}/4000\n".format(index + 1))


# Search songs by name and artist, extract lyrics and save to file
def scrape_extract_save(artist_name, song_name, genre_name):
    # Song_url is in array, use song_url[0] when accessing
    song_url = request_song_url(artist_name=artist_name,
                                song_name=song_name,
                                song_cap=1,
                                print_response=False)
    # Check if list is empty
    if not song_url:
        print('List is empty, could not find song {} by {}'.format(song_name, artist_name))
        f = open('../data/lyrics/songs_not_found.txt', 'a+')
        # Move read cursor to the start of file
        f.seek(0)
        # If file is not empty then append '\n'
        if len(f.read(100)) > 0:
            f.write("\n")
        f.write(str(song_name) + ' by ' + str(artist_name))
        f.close()
        return

    song_lyrics = scrape_song_lyrics(song_url[0])
    if song_lyrics == "":
        return
    save_lyrics_to_file(artist_name=artist_name,
                        song_name=song_name,
                        genre_name=genre_name,
                        lyrics=song_lyrics)


genius = lg.Genius('Qc5VC8btnP5JS4GLgoR6j1SnGWqJafxshVymsRHNff_Gdmk-CO5sJuefbtFCfTlO',
                   skip_non_songs=True,
                   excluded_terms=["(Remix)", "(Live)"],
                   remove_section_headers=True)


# Scrape and save working
def scrape_extract_save_working(artist_name, song_name, genre_name, index):

    try:
        song = (genius.search_song(song_name, artist_name.split(',')[0]))
        song_lyrics = song.lyrics

        # Check if list is empty
        if not song.lyrics:
            print('List is empty, could not find song {} by {}'.format(song_name, artist_name))
            f = open('../data/lyrics/songs_not_found.txt', 'a+')
            # Move read cursor to the start of file
            f.seek(0)
            # If file is not empty then append '\n'
            if len(f.read(100)) > 0:
                f.write("\n")
            f.write(str(song_name) + ' by ' + str(artist_name))
            f.close()
            return

        if song_lyrics == "":
            return
        save_lyrics_to_file(artist_name=artist_name,
                            song_name=song_name,
                            genre_name=genre_name,
                            lyrics=song_lyrics,
                            index=index)

    finally:
        return



