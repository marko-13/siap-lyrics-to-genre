import requests
import lyricsgenius as lg
from data.credentials import myAccessToken
import csv


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


# Save lyrics to file
def save_lyrics_to_file(artist_name, song_name, genre_name, lyrics, index):
    # delete ; whereever it appears
    lyrics = lyrics.replace(";", "")
    lyrics = lyrics.replace("\n", " ")
    artist_name = artist_name.replace(";", "")
    song_name = song_name.replace(";", "")

    f = open('../data/lyrics/' + genre_name.lower() + '.txt', 'ab+')
    # Move read cursor to the start of file

    f.write("\n\n\n\n\n".encode('utf-8'))
    text_to_write = artist_name + '\n' + song_name + '\n' + lyrics
    f.write(text_to_write.encode('utf-8'))
    f.close()

    # SAVE FILE TO CSV
    with open('../data/lyrics_csv/' + genre_name.lower() + '.csv', 'a+', encoding='utf-8') as csv_f:
        csv_writer = csv.writer(csv_f, delimiter=';')
        csv_writer.writerow([artist_name, song_name, lyrics])

    print("Added lyrics for song {} by {}".format(song_name, artist_name))
    print("Finished {}/4000\n".format(index + 1))


genius = lg.Genius(myAccessToken,
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



