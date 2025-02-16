import time
import csv
import os
from ytmusicapi import YTMusic
from googleapiclient.discovery import build
from dotenv import load_dotenv
import re

yt = YTMusic("headers_auth.json")

load_dotenv()
api_key = os.getenv("GOOGLE_APIKEY")

youtube = build('youtube', 'v3', developerKey=api_key)

def get_playlist_items(playlist_id):
    items = []
    request = youtube.playlistItems().list(
        part="snippet",
        playlistId=playlist_id,
        maxResults=50
    )
    while request:
        print("Requesting playlist items...")
        response = request.execute()
        items.extend(response['items'])
        request = youtube.playlistItems().list_next(request, response)
    return items

# Get all playlists
playlists = yt.get_library_playlists(limit=100)

for playlist in playlists:
    playlist_id = playlist['playlistId']
    playlist_name = playlist['title']

    # Skip the "Liked Music" playlist
    if playlist_name.lower() == "liked music" or playlist_name.lower() == "waiata anthems" or playlist_name.lower() == "witchy grrrl pop":
        continue
    
    time.sleep(1)
    print(f"Processing playlist: {playlist_name} (ID: {playlist_id})")

    # Open a CSV file to write the playlist items
    filename = f"playlists/{re.sub(r'[^a-zA-Z0-9_]', '', playlist_name.replace(' ', '_').replace('/', '-').replace('__', '_'))}.csv"
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Title', 'Artist', 'VideoId'])
        
        try:
          items = get_playlist_items(playlist_id)
        except Exception as e:
          print(f"Error: {e}")
          continue
        for item in items:
            snippet = item['snippet']
            title = snippet['title']
            artist = snippet.get('videoOwnerChannelTitle', 'N/A').replace(' - Topic', '')
            video_id = snippet['resourceId']['videoId']

            writer.writerow([title, artist, video_id])
            
            # Print the result
            print(f"Saved: {title} by {artist}")