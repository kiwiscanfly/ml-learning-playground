import time
import csv
import os
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_APIKEY")
playlist_id = os.getenv("FAVOURITE_PLAYLIST_ID")

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

def get_video_comments(video_id):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )
    while request:
        print("Requesting comments...")
        response = request.execute()
        comments.extend(response['items'])
        request = youtube.commentThreads().list_next(request, response)
    return comments

# Load existing CSV file into a DataFrame
filename = "favourite_playlist.csv"
if os.path.exists(filename):
    df = pd.read_csv(filename)
else:
    df = pd.DataFrame(columns=['Title', 'Channel', 'VideoId', 'Comment'])

# Get items from favourite playlist
items = get_playlist_items(playlist_id)

# Open the CSV file to append new entries
with open(filename, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    for item in items:
        snippet = item['snippet']
        title = snippet['title']
        artist = snippet.get('videoOwnerChannelTitle', 'N/A').replace(' - Topic', '')
        video_id = snippet['resourceId']['videoId']

        # Check if the video is already in the CSV
        if video_id in df['VideoId'].values:
            print(f"Video {video_id} already in CSV, skipping...")
            continue

        try:
            comments = get_video_comments(video_id)
        except Exception as e:
            print(f"Error: {e}")
            continue

        for comment in comments:
            comment_text = comment['snippet']['topLevelComment']['snippet']['textDisplay']
            writer.writerow([title, artist, video_id, comment_text])
            # Append new entry to the DataFrame
            df = df.append({'Title': title, 'Channel': artist, 'VideoId': video_id, 'Comment': comment_text}, ignore_index=True)

        time.sleep(1)