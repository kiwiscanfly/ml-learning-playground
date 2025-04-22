import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Directory containing the playlist text files
playlist_dir = "music-playlists"

# Output directory for CSV files
output_dir = "csv-playlists"
os.makedirs(output_dir, exist_ok=True)

def process_playlist(file_path):
  """Send the content of a playlist file to ChatGPT and get a CSV response."""
  with open(file_path, 'r') as file:
    playlist_content = file.read()

  # Send the content to ChatGPT
  response = client.responses.create(
      model="gpt-4o",
      instructions="You are a data engineer. Clean up this playlist return only Artist and Title columns as a CSV, replace the youtube username with the artist and remove artifacts that are not part of the title such as '(Official Lyric Video)', 'Remastered - with lyrics', or '(2018)', if the title includes the album name, remove it. Remove any rows with 'Deleted video' or 'Private video' as the tile. Return only the CSV and no description, make sure the CSV is properly escaped and quoted.",
      input=playlist_content,
  )

  print(f"Response: {response}")
  print(f"Response text: {response.output_text}")

  return response.output_text.strip('```csv').strip('```').strip()

def save_csv(csv_content, output_file):
  """Save the CSV content to a file."""
  with open(output_file, 'w', newline='') as file:
    file.write(csv_content)


# Iterate through each text file in the playlist directory
for filename in os.listdir(playlist_dir):
  print(f"Found file: {filename}")
  if filename.endswith(".csv"):
    file_path = os.path.join(playlist_dir, filename)
    print(f"Processing {file_path}...")

    try:
      # Process the playlist and get the CSV content
      csv_content = process_playlist(file_path)

      # Save the CSV content to a file
      output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.csv")
      save_csv(csv_content, output_file)
      print(f"Saved CSV to {output_file}")
    except Exception as e:
      print(f"Failed to process {file_path}: {e}")
