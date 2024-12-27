import requests
import pandas as pd

# Step 1: Set up API details
API_KEY = "AIzaSyDSKBCSfbi4_-g1MMcSjg-_a3tQYvSQ53I"  # Replace with your API key
BASE_URL = "https://www.googleapis.com/youtube/v3/videos"

# Step 2: Define parameters for the API request
params = {
    "part": "snippet,statistics",  # Fetch both video metadata and statistics
    "chart": "mostPopular",        # Get trending videos
    "regionCode": "IN",            # Region code for India
    "maxResults": 50,              # Number of videos to fetch
    "key": API_KEY
}

# Step 3: Make the API request
response = requests.get(BASE_URL, params=params)

# Step 4: Handle the response
if response.status_code == 200:
    data = response.json()
    print("API Request Successful!")
else:
    print("API Request Failed:", response.status_code)
    data = {}

# Step 5: Extract and Store the Data
if "items" in data:
    videos = []
    for item in data["items"]:
        video_info = {
            "Video ID": item["id"],
            "Title": item["snippet"]["title"],
            "Channel": item["snippet"]["channelTitle"],
            "Published At": item["snippet"]["publishedAt"],
            "Views": item["statistics"].get("viewCount", "N/A"),
            "Likes": item["statistics"].get("likeCount", "N/A"),
            "Comments": item["statistics"].get("commentCount", "N/A"),
            "Tags": item["snippet"].get("tags", "N/A"),
        }
        videos.append(video_info)
    
    # Create a DataFrame
    df = pd.DataFrame(videos)
    print(df)
    
    # Save to CSV
    df.to_csv("youtube_trending_videos.csv", index=False)
    print("Data saved to youtube_trending_videos.csv")
else:
    print("No data found in API response.")
