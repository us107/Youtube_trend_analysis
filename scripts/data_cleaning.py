import pandas as pd
import os

# Step 1: Define file paths
RAW_DATA_FILE = os.path.join("data", "youtube_trending_videos.csv")
CLEANED_DATA_FILE = os.path.join("data", "youtube_trending_cleaned.csv")

# Step 2: Load the collected data
try:
    df = pd.read_csv(RAW_DATA_FILE)
    print("Data successfully loaded from:", RAW_DATA_FILE)
except FileNotFoundError:
    print("Error: Raw data file not found. Ensure 'youtube_trending_videos.csv' is in the 'data/' directory.")
    exit()

# Step 3: Inspect the data
print("Initial Data Overview:")
print(df.info())
print(df.head())

# Step 4: Handle missing values
df["Tags"].fillna("Unknown", inplace=True)  # Replace missing tags with "Unknown"
df["Comments"].fillna(0, inplace=True)      # Replace missing comment counts with 0
df["Likes"].fillna(0, inplace=True)         # Replace missing likes with 0

# Step 5: Convert data types
df["Views"] = pd.to_numeric(df["Views"], errors="coerce").fillna(0).astype(int)
df["Likes"] = pd.to_numeric(df["Likes"], errors="coerce").fillna(0).astype(int)
df["Comments"] = pd.to_numeric(df["Comments"], errors="coerce").fillna(0).astype(int)
df["Published At"] = pd.to_datetime(df["Published At"], errors="coerce")

# Step 6: Remove duplicate rows
df.drop_duplicates(inplace=True)

# Step 7: Add a new feature: Engagement Score
df["Engagement Score"] = df["Views"] + 10 * df["Likes"] + 100 * df["Comments"]

# Step 8: Save the cleaned data
os.makedirs("data", exist_ok=True)
df.to_csv(CLEANED_DATA_FILE, index=False)
print("Cleaned data saved to:", CLEANED_DATA_FILE)
