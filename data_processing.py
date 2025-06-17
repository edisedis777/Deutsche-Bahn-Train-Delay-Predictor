import os
import requests
import pandas as pd

# Constants
RAW_DIR = './raw_data'
REPO_OWNER = 'piebro'
REPO_NAME = 'deutsche-bahn-data'
BRANCH = 'master'  # or 'main' if applicable

def download_csv_files():
    os.makedirs(RAW_DIR, exist_ok=True)
    
    # GitHub API to list repo contents
    api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents?ref={BRANCH}"
    
    print("Fetching file list from GitHub repo...")
    response = requests.get(api_url)
    response.raise_for_status()
    files = response.json()
    
    # Filter CSV files only
    csv_files = [f for f in files if f['name'].endswith('.csv')]
    
    for file in csv_files:
        download_url = file['download_url']
        local_path = os.path.join(RAW_DIR, file['name'])
        print(f"Downloading {file['name']}...")
        r = requests.get(download_url)
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            f.write(r.content)
    print("All CSV files downloaded.")

def preprocess_data():
    files = [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    if not files:
        raise RuntimeError("No CSV files found in raw_data folder. Run download first.")
    
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Convert times
    for col in ['planned_arrival', 'actual_arrival']:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Compute delay in minutes
    df['delay_minutes'] = (df['actual_arrival'] - df['planned_arrival']).dt.total_seconds() / 60.0

    # Drop rows with missing critical data
    df = df.dropna(subset=['delay_minutes', 'station_id', 'train_type', 'planned_arrival'])

    # Add weekday and hour features
    df['weekday'] = df['planned_arrival'].dt.dayofweek
    df['hour'] = df['planned_arrival'].dt.hour

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['station_id', 'train_type'], drop_first=True)

    # Save processed data
    os.makedirs('./processed_data', exist_ok=True)
    df.to_csv('./processed_data/cleaned_train_data.csv', index=False)
    print("Preprocessing complete. Saved to ./processed_data/cleaned_train_data.csv")

if __name__ == "__main__":
    download_csv_files()
    preprocess_data()
