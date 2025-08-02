"""
Spotify Integration Module
Tích hợp Spotify Web API để lấy dữ liệu training cho mô hình phân loại thể loại nhạc
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import json
import base64
from urllib.parse import quote
import warnings
warnings.filterwarnings('ignore')

class SpotifyIntegration:
    def __init__(self):
        self.client_id = os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        self.access_token = None
        self.base_url = "https://api.spotify.com/v1"
        
        if not self.client_id or not self.client_secret:
            print("⚠️ Cảnh báo: Không tìm thấy Spotify credentials")
            print("Đặt SPOTIFY_CLIENT_ID và SPOTIFY_CLIENT_SECRET environment variables")
        else:
            self.access_token = self.get_access_token()
    
    def get_access_token(self) -> Optional[str]:
        """Lấy access token từ Spotify"""
        if not self.client_id or not self.client_secret:
            return None
        
        try:
            # Encode credentials
            credentials = f"{self.client_id}:{self.client_secret}"
            encoded_credentials = base64.b64encode(credentials.encode()).decode()
            
            # Request token
            token_url = "https://accounts.spotify.com/api/token"
            headers = {
                "Authorization": f"Basic {encoded_credentials}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            data = {"grant_type": "client_credentials"}
            
            response = requests.post(token_url, headers=headers, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            return token_data.get("access_token")
            
        except Exception as e:
            print(f"❌ Lỗi lấy access token: {e}")
            return None
    
    def search_tracks_by_genre(self, genre: str, limit: int = 50) -> List[Dict]:
        """
        Tìm kiếm tracks theo thể loại
        
        Args:
            genre: Thể loại nhạc
            limit: Số lượng tracks tối đa
        
        Returns:
            List of track dictionaries
        """
        if not self.access_token:
            print("❌ Không có access token")
            return []
        
        try:
            # Search query
            query = f"genre:{genre}"
            encoded_query = quote(query)
            
            # API request
            url = f"{self.base_url}/search"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            params = {
                "q": query,
                "type": "track",
                "limit": min(limit, 50),  # Spotify limit
                "market": "US"
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            tracks = data.get("tracks", {}).get("items", [])
            
            # Extract track info
            track_list = []
            for track in tracks:
                track_info = {
                    "id": track["id"],
                    "name": track["name"],
                    "artist": track["artists"][0]["name"] if track["artists"] else "Unknown",
                    "album": track["album"]["name"],
                    "duration_ms": track["duration_ms"],
                    "popularity": track["popularity"],
                    "preview_url": track["preview_url"],
                    "external_url": track["external_urls"]["spotify"],
                    "genre": genre
                }
                track_list.append(track_info)
            
            print(f"✅ Tìm thấy {len(track_list)} tracks cho thể loại '{genre}'")
            return track_list
            
        except Exception as e:
            print(f"❌ Lỗi tìm kiếm tracks cho '{genre}': {e}")
            return []
    
    def get_audio_features(self, track_ids: List[str]) -> List[Dict]:
        """
        Lấy audio features cho danh sách tracks
        
        Args:
            track_ids: List of Spotify track IDs
        
        Returns:
            List of audio features dictionaries
        """
        if not self.access_token:
            print("❌ Không có access token")
            return []
        
        if not track_ids:
            return []
        
        try:
            # Spotify API limit: 100 tracks per request
            features_list = []
            
            for i in range(0, len(track_ids), 100):
                batch_ids = track_ids[i:i+100]
                ids_param = ",".join(batch_ids)
                
                url = f"{self.base_url}/audio-features"
                headers = {"Authorization": f"Bearer {self.access_token}"}
                params = {"ids": ids_param}
                
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                features = data.get("audio_features", [])
                features_list.extend(features)
                
                # Rate limiting
                time.sleep(0.1)
            
            print(f"✅ Lấy audio features cho {len(features_list)} tracks")
            return features_list
            
        except Exception as e:
            print(f"❌ Lỗi lấy audio features: {e}")
            return []
    
    def create_training_dataset(self, genres: List[str], tracks_per_genre: int = 100) -> pd.DataFrame:
        """
        Tạo dataset training từ Spotify
        
        Args:
            genres: List of genres to collect
            tracks_per_genre: Number of tracks per genre
        
        Returns:
            DataFrame with track data and audio features
        """
        print(f"🎵 Bắt đầu tạo dataset training cho {len(genres)} thể loại...")
        
        all_tracks = []
        
        for genre in genres:
            print(f"🔍 Đang xử lý thể loại: {genre}")
            
            # Search tracks
            tracks = self.search_tracks_by_genre(genre, tracks_per_genre)
            
            if not tracks:
                print(f"⚠️ Không tìm thấy tracks cho {genre}")
                continue
            
            # Get track IDs
            track_ids = [track["id"] for track in tracks]
            
            # Get audio features
            features = self.get_audio_features(track_ids)
            
            # Combine track info with features
            for i, track in enumerate(tracks):
                if i < len(features) and features[i]:
                    track_data = {**track, **features[i]}
                    all_tracks.append(track_data)
            
            # Rate limiting
            time.sleep(1)
        
        # Create DataFrame
        df = pd.DataFrame(all_tracks)
        
        print(f"✅ Dataset hoàn tất: {len(df)} tracks, {len(df.columns)} features")
        return df
    
    def download_preview_audio(self, preview_url: str, save_path: str) -> bool:
        """
        Tải preview audio từ Spotify
        
        Args:
            preview_url: Spotify preview URL
            save_path: Path to save audio file
        
        Returns:
            True if successful
        """
        if not preview_url:
            return False
        
        try:
            response = requests.get(preview_url, stream=True)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
            
        except Exception as e:
            print(f"❌ Lỗi tải preview audio: {e}")
            return False
    
    def create_spotify_dataset(self, genres: List[str], tracks_per_genre: int = 50) -> Dict:
        """
        Tạo dataset hoàn chỉnh với audio files
        
        Args:
            genres: List of genres
            tracks_per_genre: Tracks per genre
        
        Returns:
            Dictionary with dataset info
        """
        print("🎵 Tạo Spotify dataset hoàn chỉnh...")
        
        # Create directories
        os.makedirs("data/spotify", exist_ok=True)
        for genre in genres:
            os.makedirs(f"data/spotify/{genre}", exist_ok=True)
        
        # Get track data
        df = self.create_training_dataset(genres, tracks_per_genre)
        
        if df.empty:
            return {"success": False, "message": "Không thể tạo dataset"}
        
        # Download preview audio
        downloaded_count = 0
        for _, row in df.iterrows():
            if pd.notna(row.get("preview_url")):
                genre = row["genre"]
                track_id = row["id"]
                filename = f"{track_id}.mp3"
                save_path = f"data/spotify/{genre}/{filename}"
                
                if self.download_preview_audio(row["preview_url"], save_path):
                    downloaded_count += 1
                
                # Rate limiting
                time.sleep(0.5)
        
        # Save metadata
        metadata_path = "data/spotify/metadata.csv"
        df.to_csv(metadata_path, index=False)
        
        result = {
            "success": True,
            "total_tracks": len(df),
            "downloaded_audio": downloaded_count,
            "genres": genres,
            "metadata_path": metadata_path,
            "audio_directory": "data/spotify"
        }
        
        print(f"✅ Dataset hoàn tất:")
        print(f"  - Tổng tracks: {result['total_tracks']}")
        print(f"  - Audio files: {result['downloaded_audio']}")
        print(f"  - Metadata: {metadata_path}")
        
        return result
    
    def get_genre_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Phân tích thống kê dataset theo thể loại
        
        Args:
            df: Dataset DataFrame
        
        Returns:
            Dictionary with statistics
        """
        if df.empty:
            return {}
        
        stats = {}
        
        # Genre distribution
        genre_counts = df["genre"].value_counts()
        stats["genre_distribution"] = genre_counts.to_dict()
        
        # Audio features statistics
        audio_features = [
            "danceability", "energy", "key", "loudness", "mode",
            "speechiness", "acousticness", "instrumentalness",
            "liveness", "valence", "tempo"
        ]
        
        feature_stats = {}
        for feature in audio_features:
            if feature in df.columns:
                feature_stats[feature] = {
                    "mean": df[feature].mean(),
                    "std": df[feature].std(),
                    "min": df[feature].min(),
                    "max": df[feature].max()
                }
        
        stats["audio_features"] = feature_stats
        
        return stats
    
    def search_playlists_by_genre(self, genre: str, limit: int = 20) -> List[Dict]:
        """
        Tìm kiếm playlists theo thể loại
        
        Args:
            genre: Thể loại nhạc
            limit: Số lượng playlists
        
        Returns:
            List of playlist dictionaries
        """
        if not self.access_token:
            return []
        
        try:
            query = f"genre:{genre}"
            encoded_query = quote(query)
            
            url = f"{self.base_url}/search"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            params = {
                "q": query,
                "type": "playlist",
                "limit": min(limit, 50)
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            playlists = data.get("playlists", {}).get("items", [])
            
            playlist_list = []
            for playlist in playlists:
                playlist_info = {
                    "id": playlist["id"],
                    "name": playlist["name"],
                    "description": playlist.get("description", ""),
                    "tracks_count": playlist["tracks"]["total"],
                    "external_url": playlist["external_urls"]["spotify"],
                    "genre": genre
                }
                playlist_list.append(playlist_info)
            
            return playlist_list
            
        except Exception as e:
            print(f"❌ Lỗi tìm kiếm playlists: {e}")
            return []

def main():
    """Test Spotify integration"""
    spotify = SpotifyIntegration()
    
    if not spotify.access_token:
        print("❌ Không thể kết nối Spotify API")
        return
    
    # Test search
    genres = ["pop", "rock", "jazz", "classical"]
    
    for genre in genres:
        tracks = spotify.search_tracks_by_genre(genre, limit=5)
        print(f"{genre}: {len(tracks)} tracks")
    
    # Test audio features
    if tracks:
        track_ids = [t["id"] for t in tracks[:3]]
        features = spotify.get_audio_features(track_ids)
        print(f"Audio features: {len(features)}")

if __name__ == "__main__":
    main() 