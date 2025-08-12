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
        # Try environment variables first
        self.client_id = os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
        
        # Fallback to hardcoded client ID for testing
        if not self.client_id:
            self.client_id = "abc9aef9bc5545d093bdd910c9298b3e"
            print("🔑 Using hardcoded Spotify Client ID for testing")
        
        self.access_token = None
        self.base_url = "https://api.spotify.com/v1"
        self.demo_mode = False
        
        if not self.client_secret:
            print("⚠️ Cảnh báo: Không tìm thấy Spotify Client Secret")
            print("💡 Chuyển sang chế độ demo với dữ liệu mẫu")
            self.demo_mode = True
        else:
            self.access_token = self.get_access_token()
            if not self.access_token:
                print("⚠️ Không thể lấy access token, chuyển sang demo mode")
                self.demo_mode = True
    
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
    
    def _get_demo_tracks(self, genre: str, limit: int = 10) -> List[Dict]:
        """Tạo demo tracks khi không có Spotify API"""
        demo_tracks = {
            'pop': [
                {'id': 'demo_pop_1', 'name': 'Pop Song 1', 'artist': 'Pop Artist 1', 'popularity': 85},
                {'id': 'demo_pop_2', 'name': 'Pop Song 2', 'artist': 'Pop Artist 2', 'popularity': 78},
                {'id': 'demo_pop_3', 'name': 'Pop Song 3', 'artist': 'Pop Artist 3', 'popularity': 92}
            ],
            'rock': [
                {'id': 'demo_rock_1', 'name': 'Rock Song 1', 'artist': 'Rock Artist 1', 'popularity': 88},
                {'id': 'demo_rock_2', 'name': 'Rock Song 2', 'artist': 'Rock Artist 2', 'popularity': 75},
                {'id': 'demo_rock_3', 'name': 'Rock Song 3', 'artist': 'Rock Artist 3', 'popularity': 82}
            ],
            'jazz': [
                {'id': 'demo_jazz_1', 'name': 'Jazz Song 1', 'artist': 'Jazz Artist 1', 'popularity': 65},
                {'id': 'demo_jazz_2', 'name': 'Jazz Song 2', 'artist': 'Jazz Artist 2', 'popularity': 72},
                {'id': 'demo_jazz_3', 'name': 'Jazz Song 3', 'artist': 'Jazz Artist 3', 'popularity': 68}
            ],
            'classical': [
                {'id': 'demo_classical_1', 'name': 'Classical Song 1', 'artist': 'Classical Artist 1', 'popularity': 60},
                {'id': 'demo_classical_2', 'name': 'Classical Song 2', 'artist': 'Classical Artist 2', 'popularity': 55},
                {'id': 'demo_classical_3', 'name': 'Classical Song 3', 'artist': 'Classical Artist 3', 'popularity': 70}
            ],
            'hiphop': [
                {'id': 'demo_hiphop_1', 'name': 'Hip Hop Song 1', 'artist': 'Hip Hop Artist 1', 'popularity': 90},
                {'id': 'demo_hiphop_2', 'name': 'Hip Hop Song 2', 'artist': 'Hip Hop Artist 2', 'popularity': 85},
                {'id': 'demo_hiphop_3', 'name': 'Hip Hop Song 3', 'artist': 'Hip Hop Artist 3', 'popularity': 88}
            ]
        }
        
        return demo_tracks.get(genre.lower(), [])[:limit]
    
    def search_tracks_by_genre(self, genre: str, limit: int = 50) -> List[Dict]:
        """
        Tìm kiếm tracks theo thể loại
        
        Args:
            genre: Thể loại nhạc
            limit: Số lượng tracks tối đa
        
        Returns:
            List of track dictionaries
        """
        if self.demo_mode:
            print(f"🎵 Demo mode: Tìm kiếm {genre} tracks")
            return self._get_demo_tracks(genre, limit)
        
        if not self.access_token:
            print("❌ Không có access token")
            return self._get_demo_tracks(genre, limit)
        
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
                    "popularity": track.get("popularity", 0),
                    "duration_ms": track.get("duration_ms", 0),
                    "album": track["album"]["name"] if track["album"] else "Unknown"
                }
                track_list.append(track_info)
            
            return track_list
            
        except Exception as e:
            print(f"❌ Lỗi tìm kiếm Spotify: {e}")
            print("🔄 Chuyển sang demo mode")
            return self._get_demo_tracks(genre, limit)
    
    def get_audio_features(self, track_ids: List[str]) -> List[Dict]:
        """Lấy audio features cho tracks"""
        if self.demo_mode:
            print("🎵 Demo mode: Tạo audio features mẫu")
            demo_features = []
            for track_id in track_ids:
                # Tạo features ngẫu nhiên dựa trên genre
                if 'pop' in track_id:
                    features = {
                        'id': track_id,
                        'danceability': 0.8,
                        'energy': 0.7,
                        'valence': 0.6,
                        'tempo': 120.0,
                        'acousticness': 0.2,
                        'instrumentalness': 0.1
                    }
                elif 'rock' in track_id:
                    features = {
                        'id': track_id,
                        'danceability': 0.5,
                        'energy': 0.9,
                        'valence': 0.4,
                        'tempo': 140.0,
                        'acousticness': 0.1,
                        'instrumentalness': 0.3
                    }
                elif 'jazz' in track_id:
                    features = {
                        'id': track_id,
                        'danceability': 0.6,
                        'energy': 0.4,
                        'valence': 0.5,
                        'tempo': 90.0,
                        'acousticness': 0.8,
                        'instrumentalness': 0.7
                    }
                else:
                    features = {
                        'id': track_id,
                        'danceability': 0.6,
                        'energy': 0.6,
                        'valence': 0.5,
                        'tempo': 110.0,
                        'acousticness': 0.4,
                        'instrumentalness': 0.2
                    }
                demo_features.append(features)
            return demo_features
        
        if not self.access_token:
            return []
        
        try:
            # Spotify API limit: 100 tracks per request
            all_features = []
            for i in range(0, len(track_ids), 100):
                batch = track_ids[i:i+100]
                
                url = f"{self.base_url}/audio-features"
                headers = {"Authorization": f"Bearer {self.access_token}"}
                params = {"ids": ",".join(batch)}
                
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                features = data.get("audio_features", [])
                all_features.extend(features)
            
            return all_features
            
        except Exception as e:
            print(f"❌ Lỗi lấy audio features: {e}")
            return []
    
    def create_training_dataset(self, genres: List[str], tracks_per_genre: int = 50) -> pd.DataFrame:
        """Tạo dataset training từ Spotify"""
        if self.demo_mode:
            print("🎵 Demo mode: Tạo dataset training mẫu")
            return self._create_demo_dataset(genres, tracks_per_genre)
        
        all_tracks = []
        
        for genre in genres:
            print(f"🔍 Tìm kiếm {genre} tracks...")
            tracks = self.search_tracks_by_genre(genre, tracks_per_genre)
            
            if tracks:
                track_ids = [track['id'] for track in tracks]
                features = self.get_audio_features(track_ids)
                
                for track, feature in zip(tracks, features):
                    if feature:
                        track_data = {
                            'genre': genre,
                            'track_id': track['id'],
                            'track_name': track['name'],
                            'artist': track['artist'],
                            'danceability': feature.get('danceability', 0),
                            'energy': feature.get('energy', 0),
                            'valence': feature.get('valence', 0),
                            'tempo': feature.get('tempo', 0),
                            'acousticness': feature.get('acousticness', 0),
                            'instrumentalness': feature.get('instrumentalness', 0)
                        }
                        all_tracks.append(track_data)
        
        df = pd.DataFrame(all_tracks)
        print(f"✅ Dataset created: {len(df)} tracks, {len(genres)} genres")
        return df
    
    def _create_demo_dataset(self, genres: List[str], tracks_per_genre: int = 50) -> pd.DataFrame:
        """Tạo dataset demo"""
        all_tracks = []
        
        for genre in genres:
            for i in range(tracks_per_genre):
                # Tạo features ngẫu nhiên dựa trên genre
                if genre == 'pop':
                    features = {
                        'danceability': np.random.uniform(0.7, 0.9),
                        'energy': np.random.uniform(0.6, 0.8),
                        'valence': np.random.uniform(0.5, 0.7),
                        'tempo': np.random.uniform(100, 140),
                        'acousticness': np.random.uniform(0.1, 0.3),
                        'instrumentalness': np.random.uniform(0.0, 0.2)
                    }
                elif genre == 'rock':
                    features = {
                        'danceability': np.random.uniform(0.4, 0.6),
                        'energy': np.random.uniform(0.8, 1.0),
                        'valence': np.random.uniform(0.3, 0.5),
                        'tempo': np.random.uniform(120, 160),
                        'acousticness': np.random.uniform(0.0, 0.2),
                        'instrumentalness': np.random.uniform(0.2, 0.4)
                    }
                elif genre == 'jazz':
                    features = {
                        'danceability': np.random.uniform(0.5, 0.7),
                        'energy': np.random.uniform(0.3, 0.5),
                        'valence': np.random.uniform(0.4, 0.6),
                        'tempo': np.random.uniform(80, 100),
                        'acousticness': np.random.uniform(0.7, 0.9),
                        'instrumentalness': np.random.uniform(0.6, 0.8)
                    }
                else:
                    features = {
                        'danceability': np.random.uniform(0.5, 0.7),
                        'energy': np.random.uniform(0.5, 0.7),
                        'valence': np.random.uniform(0.4, 0.6),
                        'tempo': np.random.uniform(100, 120),
                        'acousticness': np.random.uniform(0.3, 0.5),
                        'instrumentalness': np.random.uniform(0.1, 0.3)
                    }
                
                track_data = {
                    'genre': genre,
                    'track_id': f'demo_{genre}_{i}',
                    'track_name': f'{genre.title()} Song {i+1}',
                    'artist': f'{genre.title()} Artist {i+1}',
                    **features
                }
                all_tracks.append(track_data)
        
        df = pd.DataFrame(all_tracks)
        print(f"✅ Demo dataset created: {len(df)} tracks, {len(genres)} genres")
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