#!/usr/bin/env python3
"""
Test improved ML feature extraction
"""

from modules.advanced_genre_classifier import AdvancedGenreClassifier
import os

def test_improved_ml():
    classifier = AdvancedGenreClassifier()
    print("🎵 Testing improved ML classification...")
    
    # Tìm file rock trong uploads để test
    upload_files = os.listdir('uploads/')
    rock_files = [f for f in upload_files if '.wav' in f or '.mp3' in f]
    
    if rock_files:
        test_file = os.path.join('uploads', rock_files[0])
        print(f"Testing với file: {test_file}")
        
        result = classifier.option2_custom_ml_classify(test_file)
        print(f"Predicted genre: {result.get('predicted_genre', 'unknown')}")
        print(f"Confidence: {result.get('confidence', 0):.1%}")
        print(f"Status: {result.get('status', 'unknown')}")
        
        if 'all_probabilities' in result:
            print("\n📊 All genre probabilities:")
            probs = result['all_probabilities']
            for genre, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                print(f"  {genre}: {prob:.1%}")
                
    else:
        print("⚠️ No test files found in uploads/")
        
if __name__ == "__main__":
    test_improved_ml()
