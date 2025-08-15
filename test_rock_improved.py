#!/usr/bin/env python3
"""
Test rock classification with improved ML features
"""

from modules.advanced_genre_classifier import AdvancedGenreClassifier
import os

def test_rock_classification():
    classifier = AdvancedGenreClassifier()
    print("🎸 Testing ROCK classification with improved ML features...")
    
    rock_files = ['rock.00003.wav', 'rock.00004.wav', 'rock.00006.wav', 'rock.00008.wav']
    
    for rock_file in rock_files:
        if os.path.exists(f'uploads/{rock_file}'):
            print(f"\n🎵 Testing: {rock_file}")
            result = classifier.option2_custom_ml_classify(f'uploads/{rock_file}')
            
            predicted = result.get('predicted_genre', 'unknown')
            confidence = result.get('confidence', 0)
            
            if predicted == 'rock':
                print(f"✅ CORRECT: Predicted {predicted} with {confidence:.1%} confidence")
            else:
                print(f"❌ WRONG: Predicted {predicted} instead of rock ({confidence:.1%})")
            
            if 'all_probabilities' in result:
                probs = result['all_probabilities']
                rock_prob = probs.get('rock', 0)
                blues_prob = probs.get('blues', 0)
                print(f"   Rock probability: {rock_prob:.1%}")
                print(f"   Blues probability: {blues_prob:.1%}")
                
                # Show top 3 predictions
                top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"   Top 3: {', '.join([f'{g}({p:.1%})' for g, p in top_3])}")
        
if __name__ == "__main__":
    test_rock_classification()
