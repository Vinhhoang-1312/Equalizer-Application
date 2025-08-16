#!/usr/bin/env python3
"""
Demo: Detailed Genre Classification Information
"""

from modules.advanced_genre_classifier import AdvancedGenreClassifier
import json
import os

def demo_detailed_classification():
    print("🎵 DEMO: Chi Tiết Phân Loại Genre với ML")
    print("=" * 60)
    
    classifier = AdvancedGenreClassifier()
    
    # Test với rock file
    test_file = 'uploads/rock.00004.wav'
    if not os.path.exists(test_file):
        print(f"⚠️ Test file not found: {test_file}")
        return
    
    print(f"🎸 Analyzing: {test_file}")
    result = classifier.option2_custom_ml_classify(test_file)
    
    print("\n📊 CLASSIFICATION RESULT:")
    print(f"   Genre: {result['predicted_genre']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   Method: {result['method']}")
    
    print("\n🗂️ DATASET INFORMATION:")
    dataset = result['dataset_info']
    print(f"   📂 Dataset: {dataset['name']}")
    print(f"   🎵 Total samples: {dataset['total_samples']}")
    print(f"   ⚖️ Samples per genre: {dataset['samples_per_genre']}")
    print(f"   ⏱️ Audio length: {dataset['audio_length']}")
    print(f"   🎯 Average accuracy: {dataset['average_accuracy']}")
    print(f"   ✅ Validation: {dataset['cross_validation']}")
    
    print("\n🔍 MOST IMPORTANT FEATURES:")
    for i, feature in enumerate(dataset['feature_importance']['most_important_features'], 1):
        print(f"   {i}. {feature}")
    
    print(f"\n🎼 GENRE SIGNATURE - {result['predicted_genre'].upper()}:")
    signature = dataset['feature_importance']['genre_signatures'][result['predicted_genre']]
    print(f"   {signature}")
    
    print("\n🔧 TECHNICAL DETAILS:")
    tech = result['technical_details']
    print(f"   🤖 Algorithm: {tech['model_details']['algorithm']}")
    print(f"   🌳 Number of trees: {tech['model_details']['n_estimators']}")
    print(f"   📏 Feature scaling: {tech['model_details']['feature_scaling']}")
    print(f"   🗳️ Prediction method: {tech['model_details']['prediction_method']}")
    
    print("\n📈 AUDIO FEATURES ANALYZED:")
    features = result['audio_features']
    print(f"   🎵 Spectral Centroid: {features['spectral_centroid']:.0f} Hz")
    print(f"   🥁 Tempo: {features['tempo']:.0f} BPM")
    print(f"   ⚡ Zero Crossing Rate: {features['zero_crossing_rate']:.3f}")
    print(f"   🎼 Harmonic Ratio: {features['harmonic_ratio']:.2f}")
    
    print("\n🧠 WHY THIS GENRE? (AI Reasoning):")
    for i, reason in enumerate(result['classification_reasoning'], 1):
        print(f"   {i}. {reason}")
    
    print("\n🏆 TOP 3 PREDICTIONS:")
    ml_analysis = result['ml_analysis']
    for i, (genre, prob) in enumerate(ml_analysis['top_3_predictions'], 1):
        print(f"   {i}. {genre}: {prob:.1%}")
    
    print("\n" + "=" * 60)
    print("✨ BÂY GIỜ FRONTEND SẼ HIỂN THỊ TẤT CẢ THÔNG TIN NÀY!")
    print("📱 Mở web app tại: http://127.0.0.1:5000")
    print("🎯 Upload file rock và click 'Musicnn AI' để xem chi tiết!")

if __name__ == "__main__":
    demo_detailed_classification()
