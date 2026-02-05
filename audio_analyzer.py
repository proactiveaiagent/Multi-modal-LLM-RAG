import sys
import os
import whisper
from pathlib import Path

sys.stderr = open(os.devnull, 'w')

class AudioAnalyzer:
    
    def __init__(self, model_size="base"):
        self.model = whisper.load_model(model_size)
    
    def extract_audio(self, video_path):
        try:
            result = self.model.transcribe(str(video_path))
            
            text = result.get('text', '').strip()
            detected_language = result.get('language', 'unknown')
            
            return {
                'text': text,
                'language': detected_language,
                'word_count': len(text.split()) if text else 0,
                'has_speech': bool(text)
            }
        except Exception as e:
            return {
                'text': '',
                'language': 'unknown',
                'word_count': 0,
                'has_speech': False
            }
    
    def analyze_video_audio(self, video_path):
        audio_data = self.extract_audio(video_path)
        
        if not audio_data['has_speech']:
            return "No speech detected"
        
        return audio_data['text']

if __name__ == "__main__":
    analyzer = AudioAnalyzer()
    
    test_video = "videos/scene1_family_reunion/video1_first_year_meet_grandma.mp4"
    if Path(test_video).exists():
        result = analyzer.extract_audio(test_video)
        print(f"Language: {result['language']}")
        print(f"Audio: {result['text']}")
