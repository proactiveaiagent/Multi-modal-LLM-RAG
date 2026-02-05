import sys
import os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from audio_analyzer import AudioAnalyzer
from tqdm import tqdm

sys.stderr = open(os.devnull, 'w')

class AudioMemoryBuilder:
    
    def __init__(self, audio_db_path="./audio_memory_db"):
        self.audio_db_path = Path(audio_db_path)
        self.audio_db_path.mkdir(exist_ok=True)
        
        print("Initializing audio memory system...")
        
        self.client = chromadb.PersistentClient(path=str(self.audio_db_path))
        self.collection = self.client.get_or_create_collection("audio_transcripts")
        
        self.embedder = SentenceTransformer('/models/iic/nlp_corom_sentence-embedding_chinese-base')
        self.audio_analyzer = AudioAnalyzer()
        
        print(f"Audio DB: {self.audio_db_path}")
        print(f"Current audio records: {self.collection.count()}\n")
    
    def process_all_videos(self, videos_dir="videos"):
        videos_dir = Path(videos_dir)
        
        video_files = []
        for scene_dir in sorted(videos_dir.glob("scene*")):
            if scene_dir.is_dir():
                video_files.extend(sorted(scene_dir.glob("*.mp4")))
        
        if not video_files:
            print("No videos found")
            return
        
        print(f"Processing {len(video_files)} videos...\n")
        
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        
        stats = {'zh': 0, 'en': 0, 'no_speech': 0, 'other': 0}
        
        for video_path in tqdm(video_files, desc="Extracting audio"):
            video_name = video_path.name
            
            audio_result = self.audio_analyzer.extract_audio(str(video_path))
            
            if not audio_result['has_speech']:
                doc_text = f"Video: {video_name} - No speech detected"
                stats['no_speech'] += 1
            else:
                doc_text = f"Video: {video_name} - Audio: {audio_result['text']}"
                lang = audio_result['language']
                if lang == 'zh':
                    stats['zh'] += 1
                elif lang == 'en':
                    stats['en'] += 1
                else:
                    stats['other'] += 1
            
            embedding = self.embedder.encode(doc_text).tolist()
            
            ids.append(video_name)
            documents.append(doc_text)
            embeddings.append(embedding)
            metadatas.append({
                'video': video_name,
                'has_speech': audio_result['has_speech'],
                'language': audio_result['language'],
                'word_count': audio_result['word_count'],
                'transcript': audio_result['text']
            })
        
        sys.stderr = sys.__stderr__
        print("\nStoring audio data to vector database...")
        sys.stderr = open(os.devnull, 'w')
        
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        sys.stderr = sys.__stderr__
        
        print(f"\nDone! Total audio records: {self.collection.count()}")
        print(f"\nLanguage Statistics:")
        print(f"  Chinese: {stats['zh']}")
        print(f"  English: {stats['en']}")
        print(f"  No Speech: {stats['no_speech']}")
        print(f"  Other: {stats['other']}")
    
    def show_all_transcripts(self):
        sys.stderr = sys.__stderr__
        
        results = self.collection.get()
        
        print("\n" + "="*60)
        print("All Audio Transcripts:")
        print("="*60)
        
        for i, (vid, meta) in enumerate(zip(results['ids'], results['metadatas']), 1):
            print(f"\n[{i}] {vid}")
            print(f"  Language: {meta.get('language', 'unknown')}")
            
            transcript = meta.get('transcript', '')
            if transcript:
                if len(transcript) > 80:
                    print(f"  Text: {transcript[:80]}...")
                else:
                    print(f"  Text: {transcript}")
            else:
                print(f"  Text: (No speech)")

if __name__ == "__main__":
    builder = AudioMemoryBuilder()
    builder.process_all_videos()
    
    print("\n" + "="*60)
    print("Audio Memory Database Built Successfully!")
    print("="*60)
    
    builder.show_all_transcripts()
