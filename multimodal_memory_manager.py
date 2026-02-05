import sys
import os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

sys.stderr = open(os.devnull, 'w')

class MultimodalMemoryManager:
    
    def __init__(self, visual_db="./test_memory_db", audio_db="./audio_memory_db"):
        self.visual_db_path = Path(visual_db)
        self.audio_db_path = Path(audio_db)
        
        self.visual_client = chromadb.PersistentClient(path=str(self.visual_db_path))
        self.visual_collection = self.visual_client.get_collection("test_cases")
        
        self.audio_client = chromadb.PersistentClient(path=str(self.audio_db_path))
        self.audio_collection = self.audio_client.get_collection("audio_transcripts")
        
        self.embedder = SentenceTransformer('/models/iic/nlp_corom_sentence-embedding_chinese-base')
    
    def search_visual_memory(self, video_name, description):
        query_text = f"Video: {video_name}\nDescription: {description}"
        query_embedding = self.embedder.encode(query_text).tolist()
        
        results = self.visual_collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )
        
        if not results['ids'] or len(results['ids'][0]) == 0:
            return None
        
        best_match_id = results['ids'][0][0]
        best_match_metadata = results['metadatas'][0][0]
        
        import json
        standard_output = json.loads(best_match_metadata['standard_output'])
        
        return {
            'matched_video': best_match_id,
            'standard_output': standard_output
        }
    
    def search_audio_memory(self, video_name):
        try:
            results = self.audio_collection.get(
                ids=[video_name]
            )
            
            if results['ids']:
                return {
                    'video': video_name,
                    'transcript': results['metadatas'][0].get('transcript', ''),
                    'has_speech': results['metadatas'][0].get('has_speech', False)
                }
        except:
            pass
        
        return None
    
    def query_multimodal(self, video_name, visual_description):
        
        visual_result = self.search_visual_memory(video_name, visual_description)
        
        audio_result = self.search_audio_memory(video_name)
        
        return {
            'visual': visual_result,
            'audio': audio_result
        }
    
    def get_stats(self):
        return {
            'visual_records': self.visual_collection.count(),
            'audio_records': self.audio_collection.count()
        }

if __name__ == "__main__":
    sys.stderr = sys.__stderr__
    
    manager = MultimodalMemoryManager()
    
    stats = manager.get_stats()
    print(f"Visual Memory: {stats['visual_records']} records")
    print(f"Audio Memory: {stats['audio_records']} records")
    
    print("\nTest Query:")
    result = manager.query_multimodal(
        "video2_second_year_forget_grandma.mp4",
        "elderly woman greeting"
    )
    
    print(f"\nVisual Match: {result['visual']['matched_video'] if result['visual'] else 'None'}")
    print(f"Audio Transcript: {result['audio']['transcript'][:50] if result['audio'] else 'None'}...")
