import chromadb
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path

class RAGMemoryManager:
    
    def __init__(self, db_path="./test_memory_db"):
        self.db_path = Path(db_path)
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_collection("test_cases")
        self.embedder = SentenceTransformer('/models/iic/nlp_corom_sentence-embedding_chinese-base')
    
    def search_similar_case(self, video_name, initial_description):
        
        query_text = f"Video: {video_name}\nDescription: {initial_description}"
        query_embedding = self.embedder.encode(query_text).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1
        )
        
        if not results['ids'] or len(results['ids'][0]) == 0:
            return None
        
        best_match_id = results['ids'][0][0]
        best_match_metadata = results['metadatas'][0][0]
        
        standard_output = json.loads(best_match_metadata['standard_output'])
        
        return {
            'matched_video': best_match_id,
            'standard_output': standard_output
        }