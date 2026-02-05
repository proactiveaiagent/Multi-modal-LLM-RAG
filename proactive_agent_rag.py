import sys
import os
from pathlib import Path
import json
import torch

from memory_manager_rag import RAGMemoryManager
from frame_extractor import FrameExtractor
from llava_analyzer_biren import LLaVAAnalyzerBiren

class ProactiveAgentRAG:
    
    def __init__(self, videos_dir="videos", output_dir="output"):
        print("="*60)
        print("Proactive Agent RAG Version")
        print("="*60)
        
        self.videos_dir = Path(videos_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.rag_memory = RAGMemoryManager()
        self.frame_extractor = FrameExtractor()
        self.analyzer = LLaVAAnalyzerBiren()
        
        print("Initialization complete\n")
    
    def process_all_videos(self):
        
        video_files = []
        for scene_dir in sorted(self.videos_dir.glob("scene*")):
            if scene_dir.is_dir():
                video_files.extend(sorted(scene_dir.glob("*.mp4")))
        
        if not video_files:
            print(f"No videos found in {self.videos_dir}")
            return
        
        print(f"Found {len(video_files)} videos\n")
        
        for i, video_path in enumerate(video_files, 1):
            print("\n" + "="*60)
            print(f"[{i}/{len(video_files)}] {video_path.name}")
            print("="*60)
            
            self.process_single_video(video_path)
        
        print("\n" + "="*60)
        print("All videos processed")
        print("="*60)
    
    def process_single_video(self, video_path):
        
        video_name = video_path.name
        
        print("\n[1/4] Extracting frames...")
        frames, frame_info = self.frame_extractor.extract_key_frames(str(video_path), num_frames=3)
        
        if not frames:
            print(f"Skip: no frames extracted")
            return
        
        print("\n[2/4] Initial analysis...")
        initial_prompt = self._build_initial_prompt()
        initial_analysis, _ = self.analyzer.analyze_frames(frames, initial_prompt)
        
        initial_description = initial_analysis.get('description', 
                                                   initial_analysis.get('scene', 'unknown scene'))
        
        print("\n[3/4] RAG search...")
        similar_case = self.rag_memory.search_similar_case(video_name, initial_description)
        
        if similar_case:
            print(f"  Matched: {similar_case['matched_video']}")
            final_output = similar_case['standard_output']
        else:
            print(f"  No match found, using initial analysis")
            final_output = initial_analysis
        
        print("\n[4/4] Save result...")
        self.print_result(video_name, final_output)
        self.save_result(video_name, final_output)
    
    def _build_initial_prompt(self):
        return """Analyze these video frames and provide a brief description.
Focus on: scene location, people, objects, user actions.

Output JSON:
{
  "scene": "location",
  "description": "brief description of what's happening"
}"""
    
    def print_result(self, video_name, output):
        print("\n" + "-"*60)
        print("Result")
        print("-"*60)
        print(json.dumps(output, indent=2, ensure_ascii=False))
        
        if output.get('need_reminder', False):
            print("\n" + "!"*60)
            print("REMINDER TRIGGERED")
            print("!"*60)
            print(f"Message: {output.get('reminder_message', 'N/A')}")
            print(f"Urgency: {output.get('urgency', 'N/A')}")
    
    def save_result(self, video_name, output):
        result = {
            "video": video_name,
            "output": output
        }
        
        output_file = self.output_dir / f"{Path(video_name).stem}_result.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"Saved: {output_file}")

if __name__ == "__main__":
    agent = ProactiveAgentRAG()
    agent.process_all_videos()