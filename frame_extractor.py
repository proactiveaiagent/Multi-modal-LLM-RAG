import cv2
from pathlib import Path
from PIL import Image

class FrameExtractor:
    
    def __init__(self):
        self.strategy = "smart"
    
    def extract_key_frames(self, video_path, num_frames=3, strategy=None):
        
        if not Path(video_path).exists():
            print(f"Error: video not found {video_path}")
            return None, None
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: cannot open {video_path}")
            return None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video: {duration:.1f}s, {fps:.1f}fps, {total_frames} frames")
        
        use_strategy = strategy if strategy else self.strategy
        
        if use_strategy == "smart" and num_frames == 3:
            if duration <= 6:
                frame_indices = [
                    int(total_frames * 0.2),
                    int(total_frames * 0.5),
                    max(total_frames - 2, int(total_frames * 0.8))
                ]
                print(f"Short video sampling: 20%, 50%, 80%")
            else:
                frame_indices = [
                    int(total_frames * 0.15),
                    int(total_frames * 0.5),
                    max(total_frames - 2, int(total_frames * 0.85))
                ]
                print(f"Smart sampling: 15%, 50%, 85%")
        else:
            if total_frames <= num_frames:
                frame_indices = list(range(total_frames))
            else:
                step = total_frames / (num_frames + 1)
                frame_indices = [int(step * (i + 1)) for i in range(num_frames)]
            print(f"Uniform sampling")
        
        print(f"Frame indices: {frame_indices}")
        
        frames = []
        frame_info = {
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration,
            'extracted_indices': frame_indices,
            'extracted_timestamps': [],
            'strategy': use_strategy
        }
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
                
                timestamp = idx / fps if fps > 0 else 0
                frame_info['extracted_timestamps'].append(timestamp)
                
                print(f"  Frame {idx} ({timestamp:.1f}s)")
            else:
                print(f"  Warning: cannot read frame {idx}")
        
        cap.release()
        print(f"Extraction complete: {len(frames)} frames")
        
        return frames, frame_info
