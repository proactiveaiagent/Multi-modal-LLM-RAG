import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch_br
import torch.nn as nn
from transformers import AutoProcessor, LlavaForConditionalGeneration
import json


class CpuOffloadConv2d(nn.Module):
    def __init__(self, original_conv: nn.Conv2d):
        super().__init__()
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        self.cpu_conv = original_conv.to(device="cpu", dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor):
        original_device = x.device
        original_dtype = x.dtype
        x_cpu = x.to(device="cpu", dtype=torch.bfloat16)
        out_cpu = self.cpu_conv(x_cpu)
        return out_cpu.to(device=original_device, dtype=original_dtype)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.cpu_conv, name)

def patch_vision_model_to_cpu(model):
    print(">>> [Patch] Offloading Vision Tower Conv2d to CPU...")
    
    vision_tower = None
    if hasattr(model, "model") and hasattr(model.model, "vision_tower"):
        vision_tower = model.model.vision_tower
    elif hasattr(model, "vision_tower"):
        vision_tower = model.vision_tower
    
    if vision_tower is None:
        print("Warning: no vision_tower")
        return model

    count = 0
    def recursive_replace(module):
        nonlocal count
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                wrapped_conv = CpuOffloadConv2d(child)
                setattr(module, name, wrapped_conv)
                count += 1
            else:
                recursive_replace(child)

    recursive_replace(vision_tower)
    print(f">>>> [Patch] Completed. Offloaded {count} Conv2d layers.")
    return model


class LLaVAAnalyzerBiren:
    def __init__(self, model_path="/models/llava-1.5-7b", gpu_id=1):
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.device = self._get_device()
        
        print(f">>> [Init] Selected Device: {self.device}")
        
        try:
            print(">>> [Init] Loading Processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            print(">>> [Fix] Fixing patch_size...")
            if hasattr(self.processor, 'image_processor'):
                self.processor.image_processor.patch_size = 14
                print(f"    Set processor.image_processor.patch_size = 14")
            
            if hasattr(self.processor, 'patch_size'):
                self.processor.patch_size = 14
                print(f"    Set processor.patch_size = 14")
                
            print(f">>> [Verify] processor.image_processor.patch_size = {getattr(self.processor.image_processor, 'patch_size', 'NOT FOUND')}")
            
            print(">>> [Init] Loading Model (BF16)...")
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )

            patch_vision_model_to_cpu(self.model)
            
            print(f">>> [Init] Moving model to {self.device} ...")
            self.model.to(self.device)
            print(">>> [Init] Model Ready!")
            
        except Exception as e:
            print(f"!!! Initialization Failed: {e}")
            import traceback
            traceback.print_exc()
            self.device = torch.device("cpu")

    def _get_device(self):
        try:
            if torch.supa.is_available():
                gpu_count = torch.supa.device_count()
                if self.gpu_id < gpu_count:
                    return torch.device(f"supa:{self.gpu_id}")
                else:
                    print(f">>> GPU {self.gpu_id} ，GPU 0")
                    return torch.device("supa:0")
        except:
            pass
        return torch.device("cpu")

    def analyze_frames(self, frames, prompt):
        if not isinstance(frames, list):
            frames = [frames]
        
        num_images = len(frames)
        clean_prompt = prompt.replace("USER: <image>", "").replace("<image>", "").replace("USER:", "").strip()
        image_tokens = "<image>\n" * num_images
        final_prompt = f"USER: {image_tokens}{clean_prompt}\nASSISTANT:"

        if hasattr(self.processor, 'image_processor'):
            if not hasattr(self.processor.image_processor, 'patch_size') or self.processor.image_processor.patch_size is None:
                self.processor.image_processor.patch_size = 14
                print(">>> [Fix] Re-setting patch_size to 14 before processing")

        inputs = self.processor(text=final_prompt, images=frames, return_tensors="pt")
        
        for k, v in inputs.items():
            v_dev = v.to(self.device)
            if k == 'pixel_values' and self.model.dtype == torch.bfloat16:
                 v_dev = v_dev.to(torch.bfloat16)
            inputs[k] = v_dev

        try:
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.9
                )
            
            decoded_output = self.processor.batch_decode(output, skip_special_tokens=True)[0]
            
            if "ASSISTANT:" in decoded_output:
                content = decoded_output.split("ASSISTANT:")[-1].strip()
            else:
                content = decoded_output
            
            try:
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = content[start:end]
                    return json.loads(json_str), content
                else:
                    return {"scene": "unknown", "raw": content}, content
            except:
                return {"scene": "analyzed", "raw": content, "need_reminder": False}, content

        except Exception as e:
            print(f"!!! Inference Error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "need_reminder": False}, str(e)

LLaVAAnalyzer = LLaVAAnalyzerBiren
