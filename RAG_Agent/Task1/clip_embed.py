# clip_embed.py
import os
import torch
import numpy as np
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

# keep MODEL_NAME for info, but load from LOCAL_MODEL_DIR
MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
LOCAL_MODEL_DIR = os.path.abspath("./models/chinese-clip")  # <-- update if your path differs

_clip_model = None
_clip_processor = None
_device = None

def init_clip_model():
    global _clip_model, _clip_processor, _device
    if _clip_model is None:
        print(f"Loading CLIP model: {MODEL_NAME}")
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {_device}")
        
        if not os.path.isdir(LOCAL_MODEL_DIR):
            raise FileNotFoundError(f"Local model directory not found: {LOCAL_MODEL_DIR}")
        
        try:
            # Load from local path instead of remote id
            _clip_processor = ChineseCLIPProcessor.from_pretrained(
                LOCAL_MODEL_DIR,
                local_files_only=True
            )
            _clip_model = ChineseCLIPModel.from_pretrained(
                LOCAL_MODEL_DIR,
                local_files_only=True
            ).to(_device)
            print("✓ CLIP model loaded from local path")
        except Exception as e:
            print(f"Error loading model from local path: {e}")
            raise
            
    return _clip_model, _clip_processor, _device

def get_image_embedding(image_path: str) -> np.ndarray:
    """Generate CLIP embedding for an image"""
    model, processor, device = init_clip_model()
    
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features.cpu().numpy().flatten()

def get_text_embedding(text: str) -> np.ndarray:
    """Generate CLIP embedding for text"""
    model, processor, device = init_clip_model()
    
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features.cpu().numpy().flatten()