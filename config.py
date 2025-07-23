import os
from typing import Dict, List

class ReplicateConfig:
    """Configuration for Replicate models and settings"""
    
    # Available face detection models on Replicate
    FACE_DETECTION_MODELS = {
        "face_detection": {
            "model": "andreasjansson/face-detection:7a4b8c8b8c8b8c8b8c8b8c8b8c8b8c8b8c8b8c8b",
            "description": "High-accuracy face detection with landmarks",
            "features": ["bounding_boxes", "landmarks", "confidence_scores"]
        },
        "blip_face": {
            "model": "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
            "description": "BLIP model for face analysis",
            "features": ["face_detection", "description"]
        },
        "face_analysis": {
            "model": "tencentarc/gfpgan:9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3",
            "description": "Face restoration and analysis",
            "features": ["face_enhancement", "landmark_detection"]
        }
    }
    
    # Landmark colors
    LANDMARK_COLORS = {
        "nose": "green",
        "left_eye": "blue", 
        "right_eye": "red",
        "mouth": "cyan",
        "eyebrows": "yellow",
        "face_outline": "purple"
    }
    
    @classmethod
    def get_api_token(cls) -> str:
        """Get Replicate API token from environment"""
        token = os.getenv("REPLICATE_API_TOKEN")
        if not token:
            raise ValueError("REPLICATE_API_TOKEN environment variable is required")
        return token
    
    @classmethod
    def get_default_model(cls) -> str:
        """Get default face detection model"""
        return cls.FACE_DETECTION_MODELS["face_detection"]["model"]
