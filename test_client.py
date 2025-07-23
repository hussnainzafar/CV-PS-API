import requests
import cv2
import numpy as np

def test_api():
    """Test the facial landmarks detection API"""
    
    # Create a test image with a face (you can replace this with an actual image file)
    # For testing, we'll create a simple colored rectangle
    # In practice, you'd load an actual face image
    
    url = "http://localhost:8000/detect-landmarks"
    
    # Example: Load an image file for testing
    # image_path = "test_face.jpg"
    # with open(image_path, "rb") as f:
    #     files = {"file": ("test_face.jpg", f, "image/jpeg")}
    #     response = requests.post(url, files=files)
    
    # For demo purposes, let's just test the health endpoint
    health_response = requests.get("http://localhost:8000/health")
    print("Health check:", health_response.json())
    
    # Test root endpoint
    root_response = requests.get("http://localhost:8000/")
    print("Root endpoint:", root_response.json())

if __name__ == "__main__":
    test_api()
