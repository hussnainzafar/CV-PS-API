import requests
import os
from pathlib import Path

def test_replicate_api():
    """Test the Replicate-powered facial landmarks API"""
    
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    print("Testing health endpoint...")
    health_response = requests.get(f"{base_url}/health")
    print("Health check:", health_response.json())
    
    # Test models endpoint
    print("\nTesting models endpoint...")
    models_response = requests.get(f"{base_url}/models")
    print("Available models:", models_response.json())
    
    # Test root endpoint
    print("\nTesting root endpoint...")
    root_response = requests.get(f"{base_url}/")
    print("Root endpoint:", root_response.json())
    
    # Test landmark detection (you need to provide an actual image file)
    # Uncomment and modify the path below to test with a real image
    """
    image_path = "test_face.jpg"  # Replace with your image path
    if Path(image_path).exists():
        print(f"\nTesting landmark detection with {image_path}...")
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, "image/jpeg")}
            response = requests.post(f"{base_url}/detect-landmarks", files=files)
        
        if response.status_code == 200:
            # Save the result
            with open("result_replicate_landmarks.jpg", "wb") as f:
                f.write(response.content)
            print("✅ Landmark detection successful! Result saved as 'result_replicate_landmarks.jpg'")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    else:
        print(f"⚠️  Test image {image_path} not found. Please provide a test image to test landmark detection.")
    """

if __name__ == "__main__":
    test_replicate_api()
