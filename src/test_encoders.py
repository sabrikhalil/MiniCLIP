import torch
from PIL import Image
from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder

def main():
    # ---- Test Vision Encoder ----
    # Make sure to place a sample image in the data folder named "sample.jpg"
    print("cuda is available : ", torch.cuda.is_available())
    image_path = "data/cat.jpg"
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    # Process a batch with one image.
    images = [image]
    
    # Initialize the vision encoder.
    vision_encoder = VisionEncoder()
    image_features = vision_encoder(images)
    print("Image features shape:", image_features.shape)
    # Expected shape: (1, vision_hidden_size), e.g., (1, 768)

    # ---- Test Text Encoder ----
    sample_texts = ["A photo of a cat.", "A photo of a dog."]
    text_encoder = TextEncoder()
    text_features = text_encoder(sample_texts)
    print("Text features shape:", text_features.shape)
    # Expected shape: (2, text_hidden_size), e.g., (2, 768) for GPT-2 (it may vary)

    # Optionally: Check if shapes match.
    if image_features.shape[1] != text_features.shape[1]:
        print("Note: Feature dimensions do not match. Consider adding a projection layer to align them.")
    else:
        print("Feature dimensions match!")

if __name__ == "__main__":
    main()
