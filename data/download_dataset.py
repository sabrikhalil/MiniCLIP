import os
from datasets import load_dataset

def main():
    os.makedirs("data", exist_ok=True)
    print("Loading nlphuji/flickr30k train split using datasets.load_dataset...")
    
    # This will automatically download all necessary files (images, metadata, etc.)
    dataset = load_dataset("nlphuji/flickr30k", split="test")
    
    print("Dataset loaded. Number of samples:", len(dataset))
    
    # Optionally, save the dataset locally for faster future loading.
    # This saves it in an Arrow format (usually several GB as expected).
    save_path = os.path.join("data", "flickr30k_train")
    dataset.save_to_disk(save_path)
    print("Dataset saved to disk at", save_path)

if __name__ == "__main__":
    main()
