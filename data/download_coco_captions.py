import os
from datasets import load_dataset

def main():
    os.makedirs("data", exist_ok=True)
    print("Loading coco_captions train split using datasets.load_dataset...")
    # For COCO Captions, the split is typically "train"
    dataset = load_dataset("coco_captions", split="train")
    print("Dataset loaded. Number of samples:", len(dataset))
    save_path = os.path.join("data", "coco_captions_train")
    dataset.save_to_disk(save_path)
    print("Dataset saved to disk at", save_path)

if __name__ == "__main__":
    main()
