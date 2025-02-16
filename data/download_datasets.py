import os
from datasets import load_dataset

def download_flickr30k():
    os.makedirs("data", exist_ok=True)
    print("Downloading Flickr30k dataset using nlphuji/flickr30k split 'test' (only split available)...")
    # Note: nlphuji/flickr30k only exposes a "test" split – we use that as our training data.
    dataset = load_dataset("nlphuji/flickr30k", split="test")
    print("Flickr30k loaded. Number of samples:", len(dataset))
    save_path = os.path.join("data", "flickr30k_train")
    dataset.save_to_disk(save_path)
    print("Flickr30k saved to disk at", save_path)

def download_coco():
    os.makedirs("data", exist_ok=True)
    print("Downloading COCO Captions dataset using jxie/coco_captions...")
    # For COCO captions, we assume splits "train", "validation", and "test" exist.
    try:
        dataset_train = load_dataset("jxie/coco_captions", split="train")
    except Exception as e:
        print("Error loading COCO Captions train split. Make sure you are logged in using 'huggingface-cli login'.")
        raise e
    print("COCO Captions train loaded. Number of samples:", len(dataset_train))
    save_path_train = os.path.join("data", "coco_captions_train")
    dataset_train.save_to_disk(save_path_train)
    print("COCO Captions train saved to disk at", save_path_train)

    print("Downloading COCO Captions validation split...")
    dataset_val = load_dataset("jxie/coco_captions", split="validation")
    print("COCO Captions validation loaded. Number of samples:", len(dataset_val))
    save_path_val = os.path.join("data", "coco_captions_validation")
    dataset_val.save_to_disk(save_path_val)
    print("COCO Captions validation saved to disk at", save_path_val)

    print("Downloading COCO Captions test split...")
    dataset_test = load_dataset("jxie/coco_captions", split="test")
    print("COCO Captions test loaded. Number of samples:", len(dataset_test))
    save_path_test = os.path.join("data", "coco_captions_test")
    dataset_test.save_to_disk(save_path_test)
    print("COCO Captions test saved to disk at", save_path_test)

def main():
    download_flickr30k()
    download_coco()

if __name__ == "__main__":
    main()
