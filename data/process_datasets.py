import os
from torch.utils.data import DataLoader
from torchvision import transforms
from data.custom_dataset import Flickr30kDataset, CocoCaptionsDataset, get_combined_train_dataset

def main():
    # Define a common transform that includes normalization.


    print("=== Inspecting Flickr30k Dataset ===")
    flickr_path = "data/flickr30k_train"
    if os.path.exists(flickr_path):
        flickr_ds = Flickr30kDataset(flickr_path,  tokenizer=None)
        print("Flickr30k - Number of samples:", len(flickr_ds))
        # Fetch a batch of 2 samples.
        dl = DataLoader(flickr_ds, batch_size=2, shuffle=True)
        images, captions = next(iter(dl))
        print("Batch images shape:", images.shape)
        print("Batch captions:", captions)
    else:
        print("Flickr30k dataset not found.")

    print("\n=== Inspecting Combined Dataset (Flickr30k + COCO Captions) ===")
    coco_train_path = "data/coco_captions_train"
    if os.path.exists(flickr_path) and os.path.exists(coco_train_path):
        combined_ds = get_combined_train_dataset(flickr_path, coco_train_path, tokenizer=None)
        print("Combined Dataset - Number of samples:", len(combined_ds))
        dl_combined = DataLoader(combined_ds, batch_size=2, shuffle=True)
        images, captions = next(iter(dl_combined))
        print("Combined batch images shape:", images.shape)
        print("Combined batch captions:", captions)
    else:
        print("One or both datasets (Flickr30k or COCO) not found.")

if __name__ == "__main__":
    main()
