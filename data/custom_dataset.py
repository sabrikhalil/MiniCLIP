import os
from torch.utils.data import Dataset
from datasets import load_from_disk
from PIL import Image
from torchvision import transforms

class Flickr30kDataset(Dataset):
    def __init__(self, data_path, image_transform=None, max_caption_length=64, tokenizer=None):
        """
        Args:
            data_path (str): Path to the saved dataset on disk.
            image_transform (callable, optional): A function/transform to apply to the images.
            max_caption_length (int): Maximum token length for the caption.
            tokenizer (callable, optional): A tokenizer to convert captions into token ids.
        """
        self.dataset = load_from_disk(data_path)
        self.image_transform = image_transform if image_transform is not None else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # You can add normalization here if needed, e.g.:
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.max_caption_length = max_caption_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def process_caption(self, caption_list):
        """
        Given a list of captions for an image, select one caption.
        You might choose the first caption or randomly sample one.
        Optionally tokenize and pad/truncate the caption.
        """
        # For simplicity, we choose the first caption.
        caption = caption_list[0]

        if self.tokenizer is not None:
            # Tokenize and pad/truncate the caption to a fixed length.
            tokens = self.tokenizer(
                caption,
                max_length=self.max_caption_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            return tokens
        else:
            # Otherwise, simply return the caption string.
            return caption

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # Process the image (it's already a PIL Image)
        image = sample["image"]
        image = self.image_transform(image)
        # Process the caption(s)
        caption = self.process_caption(sample["caption"])
        # You might want to return additional metadata as well (e.g., filename, img_id)
        return image, caption

# Example usage (for testing):
if __name__ == "__main__":
    # Path to the saved dataset from download_dataset.py
    dataset_path = "data/flickr30k_train"
    
    # Optionally, if you have a tokenizer (e.g., from Hugging Face):
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Otherwise, leave it as None.
    tokenizer = None
    
    dataset = Flickr30kDataset(dataset_path, tokenizer=tokenizer)
    print("Number of samples:", len(dataset))
    img, cap = dataset[0]
    print("Image tensor shape:", img.shape)
    print("Caption output:", cap)
