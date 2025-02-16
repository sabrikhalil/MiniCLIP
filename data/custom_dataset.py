import os
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_from_disk
from PIL import Image
from torchvision import transforms
import random

class BaseCaptionDataset(Dataset):
    def __init__(
        self, 
        data_path, 
        mode="train", 
        image_transform_train=None, 
        image_transform_val=None, 
        max_caption_length=64, 
        tokenizer=None
    ):
        """
        Args:
            data_path (str): Path to the saved dataset on disk.
            mode (str): "train" or "val" (or "test") to choose the correct image transformation.
            image_transform_train (callable, optional): Transform to apply to training images.
            image_transform_val (callable, optional): Transform to apply to validation/test images.
            max_caption_length (int): Maximum token length for the caption.
            tokenizer (callable, optional): Tokenizer to convert captions.
        """
        self.dataset = load_from_disk(data_path)
        self.mode = mode
        
        if self.mode == "train":
            self.image_transform = image_transform_train if image_transform_train is not None else transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:  # For validation or test, use a fixed transform.
            self.image_transform = image_transform_val if image_transform_val is not None else transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        self.max_caption_length = max_caption_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def process_caption(self, caption_field):
        # If the caption is a list (e.g., in Flickr30k), randomly select one caption.
        if isinstance(caption_field, list):
            return random.choice(caption_field)
        return caption_field

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        # Ensure the image is in RGB.
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)
        # Process the caption.
        caption = self.process_caption(sample["caption"])
        # If a tokenizer is provided, tokenize the caption.
        if self.tokenizer is not None:
            tokens = self.tokenizer(
                caption,
                max_length=self.max_caption_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            # Remove the batch dimension.
            caption = {k: v.squeeze(0) for k, v in tokens.items()}
        return image, caption

class Flickr30kDataset(BaseCaptionDataset):
    pass

class CocoCaptionsDataset(BaseCaptionDataset):
    pass

def get_combined_train_dataset(
    flickr_path, 
    coco_path, 
    image_transform_train=None, 
    image_transform_val=None, 
    max_caption_length=64, 
    tokenizer=None
):
    flickr_ds = Flickr30kDataset(
        flickr_path, 
        mode="train", 
        image_transform_train=image_transform_train, 
        image_transform_val=image_transform_val, 
        max_caption_length=max_caption_length, 
        tokenizer=tokenizer
    )
    coco_ds = CocoCaptionsDataset(
        coco_path, 
        mode="train", 
        image_transform_train=image_transform_train, 
        image_transform_val=image_transform_val, 
        max_caption_length=max_caption_length, 
        tokenizer=tokenizer
    )
    combined_ds = ConcatDataset([flickr_ds, coco_ds])
    return combined_ds
