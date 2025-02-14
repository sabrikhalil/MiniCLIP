import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import wandb

from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from data.custom_dataset import Flickr30kDataset

def compute_contrastive_loss(image_features, text_features, temperature, criterion, device):
    """
    Computes InfoNCE loss for image–text pairs.
    Returns:
      loss, loss_image, loss_text, acc_image, acc_text
    """
    # Compute cosine similarity matrix scaled by temperature.
    logits = torch.matmul(image_features, text_features.t()) / temperature
    batch_size = logits.size(0)
    targets = torch.arange(batch_size).to(device)

    loss_image = criterion(logits, targets)
    loss_text = criterion(logits.t(), targets)
    loss = (loss_image + loss_text) / 2.0

    # Retrieval accuracies.
    pred_image = logits.argmax(dim=1)
    acc_image = (pred_image == targets).float().mean().item()

    pred_text = logits.t().argmax(dim=1)
    acc_text = (pred_text == targets).float().mean().item()

    return loss, loss_image, loss_text, acc_image, acc_text

def run_validation(vision_encoder, text_encoder, val_loader, criterion, device, temperature):
    """Run validation on the entire validation set and return average metrics."""
    vision_encoder.eval()
    text_encoder.eval()
    total_loss = 0.0
    total_loss_image = 0.0
    total_loss_text = 0.0
    total_acc_image = 0.0
    total_acc_text = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, captions in val_loader:
            images = images.to(device)
            text_features = text_encoder(captions)
            image_features = vision_encoder(images)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            loss, loss_image, loss_text, acc_image, acc_text = compute_contrastive_loss(
                image_features, text_features, temperature, criterion, device
            )
            total_loss += loss.item()
            total_loss_image += loss_image.item()
            total_loss_text += loss_text.item()
            total_acc_image += acc_image
            total_acc_text += acc_text
            num_batches += 1

    avg_metrics = {
        "val_loss": total_loss / num_batches,
        "val_loss_image": total_loss_image / num_batches,
        "val_loss_text": total_loss_text / num_batches,
        "val_acc_image": total_acc_image / num_batches,
        "val_acc_text": total_acc_text / num_batches,
    }
    return avg_metrics

def main():
    # Initialize WandB for experiment tracking.
    wandb.init(
        project="MiniCLIP",
        entity="khalil-sabri01",
        config={
            "learning_rate": 1e-4,
            "batch_size": 32,
            "num_epochs": 5,
            "temperature": 0.07,
            "train_val_split": 0.9,
        }
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize encoders.
    vision_encoder = VisionEncoder().to(device)
    text_encoder = TextEncoder().to(device)

    # Define image transform.
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load the Flickr30k dataset saved to disk.
    dataset_path = "data/flickr30k_train"
    full_dataset = Flickr30kDataset(dataset_path, image_transform=image_transform, tokenizer=None)
    train_size = int(len(full_dataset) * config.train_val_split)
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    optimizer = optim.Adam(list(vision_encoder.parameters()) + list(text_encoder.parameters()), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(config.num_epochs):
        vision_encoder.train()
        text_encoder.train()
        for images, captions in train_loader:
            global_step += 1
            images = images.to(device)

            # Forward pass through encoders.
            image_features = vision_encoder(images)  # shape: (B, d)
            text_features = text_encoder(captions)     # text encoder moves its inputs to device internally

            # Normalize embeddings.
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # Compute loss and accuracy.
            loss, loss_image, loss_text, acc_image, acc_text = compute_contrastive_loss(
                image_features, text_features, config.temperature, criterion, device
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log metrics for the current iteration.
            wandb.log({
                "global_step": global_step,
                "train_loss": loss.item(),
                "train_loss_image": loss_image.item(),
                "train_loss_text": loss_text.item(),
                "train_acc_image": acc_image,
                "train_acc_text": acc_text,
                "epoch": epoch + 1,
            })
            print(f"Step {global_step}: Loss={loss.item():.4f}, Img Loss={loss_image.item():.4f}, Text Loss={loss_text.item():.4f}, Img Acc={acc_image*100:.2f}%, Text Acc={acc_text*100:.2f}%")

            # Every 10 iterations, run full validation.
            if global_step % 10 == 0:
                val_metrics = run_validation(vision_encoder, text_encoder, val_loader, criterion, device, config.temperature)
                wandb.log({
                    "global_step": global_step,
                    **val_metrics
                })
                print(f"Validation at step {global_step}: Loss={val_metrics['val_loss']:.4f}, Img Acc={val_metrics['val_acc_image']*100:.2f}%, Text Acc={val_metrics['val_acc_text']*100:.2f}%")
    wandb.finish()

if __name__ == "__main__":
    main()
