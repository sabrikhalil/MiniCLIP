import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from models.vision_encoder import VisionEncoder
from models.text_encoder import TextEncoder
from data.custom_dataset import Flickr30kDataset, CocoCaptionsDataset, get_combined_train_dataset
from src.downstream_task import zero_shot_classification

# --------------------------
# Projection Head Module
# --------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ProjectionHead, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.linear(x)

# --------------------------
# Contrastive Loss Function (Fixed Temperature)
# --------------------------
def compute_contrastive_loss(img_feats, txt_feats, fixed_scale, criterion, device):
    logits = torch.matmul(img_feats, txt_feats.t()) * fixed_scale
    batch_size = logits.size(0)
    targets = torch.arange(batch_size).to(device)
    loss_img = criterion(logits, targets)
    loss_txt = criterion(logits.t(), targets)
    loss = (loss_img + loss_txt) / 2.0
    acc_img = (logits.argmax(dim=1) == targets).float().mean().item()
    acc_txt = (logits.t().argmax(dim=1) == targets).float().mean().item()
    return loss, loss_img, loss_txt, acc_img, acc_txt

# --------------------------
# Validation Function (Limited Batches)
# --------------------------
def run_validation(vision_encoder, text_encoder, image_proj, text_proj, val_loader, criterion, device, fixed_scale, max_batches=4):
    vision_encoder.eval()
    text_encoder.eval()
    image_proj.eval()
    text_proj.eval()
    
    total = {"loss": 0.0, "loss_img": 0.0, "loss_txt": 0.0, "acc_img": 0.0, "acc_txt": 0.0}
    num_batches = 0
    with torch.no_grad():
        for idx, (images, captions) in enumerate(val_loader):
            if idx >= max_batches:
                break
            images = images.to(device)
            # Get encoder outputs and apply projection heads.
            img_feats = vision_encoder(images)
            txt_feats = text_encoder(captions)
            img_feats = image_proj(img_feats)
            txt_feats = text_proj(txt_feats)
            
            # Normalize embeddings.
            img_feats = img_feats / img_feats.norm(dim=1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=1, keepdim=True)
            
            loss, loss_img, loss_txt, acc_img, acc_txt = compute_contrastive_loss(
                img_feats, txt_feats, fixed_scale, criterion, device
            )
            total["loss"] += loss.item()
            total["loss_img"] += loss_img.item()
            total["loss_txt"] += loss_txt.item()
            total["acc_img"] += acc_img
            total["acc_txt"] += acc_txt
            num_batches += 1
    return {k: v / num_batches for k, v in total.items()}

# --------------------------
# Checkpoint Saving
# --------------------------
def save_checkpoint(vision_encoder, text_encoder, image_proj, text_proj, epoch, global_step, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(vision_encoder.state_dict(), os.path.join(checkpoint_dir, f"vision_encoder_e{epoch}_s.pt"))
    torch.save(text_encoder.state_dict(), os.path.join(checkpoint_dir, f"text_encoder_e{epoch}_s.pt"))
    torch.save(image_proj.state_dict(), os.path.join(checkpoint_dir, f"image_proj_e{epoch}_s.pt"))
    torch.save(text_proj.state_dict(), os.path.join(checkpoint_dir, f"text_proj_e{epoch}_s.pt"))
    print(f"Checkpoint saved at step {global_step} (epoch {epoch}).")

# --------------------------
# Main Training Loop
# --------------------------
def main():
    # Configuration parameters.
    default_mode = "combined"  # "combined" for Flickr30k + COCO train; "coco" for COCO alone.
    wandb.init(
        project="MiniCLIP",
        entity="khalil-sabri01",
        config={
            "learning_rate": 1e-4,
            "batch_size": 32,
            "num_epochs": 100,
            "temperature_init": 0.07,  # fixed temperature value
            "proj_dim": 512,
            "train_val_split": 0.9,
            "val_interval": 50,       # Validate & checkpoint every 100 iterations.
            "downstream_interval": 500,  # Run downstream evaluation every 500 iterations.
            "max_val_batches": 10,
            "training_mode": default_mode
        }
    )
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize encoders.
    vision_encoder = VisionEncoder().to(device)
    text_encoder = TextEncoder().to(device)

    # Determine output dimensions.
    dummy_img = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        dummy_img_feat = vision_encoder(dummy_img)
    img_dim = dummy_img_feat.size(-1)
    dummy_txt = ["dummy text"]
    with torch.no_grad():
        dummy_txt_feat = text_encoder(dummy_txt)
    text_dim = dummy_txt_feat.size(-1)

    # Create projection heads.
    image_proj = ProjectionHead(img_dim, config.proj_dim).to(device)
    text_proj = ProjectionHead(text_dim, config.proj_dim).to(device)

    # Fixed temperature scale.
    fixed_scale = 1 / config.temperature_init  # e.g., 1/0.07 ~ 14.2857

    # Set up training dataset.
    if config.training_mode == "combined":
        flickr_path = "data/flickr30k_train"
        coco_train_path = "data/coco_captions_train"
        if not os.path.exists(flickr_path) or not os.path.exists(coco_train_path):
            print("Training datasets not found. Running download_datasets.py...")
            os.system("python data/download_datasets.py")
        # Note: No transforms are passed here since the dataset uses its own defaults.
        train_dataset = get_combined_train_dataset(
            flickr_path, 
            coco_train_path, 
            max_caption_length=64, 
            tokenizer=None
        )
    elif config.training_mode == "coco":
        coco_train_path = "data/coco_captions_train"
        if not os.path.exists(coco_train_path):
            print("COCO train dataset not found. Running download_datasets.py...")
            os.system("python data/download_datasets.py")
        from data.custom_dataset import CocoCaptionsDataset
        train_dataset = CocoCaptionsDataset(
            coco_train_path, 
            mode="train",
            max_caption_length=64, 
            tokenizer=None
        )
    else:
        raise ValueError("Unknown training_mode. Use 'combined' or 'coco'.")

    print(f"Total training samples: {len(train_dataset)}")
    wandb.log({"train_dataset_size": len(train_dataset)})

    # Set up validation dataset: using COCO validation.
    coco_val_path = "data/coco_captions_validation"
    if not os.path.exists(coco_val_path):
        print("COCO validation dataset not found. Running download_datasets.py...")
        os.system("python data/download_datasets.py")
    from data.custom_dataset import CocoCaptionsDataset
    val_dataset = CocoCaptionsDataset(
        coco_val_path, 
        mode="val",
        max_caption_length=64, 
        tokenizer=None
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)

    optimizer = optim.Adam(
        list(vision_encoder.parameters()) + list(text_encoder.parameters()) +
        list(image_proj.parameters()) + list(text_proj.parameters()),
        lr=config.learning_rate
    )
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(1, config.num_epochs + 1):
        print(f"Epoch {epoch} starting...")
        vision_encoder.train()
        text_encoder.train()
        image_proj.train()
        text_proj.train()
        
        for images, captions in train_loader:
            global_step += 1
            images = images.to(device)

            img_feats = vision_encoder(images)
            txt_feats = text_encoder(captions)

            # Apply projection heads.
            img_feats = image_proj(img_feats)
            txt_feats = text_proj(txt_feats)

            # Normalize embeddings.
            img_feats = img_feats / img_feats.norm(dim=1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=1, keepdim=True)

            loss, loss_img, loss_txt, acc_img, acc_txt = compute_contrastive_loss(
                img_feats, txt_feats, fixed_scale, criterion, device
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({
                "global_step": global_step,
                "train_loss": loss.item(),
                "train_loss_image": loss_img.item(),
                "train_loss_text": loss_txt.item(),
                "train_acc_image": acc_img,
                "train_acc_text": acc_txt,
                "epoch": epoch,
                "fixed_scale": fixed_scale,
            })
            print(f"Step {global_step}: Loss={loss.item():.4f}, Img Loss={loss_img.item():.4f}, Text Loss={loss_txt.item():.4f}, Img Acc={acc_img*100:.2f}%, Text Acc={acc_txt*100:.2f}%")

            if global_step % config.val_interval == 0:
                val_metrics = run_validation(vision_encoder, text_encoder, image_proj, text_proj,
                                             val_loader, criterion, device, fixed_scale, max_batches=config.max_val_batches)
                wandb.log({
                    "global_step": global_step,
                    "val_loss": val_metrics["loss"],
                    "val_loss_image": val_metrics["loss_img"],
                    "val_loss_text": val_metrics["loss_txt"],
                    "val_acc_image": val_metrics["acc_img"],
                    "val_acc_text": val_metrics["acc_txt"],
                })
                print(f"Validation at step {global_step}: Loss={val_metrics['loss']:.4f}, Img Acc={val_metrics['acc_img']*100:.2f}%, Text Acc={val_metrics['acc_txt']*100:.2f}%")
                save_checkpoint(vision_encoder, text_encoder, image_proj, text_proj, epoch, global_step)
            
            if global_step % config.downstream_interval == 0:
                ds_acc = zero_shot_classification(vision_encoder, text_encoder, image_proj, text_proj, fixed_scale, device)
                wandb.log({"global_step": global_step, "downstream_accuracy": ds_acc})
                print(f"Downstream (zero-shot CIFAR-10) accuracy at step {global_step}: {ds_acc:.2f}%")

    wandb.finish()

if __name__ == "__main__":
    main()
