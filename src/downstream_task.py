import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def zero_shot_classification(vision_encoder, text_encoder, image_proj, text_proj, fixed_scale, device, batch_size=64):
    """
    Performs zero-shot classification on CIFAR-10 using the current model.
    
    Args:
      vision_encoder: the vision encoder module.
      text_encoder: the text encoder module.
      image_proj: projection head for image features.
      text_proj: projection head for text features.
      fixed_scale (float): the fixed scaling factor (e.g., 1/temperature_init).
      device: torch.device.
      batch_size: batch size for CIFAR-10 evaluation.
    
    Returns:
      Accuracy (percentage) on CIFAR-10 test set.
    """
    # CIFAR-10 class names.
    class_names = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
    # Create text prompts.
    prompts = [f"a photo of a {cls}" for cls in class_names]
    
    # Compute text embeddings.
    with torch.no_grad():
        txt_embeddings = text_encoder(prompts)
        txt_embeddings = text_proj(txt_embeddings)
    txt_embeddings = txt_embeddings / txt_embeddings.norm(dim=1, keepdim=True)
    
    # Define CIFAR-10 transform (resize to 224 and normalize like ImageNet).
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    # Load CIFAR-10 test set.
    cifar_dataset = datasets.CIFAR10(root="./data/cifar10", train=False, download=True, transform=transform)
    test_loader = DataLoader(cifar_dataset, batch_size=batch_size, shuffle=False)
    
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            img_embeddings = vision_encoder(images)
            img_embeddings = image_proj(img_embeddings)
            img_embeddings = img_embeddings / img_embeddings.norm(dim=1, keepdim=True)
            # Use the fixed scale (which is a float) directly.
            logits = torch.matmul(img_embeddings, txt_embeddings.t()) * fixed_scale
            preds = logits.argmax(dim=1)
            total_correct += (preds.cpu() == labels).sum().item()
            total_samples += images.size(0)
    
    accuracy = total_correct / total_samples * 100
    return accuracy
