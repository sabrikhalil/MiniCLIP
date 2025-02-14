import torch
import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel

class VisionEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224"):
        super(VisionEncoder, self).__init__()
        # The feature extractor is used only if the input is not already a tensor.
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)

    def forward(self, images):
        # If images is a torch.Tensor, assume it has already been preprocessed (resized, normalized, etc.)
        if isinstance(images, torch.Tensor):
            # Ensure the tensor is on the same device as the model.
            images = images.to(self.model.device)
            inputs = {"pixel_values": images}
        else:
            # Otherwise, use the feature extractor (which returns CPU tensors).
            inputs = self.feature_extractor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # Use the [CLS] token embedding as the image representation.
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings
