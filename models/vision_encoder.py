import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTModel

class VisionEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224"):
        super(VisionEncoder, self).__init__()
        # Initialize the image processor with do_rescale=False because our images are pre-scaled ([0,1] range)
        self.image_processor = ViTImageProcessor.from_pretrained(model_name, do_rescale=False)
        self.model = ViTModel.from_pretrained(model_name)

    def forward(self, images):
        # If images are already a torch.Tensor (e.g., produced by transforms.ToTensor()),
        # we assume they are preprocessed and in the [0, 1] range.
        if isinstance(images, torch.Tensor):
            images = images.to(self.model.device)
            inputs = {"pixel_values": images}
        else:
            # Otherwise, use the image processor to convert PIL images into tensors.
            inputs = self.image_processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        # Use the [CLS] token embedding as the image representation.
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings
