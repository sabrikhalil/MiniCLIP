from transformers import ViTFeatureExtractor, ViTModel
import torch.nn as nn

class VisionEncoder(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224"):
        super(VisionEncoder, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)

    def forward(self, images):
        # 'images' should be a list of PIL images.
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        outputs = self.model(**inputs)
        # Use the [CLS] token embedding as the image representation.
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings
