from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, model_name="gpt2"):
        super(TextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure a padding token is set (using eos_token if not present)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name)
    
    def forward(self, texts):
        # Tokenize the list of text strings.
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        # Move all input tensors to the same device as the model.
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # For simplicity, perform mean pooling over the sequence dimension.
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings
