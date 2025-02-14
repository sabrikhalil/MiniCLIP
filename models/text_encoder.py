from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, model_name="gpt2"):
        super(TextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set the pad token if it is not already defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name)
    
    def forward(self, texts):
        # 'texts' is a list of strings.
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        # For simplicity, perform mean pooling over the sequence dimension.
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings
