import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism warning

from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch

class TextEncoder(nn.Module):
    def __init__(self, model_name="gpt2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # CLIP-style EOS handling
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "<|endoftext|>"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, texts):
        # Add EOS token to each text input
        texts = [t + self.tokenizer.eos_token for t in texts]
        
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77  # CLIP uses a 77-token limit
        ).to(self.model.device)

        outputs = self.model(**inputs)
        
        # Get EOS token position (last non-pad token)
        eos_pos = inputs.attention_mask.sum(dim=1) - 1  # (batch_size,)
        
        # Gather EOS embeddings using advanced indexing
        return outputs.last_hidden_state[torch.arange(len(eos_pos)), eos_pos]
