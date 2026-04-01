import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = f"cuda:{configs.gpu}" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)

        # 使用 GPT-2 替代 LLaMA
        self.gpt2 = GPT2LMHeadModel.from_pretrained(configs.llm_ckp_dir).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(configs.llm_ckp_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = self.tokenizer.vocab_size
        self.hidden_dim_of_gpt2 = self.gpt2.config.n_embd

        # 冻结 GPT-2 参数
        for param in self.gpt2.parameters():
            param.requires_grad = False

    def tokenize_input(self, x):
        """
        x: list of strings, batch of sequences
        """
        tokens = [self.tokenizer(text, return_tensors="pt", padding=True, truncation=True) for text in x]
        input_ids = torch.cat([t['input_ids'] for t in tokens], dim=0).to(self.device)
        embeddings = self.gpt2.transformer.wte(input_ids)
        return embeddings

    def forecast(self, x_mark_enc):
        """
        x_mark_enc: list of strings
        """
        x_mark_emb = self.tokenize_input(x_mark_enc)
        outputs = self.gpt2(inputs_embeds=x_mark_emb, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [batch, hidden_dim]
        return last_hidden

    def forward(self, x_mark_enc):
        return self.forecast(x_mark_enc)
