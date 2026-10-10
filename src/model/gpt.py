import torch.nn as nn
import torch

from .transformer_utils.rmsn import RMSNorm
from .transformer_utils.transformer_block import TransformerBlock


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, vocab_size, num_layers, num_heads, seq_dropout=0.1, ffn_dropout=0.1, attn_dropout=0.1, norm="rmsn", use_trainable_pos_embeds=False, max_seq_len=None):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = None
        if use_trainable_pos_embeds:
            assert max_seq_len is not None
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.seq_dropout = nn.Dropout(seq_dropout)

        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                mlp_hidden_dim=4 * d_model,
                num_heads=num_heads,
                ffn_dropout=ffn_dropout,
                attn_dropout=attn_dropout,
                norm=norm
            ) for _ in range(num_layers)
        ])
        if norm == "rmsn":
            self.final_norm = RMSNorm(d_model)
        else:
            self.final_norm = nn.LayerNorm(d_model)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight.data[module.padding_idx].zero()
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, RMSNorm):
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)


    def _set_context(self, vocab_size, **context):
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, self.d_model).to(self.token_embedding.weight.device)
        self._init_weights(self.token_embedding)

    def forward(self, seq, **batch):
        # seq: [B, L] (already tokenized)

        B, L = seq.shape
        device = seq.device

        pos = torch.arange(L, device=device)
        if self.pos_embedding is not None:
            pos_embed = self.pos_embedding(pos)
        else:
            dims = torch.arange(self.d_model, device=device)
            angles = pos.unsqueeze(-1) / 10000**(2 * (dims // 2) / self.d_model)

            pos_embed = torch.where(
                dims % 2 == 0,
                torch.sin(angles),
                torch.cos(angles),
            )

        seq = self.d_model**0.5 * self.token_embedding(seq)
        seq = seq + pos_embed
        seq = self.seq_dropout(seq)

        for block in self.transformer_blocks:
            seq = block(seq)

        embeds = self.final_norm(seq)
        logits = embeds @ self.token_embedding.weight.T

        return logits


class GPT(nn.Module):
    def __init__(self, decoder, tokenizer=None, beam_width=1, context_len=256, max_new_tokens=1024):
        super().__init__()

        self.context_len = context_len
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def _set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _set_context(self, **context):
        if hasattr(self.decoder, "_set_context"):
            self.decoder._set_context(**context)


    def forward(self, seq, **batch):
        # seq: [B, L] (already tokenized)
        return {'logits': self.decoder(seq, **batch)}

    def predict(self, seq, **batch):
        # seq: [1, L] (already tokenized)

        added_tokens = 0
        while seq[-1] != self.tokenizer.token_to_id['<eos>'] and added_tokens < self.max_new_tokens:
            x = seq[:, -self.context_len:]
            logits = self.decoder(x, **batch)[1, -1, :]
            best = logits.argmax()
            seq = torch.cat([seq, best])
            added_tokens += 1

        return {"seq": seq}
