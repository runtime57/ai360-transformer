import torch.nn as nn
import torch


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, seq_dropout, vocab_size, num_layers, num_heads, dropout, use_trainable_pos_embeds=False, max_seq_len=None):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = None
        if use_trainable_pos_embeds:
            assert max_seq_len is not None
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.seq_dropout = nn.Dropout(seq_dropout)

        self.transformer_decoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                batch_first=True,
                dropout=dropout
            ),
            num_layers=num_layers
        )

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
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)


    def _set_context(self, vocab_size, **context):
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, self.d_model, device=self.token_embedding.device)
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

        causal_mask = torch.ones(L, L, dtype=torch.bool, device=device).triu(diagonal=1)

        embeds = self.transformer_decoder(
            seq,
            mask=causal_mask,
            is_causal=True
        )

        logits = embeds @ self.token_embedding.weight.T

        return logits


class GPT(nn.Module):
    def __init__(self, decoder, tokenizer=None, beam_width=1, max_output_tokens=1024):
        super().__init__()

        self.decoder = decoder
        self.tokenizer = tokenizer

    def _set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _set_context(self, **context):
        if hasattr(self.decoder, "_set_context"):
            self.decoder._set_context(**context)


    def forward(self, seq, **batch):
        # seq: [B, L] (already tokenized)

        return {'logits': self.decoder(seq, **batch)}

    # def predict(self, seq, **batch):
    #     # seq: [1, L] (already tokenized) ?

    #     logits = self.decoder(seq, **batch)
