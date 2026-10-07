import torch
from torch import nn


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=128,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    activation="relu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
        )

    def forward(self, data_object, **batch):
        seq_len = data_object.size(1)
        positions = torch.arange(seq_len, device=data_object.device)

        x = self.token_embedding(data_object) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, src_mask=self.mask[:seq_len, :seq_len])
        x = self.final_layer_norm(x)

        logits = self.lm_head(x)
        return {"logits": logits}
