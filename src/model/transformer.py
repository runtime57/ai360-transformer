import torch
from torch import nn


class KekTransformer(nn.Module):
    def __init__(
        self, vocab_size, max_len=128, d_model=128, n_heads=4, n_layers=4, dropout=0.1
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=n_layers, norm=nn.LayerNorm(d_model)
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, data_object, **batch):
        seq_len = data_object.shape[1]
        positions = torch.arange(seq_len, device=data_object.device)
        x = self.token_embedding(data_object) + self.position_embedding(positions)
        x = self.dropout(x)

        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        mask = mask.triu(diagonal=1)

        x = self.blocks(x, mask=mask)
        return {"logits": self.head(x)}
