import torch
from torch import nn


class TextGenTransformer(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, d: int = 128, num_heads: int = 4, num_layers: int = 4, dropout: float = 0.1, dim_ffn: int | None = None):
        super().__init__()

        if dim_ffn is None:
            dim_ffn = 4 * d

        self.embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d)
        # learned positional embedding, one vector per position in the block
        self.pos_embedder = nn.Embedding(num_embeddings=max_len, embedding_dim=d)
        self.dropout = nn.Dropout(dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=num_heads,
            dim_feedforward=dim_ffn,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d),
        )
        self.decoder = nn.Linear(in_features=d, out_features=vocab_size)

    def forward(self, data_object, **batch):
        # data_object: (B, T) token ids
        T = data_object.size(1)
        positions = torch.arange(T, device=data_object.device)
        x = self.dropout(self.embedder(data_object) + self.pos_embedder(positions))
        # causal mask: position t may only attend to positions <= t,
        # otherwise the model just reads the next token it is asked to predict
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=data_object.device)
        x = self.transformer_encoder(x, mask=mask, is_causal=True)
        return {"logits": self.decoder(x)}  # (B, T, vocab_size)
