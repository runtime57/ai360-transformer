import torch
from torch import nn


class TransformerModel(nn.Module):
    """
    Decoder-only (GPT-style) transformer for character-level language
    modelling. Built from nn.TransformerEncoderLayer blocks with a causal
    mask, so position t only attends to positions <= t.
    """

    def __init__(
        self,
        vocab_size,
        block_size=128,
        d_model=128,
        n_heads=4,
        n_layers=4,
        dropout=0.1
    ):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model)
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, data_object, **batch):
        seq_len = data_object.size(1)
        positions = torch.arange(seq_len, device=data_object.device)
        x = self.token_embedding(data_object) + self.position_embedding(positions)
        x = self.dropout(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=data_object.device
        )
        x = self.blocks(x, mask=causal_mask)

        return {"logits": self.lm_head(x)}

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
