from torch import nn
import torch, math

class nnTransformer(nn.Module):

    def __init__(self, heads=4, dmodel=256, ffdim=1024,
                  vocabsize=65, max_len=512):
        super().__init__()
        self.src_emb = nn.Embedding(vocabsize, dmodel)
        self.tgt_emb = nn.Embedding(vocabsize, dmodel)
        self.pos_emb = nn.Embedding(max_len, dmodel)
        self.d_model = dmodel
        layer = nn.TransformerEncoderLayer(
            d_model=dmodel,
            nhead=heads,
            dim_feedforward=ffdim
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=layer, num_layers=1)
        self.out = nn.Linear(dmodel, vocabsize)
        
    def _embed(self, tokens, emb):
        pos = torch.arange(tokens.size(1), device=tokens.device)
        return emb(tokens) * math.sqrt(self.d_model) + self.pos_emb(pos)

    def forward(self, src, **batch):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            src.size(1), device=src.device)
        h = self.transformer(
            src = self._embed(src, self.tgt_emb),
            mask=tgt_mask,
            src_key_padding_mask=tgt_mask,
            batch_first=True)
        return {"logits": self.out(h)}

    

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
