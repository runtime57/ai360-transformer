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
        layer = nn.TransformerDecoderLayer(
            d_model=dmodel,
            nhead=heads,
            dim_feedforward=ffdim
        )
        self.transformer = nn.TransformerDecoder(decoder_layer=layer, num_layers=1)
        self.out = nn.Linear(dmodel, vocabsize)
        
    def _embed(self, tokens, emb):
        pos = torch.arange(tokens.size(1), device=tokens.device)
        return emb(tokens) * math.sqrt(self.d_model) + self.pos_emb(pos)

    def forward(self, src, tgt, src_pad_mask=None, tgt_pad_mask=None, **batch):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1), device=tgt.device)
        h = self.transformer(
            self._embed(src, self.src_emb), self._embed(tgt, self.tgt_emb),
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask)
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
