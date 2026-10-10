import torch.nn as nn
import torch

from .transformer_utils.transformer_block import TransformerBlock


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, vocab_size, num_layers, num_heads, seq_dropout=0.1, ffn_dropout=0.1, attn_dropout=0.1, use_trainable_pos_embeds=False, max_seq_len=None):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        self.use_trainable_pos_embeds = use_trainable_pos_embeds
        self.pos_embedding = None

        if use_trainable_pos_embeds and max_seq_len is not None:
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.seq_dropout = nn.Dropout(seq_dropout)

        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                mlp_hidden_dim=4 * d_model,
                num_heads=num_heads,
                ffn_dropout=ffn_dropout,
                attn_dropout=attn_dropout
            ) for _ in range(num_layers)
        ])

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
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.normal_(module.in_proj_weight, mean=0.0, std=0.02)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)


    def _set_context(self, vocab_size, max_seq_len, **context):
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, self.d_model).to(self.token_embedding.weight.device)
        self._init_weights(self.token_embedding)

        self.max_seq_len = max_seq_len
        if self.use_trainable_pos_embeds:
            self.pos_embedding = nn.Embedding(max_seq_len, self.d_model)
            self._init_weights(self.pos_embedding)

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
    def __init__(self, decoder, tokenizer=None, beam_width=1, beam_temperature=1.0, use_prob_sampling=False, context_len=512, max_new_tokens=1024):
        super().__init__()

        self.context_len = context_len
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.beam_width = beam_width
        self.beam_temperature = beam_temperature
        self.use_prob_sampling = use_prob_sampling

    def _set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _set_context(self, **context):
        if hasattr(self.decoder, "_set_context"):
            self.decoder._set_context(**context)


    def forward(self, seq, **batch):
        # seq: [B, L] (already tokenized)

        return {'logits': self.decoder(seq, **batch)}

    @torch.no_grad()
    def predict(self, seq, **batch):
        # seq: [1, L] (already tokenized)

        eos_id = self.tokenizer.token_to_id["<eos>"]
        prompt_len = seq.shape[-1]

        beams = seq
        scores = torch.zeros(1, device=seq.device)
        finished = (beams[:, -1] == eos_id)

        for _ in range(self.max_new_tokens):
            if finished.all():
                break

            x = beams[:, -self.context_len:]
            logits = self.decoder(x, **batch)[:, -1, :]
            log_probs = torch.log_softmax(logits.float() / self.beam_temperature, dim=-1)

            log_probs = log_probs.masked_fill(finished[:, None], float("-inf"))
            log_probs[:, eos_id] = torch.where(finished, 0.0, log_probs[:, eos_id])

            candidate_scores = scores[:, None] + log_probs
            vocab_size = log_probs.shape[-1]

            if self.use_prob_sampling:
                flat_scores = candidate_scores.flatten()
                probabilities = torch.softmax(flat_scores, dim=0)

                k = min(self.beam_width, torch.count_nonzero(probabilities).item())
                indices = torch.multinomial(probabilities, num_samples=k)
                scores = flat_scores[indices]
            else:
                k = min(self.beam_width, candidate_scores.numel())
                scores, indices = candidate_scores.flatten().topk(k)

            parent_beam = indices // vocab_size
            new_tokens = indices % vocab_size

            beams = torch.cat([beams[parent_beam], new_tokens[:, None]], dim=1)
            finished = finished[parent_beam] | (new_tokens == eos_id)

        best_index = scores.argmax().item()
        best = beams[best_index:best_index + 1]
        generated = best[0, prompt_len:]
        eos_positions = torch.where(generated == eos_id)[0]

        if eos_positions.numel() > 0:
            first_eos_idx = eos_positions[0].item()
            prompt_and_generated = prompt_len + first_eos_idx + 1
            best = best[:, :prompt_and_generated]

        return {"seq": best}
