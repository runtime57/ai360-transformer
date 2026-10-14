"""Numerical and compatibility checks for the linear attention integration."""

import io
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from datasets import Dataset, DatasetDict
from evaluate_lm import evaluate, next_token_dataset_config
from src.datasets.collate import collate_fn
from src.datasets.tiny_stories import TinyStoriesDataset
from src.loss import MtpCrossEntropyLoss, TelescopicLoss
from src.model import GPT, TransformerDecoder
from src.model.transformer_utils.linear_attention import causal_linear_attention
from src.model.transformer_utils.transformer_block import CausalSelfAttention
from src.tokenizers import BpeTokenizer


def rotate_pairs(x):
    length, dim = x.shape[-2:]
    positions = torch.arange(length, dtype=x.dtype, device=x.device)
    frequencies = 10000.0 ** (
        -torch.arange(0, dim, 2, dtype=x.dtype, device=x.device) / dim
    )
    angles = positions[:, None] * frequencies[None, :]
    cos, sin = angles.cos(), angles.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


def dense_reference(q, k, v, rope):
    """Independent quadratic formula, used only on tiny test inputs."""
    q_features, k_features = F.elu(q) + 1, F.elu(k) + 1
    q_num = rotate_pairs(q_features) if rope else q_features
    k_num = rotate_pairs(k_features) if rope else k_features
    causal = torch.ones(q.shape[-2], k.shape[-2], device=q.device).tril()
    numerator_weights = torch.einsum("bhtd,bhsd->bhts", q_num, k_num) * causal
    normalizer_weights = (
        torch.einsum("bhtd,bhsd->bhts", q_features, k_features) * causal
    )
    return torch.einsum("bhts,bhsd->bhtd", numerator_weights, v) / (
        normalizer_weights.sum(-1, keepdim=True).clamp_min(1e-6)
    )


class LinearAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_values_and_gradients_match_dense_formula(self):
        for length, chunk in ((1, 64), (7, 3), (65, 64)):
            for rope in (False, True):
                with self.subTest(length=length, chunk=chunk, rope=rope):
                    inputs = [
                        torch.randn(
                            2, 2, length, 4, dtype=torch.float64, requires_grad=True
                        )
                        for _ in range(3)
                    ]
                    rotary = (
                        (lambda q, k: (rotate_pairs(q), rotate_pairs(k)))
                        if rope
                        else None
                    )
                    actual = causal_linear_attention(
                        *inputs, chunk_size=chunk, rotary=rotary
                    )
                    expected = dense_reference(*inputs, rope)
                    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
                    direction = torch.randn_like(actual)
                    actual_grads = torch.autograd.grad(
                        (actual * direction).sum(), inputs
                    )
                    expected_grads = torch.autograd.grad(
                        (expected * direction).sum(), inputs
                    )
                    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(
                            actual_grad, expected_grad, rtol=1e-9, atol=1e-9
                        )

    def test_causal_prefixes_with_rope_and_gates(self):
        for attention_type in ("softmax", "linear"):
            for rope in (False, True):
                for gate in (
                    {},
                    {"use_headwise_gate": True},
                    {"use_elementwise_gate": True},
                ):
                    with self.subTest(attention=attention_type, rope=rope, gate=gate):
                        layer = CausalSelfAttention(
                            16,
                            2,
                            use_rope=rope,
                            max_seq_len=9,
                            attention_type=attention_type,
                            linear_chunk_size=3,
                            **gate,
                        ).eval()
                        seq = torch.randn(2, 9, 16)
                        changed = seq.clone()
                        changed[:, 5:] = torch.randn_like(changed[:, 5:])
                        torch.testing.assert_close(
                            layer(seq)[:, :5],
                            layer(changed)[:, :5],
                            rtol=1e-5,
                            atol=1e-6,
                        )

    def test_checkpoint_layout_and_roundtrip(self):
        base = TransformerDecoder(
            16,
            32,
            2,
            2,
            mlp_hidden_dim=24,
            max_seq_len=8,
            use_rope=True,
            use_headwise_gate=True,
        )
        linear = TransformerDecoder(
            16,
            32,
            2,
            2,
            mlp_hidden_dim=24,
            max_seq_len=8,
            use_rope=True,
            use_headwise_gate=True,
            attention_type="linear",
        )
        linear.load_state_dict(base.state_dict(), strict=True)
        self.assertEqual(list(base.state_dict()), list(linear.state_dict()))
        self.assertEqual(base.transformer_blocks[0].feed_forward.fc1.out_features, 24)
        buffer = io.BytesIO()
        torch.save(linear.state_dict(), buffer)
        buffer.seek(0)
        restored = TransformerDecoder(
            16,
            32,
            2,
            2,
            mlp_hidden_dim=24,
            max_seq_len=8,
            use_rope=True,
            use_headwise_gate=True,
            attention_type="linear",
        )
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        linear.eval()
        restored.eval()
        seq = torch.randint(32, (2, 8))
        torch.testing.assert_close(linear(seq), restored(seq))

    def test_rope_cache_after_inference_supports_training(self):
        for attention_type in ("softmax", "linear"):
            with self.subTest(attention=attention_type):
                layer = CausalSelfAttention(
                    16, 2, use_rope=True, max_seq_len=1, attention_type=attention_type
                )
                with torch.inference_mode():
                    layer(torch.randn(2, 8, 16))
                seq = torch.randn(2, 8, 16, requires_grad=True)
                layer(seq).square().mean().backward()
                self.assertTrue(torch.isfinite(seq.grad).all())

    def test_bfloat16_has_finite_outputs_and_gradients(self):
        inputs = [
            torch.randn(2, 2, 7, 4, dtype=torch.bfloat16, requires_grad=True)
            for _ in range(3)
        ]
        output = causal_linear_attention(*inputs, chunk_size=3)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output).all())
        output.float().square().mean().backward()
        for value in inputs:
            self.assertTrue(torch.isfinite(value.grad).all())

    def test_invalid_linear_parameters_fail_early(self):
        for options in (
            {"linear_chunk_size": 0},
            {"linear_eps": 0},
            {"attention_type": "unknown"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                CausalSelfAttention(16, 2, **options)

    def test_alibi_causality_and_incompatible_linear_combination(self):
        layer = CausalSelfAttention(16, 2, use_alibi=True).eval()
        seq = torch.randn(2, 7, 16)
        changed = seq.clone()
        changed[:, 4:] = torch.randn_like(changed[:, 4:])
        torch.testing.assert_close(layer(seq)[:, :4], layer(changed)[:, :4])
        with self.assertRaisesRegex(ValueError, "ALiBi"):
            CausalSelfAttention(16, 2, use_alibi=True, attention_type="linear")

    def test_linear_task_heads_with_gradient_checkpointing(self):
        seq = torch.randint(1, 32, (2, 7))
        for task in ("mtp", "telescopic"):
            with self.subTest(task=task):
                decoder = TransformerDecoder(
                    16,
                    32,
                    2,
                    2,
                    max_seq_len=8,
                    use_rope=True,
                    attention_type="linear",
                    linear_chunk_size=3,
                    grad_checkpointing=True,
                    n_future_tokens=3 if task == "mtp" else None,
                )
                model = GPT(
                    decoder,
                    telescopic=task == "telescopic",
                    tokenizer=SimpleNamespace(token_to_id={"<eos>": 0}),
                    context_len=8,
                    max_new_tokens=1,
                )
                output = model(seq)
                self.assertEqual(output["logits"].shape, (2, 7, 32))
                if task == "mtp":
                    self.assertEqual(output["mtp_logits"].shape, (2, 3, 7, 32))
                    target = torch.randint(1, 32, (2, 3, 7))
                    loss = MtpCrossEntropyLoss()(**output, target=target)["loss"]
                else:
                    self.assertEqual(output["prefix_logits"].shape, (2, 7, 32))
                    target = torch.randint(1, 32, (2, 7))
                    loss = TelescopicLoss()(**output, target=target)["loss"]
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                model.eval()
                self.assertEqual(model.predict(seq[:1])["seq"].shape, (1, 8))
                if task == "telescopic":
                    self.assertEqual(
                        model(seq, inference=True)["all_logits"].shape, (2, 2, 7, 32)
                    )

    def test_cached_dataset_limits_and_target_layouts(self):
        outputs = Path(__file__).resolve().parent / "outputs"
        outputs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="linear-dataset-test-", dir=outputs
        ) as folder:
            fixture = Path(folder).resolve()
            self.assertEqual(fixture.parent, outputs.resolve())
            texts = [
                f"A little fox played a game and went home. Story {i}."
                for i in range(8)
            ]
            raw = fixture / "raw"
            DatasetDict(
                {
                    "train": Dataset.from_dict({"text": texts}),
                    "validation": Dataset.from_dict({"text": texts[:4]}),
                }
            ).save_to_disk(str(raw))
            tokenizer = BpeTokenizer(vocab_size=300, min_frequency=2)
            tokenizer.train(texts)
            options = dict(
                part="test",
                max_seq_len=4,
                val_ratio=0.25,
                raw_data_path=raw,
                cache_dir=fixture / "tokens",
                allow_download=False,
                tokenizer=tokenizer,
                tokenization_workers=1,
            )
            legacy = TinyStoriesDataset(**options, limit_stories=2)
            canonical = TinyStoriesDataset(**options, max_stories=2)
            torch.testing.assert_close(legacy.tokens, canonical.tokens)
            full = TinyStoriesDataset(**options)
            self.assertGreater(len(full), len(canonical))
            batch = collate_fn([canonical[0], canonical[1]])
            self.assertEqual(batch["target"].shape, (2, 4))
            mtp = TinyStoriesDataset(**options, max_stories=2, n_future_tokens=3)
            batch_mtp = collate_fn([mtp[0], mtp[1]])
            self.assertEqual(batch_mtp["target"].shape, (2, 3, 4))
            torch.testing.assert_close(batch_mtp["target"][:, 0], batch["target"])
            with self.assertRaises(ValueError):
                TinyStoriesDataset(**options, max_stories=3, limit_stories=2)

    def test_quality_normalizes_legacy_limits_and_mtp_configs(self):
        canonical = {"max_seq_len": 8, "max_stories": 2}
        self.assertEqual(
            next_token_dataset_config(canonical),
            next_token_dataset_config({"max_seq_len": 8, "limit_stories": 2}),
        )
        self.assertEqual(
            next_token_dataset_config(canonical),
            next_token_dataset_config({**canonical, "n_future_tokens": 4}),
        )
        with self.assertRaises(ValueError):
            next_token_dataset_config({"max_stories": 3, "limit_stories": 2})

    def test_quality_metrics_weight_tokens_and_ignore_padding(self):
        class FixedModel(torch.nn.Module):
            def forward(self, seq):
                probability = 0.5 if seq[0, 0].item() == 1 else 0.25
                logits = torch.tensor([probability, 1 - probability]).log()
                return {"logits": logits.expand(*seq.shape, 2)}

        batches = [
            {"seq": torch.tensor([[1, 1]]), "target": torch.tensor([[0, 2]])},
            {"seq": torch.tensor([[2, 2, 2]]), "target": torch.tensor([[0, 0, 0]])},
        ]
        result = evaluate(FixedModel(), batches, pad_id=2, device=torch.device("cpu"))
        expected_nll = (math.log(2) + 3 * math.log(4)) / 4
        self.assertAlmostEqual(result["nll"], expected_nll, places=6)
        self.assertAlmostEqual(result["perplexity"], math.exp(expected_nll), places=6)
        self.assertEqual(result["evaluated_tokens"], 4)
        self.assertEqual(result["token_accuracy"], 0.25)
        mtp_batches = [
            {
                "seq": batch["seq"],
                "target": torch.stack(
                    (batch["target"], torch.ones_like(batch["target"])), dim=1
                ),
            }
            for batch in batches
        ]
        self.assertEqual(
            evaluate(FixedModel(), mtp_batches, pad_id=2, device=torch.device("cpu")),
            result,
        )


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
