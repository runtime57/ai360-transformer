"""Analytical matmul FLOPs for the current decoder, not GPU instruction counts."""


def estimate_attention_matmul_flops(
    *,
    d_model,
    num_layers,
    num_heads,
    vocab_size,
    seq_length,
    batch_size,
    attention_type,
    linear_chunk_size=64,
    generation_tokens=0,
    component="model",
):
    dimensions = (
        d_model,
        num_layers,
        num_heads,
        vocab_size,
        seq_length,
        batch_size,
        linear_chunk_size,
    )
    if any(not isinstance(value, int) or value <= 0 for value in dimensions):
        raise ValueError("FLOPs dimensions must be positive integers")
    if d_model % num_heads:
        raise ValueError("d_model must be divisible by num_heads")
    if attention_type not in {"softmax", "linear"} or component not in {
        "model",
        "attention",
    }:
        raise ValueError("Unknown attention type or component")
    if not isinstance(generation_tokens, int) or generation_tokens < 0:
        raise ValueError("generation_tokens must be a nonnegative integer")
    if component != "model" and generation_tokens:
        raise ValueError("Generation FLOPs require the full model")

    batch, length, width = batch_size, seq_length, d_model
    layers = num_layers if component == "model" else 1
    head_dim = width // num_heads
    chunk = min(linear_chunk_size, length)
    chunks = (length + chunk - 1) // chunk
    padded_length = chunks * chunk

    # Two attention matmuls, with one multiply-add counted as two FLOPs.
    causal_softmax_core = layers * 2 * batch * width * length * (length + 1)
    dense_softmax_core = layers * 4 * batch * width * length**2
    # K_chunk.T @ V_chunk and Q_chunk @ prefix: 2 * (2 B H N C d^2).
    # Q_chunk @ K_chunk.T and local_weights @ V_chunk: 2 * (2 B H N C^2 d).
    linear_core = layers * 4 * batch * width * padded_length * (head_dim + chunk)
    core = causal_softmax_core if attention_type == "softmax" else linear_core
    projections = layers * 8 * batch * length * width**2  # QKV + output projection
    ffn = layers * 16 * batch * length * width**2 if component == "model" else 0
    logits = 2 * batch * length * width * vocab_size if component == "model" else 0
    forward = core + projections + ffn + logits
    dense_forward = (
        forward - core + dense_softmax_core if attention_type == "softmax" else forward
    )
    return {
        "effective_layers": layers,
        "head_dim": head_dim,
        "effective_chunk_size": chunk if attention_type == "linear" else None,
        "padded_seq_length": padded_length if attention_type == "linear" else length,
        "attention_core_matmul_flops": core,
        "attention_projections_matmul_flops": projections,
        "attention_module_matmul_flops": core + projections,
        "ffn_matmul_flops": ffn,
        "vocab_projection_matmul_flops": logits,
        "forward_matmul_flops": forward,
        "generation_matmul_flops": forward * generation_tokens
        if generation_tokens
        else None,
        "dense_reference_attention_core_matmul_flops": dense_softmax_core
        if attention_type == "softmax"
        else None,
        "dense_reference_forward_matmul_flops": dense_forward
        if attention_type == "softmax"
        else None,
    }
