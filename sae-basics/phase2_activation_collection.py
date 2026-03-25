# Run: uv run sae-basics/phase2_activation_collection.py
"""
Phase 2: Activation Collection
Collect residual-stream activations from GPT-2 small on general text for SAE training.

Hook point: blocks.7.hook_resid_post  (residual stream after layer 7)
Dataset:    wikitext-103-raw-v1        (streamed from HuggingFace)
Output:     sae-basics/activations/   (memmap + normalization stats)
"""

import os
import pathlib

import numpy as np
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOOK_LAYER = 7
HOOK_POINT = f"blocks.{HOOK_LAYER}.hook_resid_post"
D_MODEL = 768
TARGET_TOKENS = 1_000_000  # ~1M activation vectors
BATCH_SEQ_LEN = 128        # tokens per text chunk fed to model
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
OUTPUT_DIR = pathlib.Path(__file__).parent / "activations"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> str:
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return "cuda"
        except RuntimeError:
            return "cpu"
    return "cpu"


def tokenize_texts(model: HookedTransformer, texts: list[str], seq_len: int) -> torch.Tensor:
    """Tokenize a batch of texts and split into fixed-length chunks."""
    all_tokens = []
    for text in texts:
        toks = model.to_tokens(text, prepend_bos=True).squeeze(0)  # (seq,)
        all_tokens.append(toks)

    # Concatenate and split into fixed-length chunks
    flat = torch.cat(all_tokens)  # (total_tokens,)
    n_chunks = len(flat) // seq_len
    if n_chunks == 0:
        return torch.empty(0, seq_len, dtype=torch.long)
    trimmed = flat[: n_chunks * seq_len]
    return trimmed.reshape(n_chunks, seq_len)


def collect_activations(
    model: HookedTransformer,
    device: str,
    target_tokens: int = TARGET_TOKENS,
) -> np.ndarray:
    """
    Stream text from the dataset, run through GPT-2 small, and collect
    activations at HOOK_POINT.  Returns array of shape (N, D_MODEL).
    """
    print(f"[INFO] Loading dataset: {DATASET_NAME}/{DATASET_CONFIG} (streaming)")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=True)

    collected = []
    total_tokens = 0
    text_buffer = []
    BUFFER_SIZE = 64  # texts to accumulate before tokenizing

    print(f"[INFO] Collecting ~{target_tokens:,} activation vectors at {HOOK_POINT} ...")

    for i, example in enumerate(ds):
        text = example["text"].strip()
        if len(text) < 50:  # skip short/empty entries
            continue
        text_buffer.append(text)

        if len(text_buffer) < BUFFER_SIZE:
            continue

        # Tokenize buffered texts into fixed-length chunks
        chunks = tokenize_texts(model, text_buffer, BATCH_SEQ_LEN)
        text_buffer = []

        if chunks.shape[0] == 0:
            continue

        # Process in mini-batches to avoid OOM
        mini_batch_size = 32
        for start in range(0, chunks.shape[0], mini_batch_size):
            batch = chunks[start : start + mini_batch_size].to(device)

            with torch.no_grad():
                _, cache = model.run_with_cache(batch, names_filter=HOOK_POINT)

            acts = cache[HOOK_POINT]  # (batch, seq, d_model)
            acts_flat = acts.reshape(-1, D_MODEL).cpu().numpy()  # (batch*seq, d_model)
            collected.append(acts_flat)
            total_tokens += acts_flat.shape[0]

            if total_tokens % 100_000 < acts_flat.shape[0]:
                print(f"  ... collected {total_tokens:,} / {target_tokens:,} vectors")

            if total_tokens >= target_tokens:
                break

        if total_tokens >= target_tokens:
            break

    # Handle any remaining texts in the buffer
    if text_buffer and total_tokens < target_tokens:
        chunks = tokenize_texts(model, text_buffer, BATCH_SEQ_LEN)
        if chunks.shape[0] > 0:
            mini_batch_size = 32
            for start in range(0, chunks.shape[0], mini_batch_size):
                batch = chunks[start : start + mini_batch_size].to(device)
                with torch.no_grad():
                    _, cache = model.run_with_cache(batch, names_filter=HOOK_POINT)
                acts = cache[HOOK_POINT].reshape(-1, D_MODEL).cpu().numpy()
                collected.append(acts)
                total_tokens += acts.shape[0]
                if total_tokens >= target_tokens:
                    break

    all_acts = np.concatenate(collected, axis=0)[:target_tokens]
    print(f"[INFO] Collected {all_acts.shape[0]:,} activation vectors, shape {all_acts.shape}")
    return all_acts


def save_activations(activations: np.ndarray, output_dir: pathlib.Path):
    """Save activations as memmap and compute/store normalization stats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as memmap
    memmap_path = output_dir / "activations.npy"
    np.save(str(memmap_path), activations)
    print(f"[INFO] Saved activations to {memmap_path}")

    # Compute and save normalization stats
    mean = activations.mean(axis=0)  # (d_model,)
    std = activations.std(axis=0)    # (d_model,)

    stats = {"mean": mean, "std": std}
    stats_path = output_dir / "normalization_stats.npz"
    np.savez(str(stats_path), **stats)
    print(f"[INFO] Saved normalization stats to {stats_path}")
    print(f"  mean norm: {np.linalg.norm(mean):.4f}")
    print(f"  mean of std: {std.mean():.4f}")

    return mean, std


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def sanity_check_shape(activations: np.ndarray):
    """Check activation shape is (N, 768)."""
    assert activations.ndim == 2, f"Expected 2D, got {activations.ndim}D"
    assert activations.shape[1] == D_MODEL, (
        f"Expected d_model={D_MODEL}, got {activations.shape[1]}"
    )
    print(f"[PASS] Activation shape: {activations.shape}")


def sanity_check_magnitude(activations: np.ndarray):
    """Check activations have reasonable magnitude."""
    norms = np.linalg.norm(activations, axis=1)
    mean_norm = norms.mean()
    print(f"[INFO] Activation norms: mean={mean_norm:.2f}, min={norms.min():.2f}, max={norms.max():.2f}")

    if mean_norm < 0.01:
        print("[FAIL] Activations are near-zero -- hook may not be firing")
    elif mean_norm > 1000:
        print("[FAIL] Activations are exploding -- check hook point or model")
    else:
        print(f"[PASS] Mean activation norm {mean_norm:.2f} is in reasonable range")

    # Check for NaN/Inf
    assert not np.any(np.isnan(activations)), "Found NaN in activations!"
    assert not np.any(np.isinf(activations)), "Found Inf in activations!"
    print("[PASS] No NaN or Inf values")


def sanity_check_reconstruction(model: HookedTransformer, device: str):
    """
    Verify that cached activations at HOOK_POINT can reconstruct the
    model's output when fed through the remaining layers.
    """
    prompt = "When Mary and John went to the store, John gave a drink to"
    tokens = model.to_tokens(prompt).to(device)

    with torch.no_grad():
        # Full forward pass
        full_logits = model(tokens)

        # Cached forward pass
        _, cache = model.run_with_cache(tokens, names_filter=HOOK_POINT)
        cached_resid = cache[HOOK_POINT]  # (1, seq, d_model)

        # Run remaining layers from the cached residual stream
        resid = cached_resid
        for layer_idx in range(HOOK_LAYER + 1, model.cfg.n_layers):
            resid = model.blocks[layer_idx](resid)
        resid = model.ln_final(resid)
        reconstructed_logits = model.unembed(resid)

    # Compare
    diff = (full_logits - reconstructed_logits).abs().max().item()
    print(f"[INFO] Max logit difference (full vs reconstructed): {diff:.6f}")

    if diff < 0.01:
        print("[PASS] Reconstruction matches full forward pass")
    elif diff < 0.1:
        print("[WARN] Small reconstruction error -- likely numerical precision")
    else:
        print("[FAIL] Large reconstruction error -- something is wrong with the hook point or layer indexing")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 2: Activation Collection")
    print("=" * 60)

    device = get_device()
    print(f"\n[INFO] Using device: {device}")

    print("\n--- Loading GPT-2 Small ---")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()

    print("\n--- Reconstruction Sanity Check ---")
    sanity_check_reconstruction(model, device)

    print("\n--- Collecting Activations ---")
    activations = collect_activations(model, device)

    print("\n--- Sanity Checks ---")
    sanity_check_shape(activations)
    sanity_check_magnitude(activations)

    print("\n--- Saving ---")
    save_activations(activations, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Phase 2 complete!")
    print(f"Activations saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
