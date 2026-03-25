# Run: uv run sae-basics/phase0_setup.py
"""
Phase 0: Environment & Dependencies
Load GPT-2 small via TransformerLens and verify IOI behavior exists.
"""

import torch
import transformer_lens
from transformer_lens import HookedTransformer


def check_gpu():
    """Check CUDA availability and determine device."""
    if torch.cuda.is_available():
        try:
            # Test that CUDA actually works (may fail on unsupported arch like sm_120)
            torch.zeros(1, device="cuda")
            print(f"[PASS] GPU available and functional: {torch.cuda.get_device_name(0)}")
            return "cuda"
        except RuntimeError as e:
            print(f"[WARN] GPU detected ({torch.cuda.get_device_name(0)}) but CUDA kernels "
                  f"not compatible with this PyTorch build. Falling back to CPU.")
            print(f"       Error: {e}")
            return "cpu"
    else:
        print("[WARN] No GPU detected, using CPU (will be slow for SAE training)")
        return "cpu"


def load_model(device: str = "cpu") -> HookedTransformer:
    """Load GPT-2 small and verify architecture."""
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()

    # Verify architecture
    cfg = model.cfg
    assert cfg.n_layers == 12, f"Expected 12 layers, got {cfg.n_layers}"
    assert cfg.n_heads == 12, f"Expected 12 heads, got {cfg.n_heads}"
    assert cfg.d_model == 768, f"Expected d_model=768, got {cfg.d_model}"
    assert cfg.d_mlp == 3072, f"Expected d_mlp=3072, got {cfg.d_mlp}"

    print(f"[PASS] GPT-2 small loaded: {cfg.n_layers} layers, {cfg.n_heads} heads, "
          f"d_model={cfg.d_model}, d_mlp={cfg.d_mlp}")
    return model


def test_generation(model: HookedTransformer):
    """Verify the model can generate coherent text."""
    prompt = "When Mary and John went to the store, John gave a drink to"
    output = model.generate(prompt, max_new_tokens=5, temperature=0)
    print(f"[INFO] Generation test:")
    print(f"  Prompt: {prompt!r}")
    print(f"  Output: {output!r}")


def test_ioi_logit_difference(model: HookedTransformer):
    """
    Quick sanity check: on a handful of IOI prompts, the model should assign
    higher logits to the indirect object (IO) than the subject (S).
    """
    prompts = [
        ("When Mary and John went to the store, John gave a drink to",  " Mary",  " John"),
        ("When Alice and Bob went to the park, Bob handed a ball to",   " Alice", " Bob"),
        ("When Sarah and Tom sat in the cafe, Tom passed a cup to",     " Sarah", " Tom"),
        ("When Emily and David walked to school, David lent a book to", " Emily", " David"),
        ("When Lisa and Mark were at the office, Mark sent a file to",  " Lisa",  " Mark"),
    ]

    logit_diffs = []

    for prompt, io_name, s_name in prompts:
        tokens = model.to_tokens(prompt)
        logits = model(tokens)  # (batch, seq, vocab)
        final_logits = logits[0, -1, :]  # logits at final position

        io_token = model.to_single_token(io_name)
        s_token = model.to_single_token(s_name)

        io_logit = final_logits[io_token].item()
        s_logit = final_logits[s_token].item()
        diff = io_logit - s_logit
        logit_diffs.append(diff)

        print(f"  {io_name.strip()} vs {s_name.strip()}: "
              f"IO={io_logit:.2f}, S={s_logit:.2f}, diff={diff:+.2f}")

    mean_diff = sum(logit_diffs) / len(logit_diffs)
    all_positive = all(d > 0 for d in logit_diffs)

    print(f"\n  Mean logit difference: {mean_diff:.2f}")
    print(f"  All positive: {all_positive}")

    if all_positive:
        print("[PASS] Model consistently prefers IO over S")
    else:
        print("[WARN] Some prompts had negative logit difference -- investigate")

    if 1.0 < mean_diff < 6.0:
        print(f"[PASS] Mean logit diff {mean_diff:.2f} is in reasonable range")
    else:
        print(f"[WARN] Mean logit diff {mean_diff:.2f} outside expected range (1-6)")


def main():
    print("=" * 60)
    print("Phase 0: Environment & Setup Verification")
    print("=" * 60)

    print("\n--- GPU Check ---")
    device = check_gpu()

    print("\n--- Loading GPT-2 Small ---")
    model = load_model(device=device)

    print("\n--- Generation Test ---")
    test_generation(model)

    print("\n--- IOI Logit Difference Test ---")
    test_ioi_logit_difference(model)

    print("\n" + "=" * 60)
    print("Phase 0 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
