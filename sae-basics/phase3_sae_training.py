# Run: uv run sae-basics/phase3_sae_training.py
"""
Phase 3: SAE Architecture & Training
Train a sparse autoencoder on the layer-7 residual stream activations collected in Phase 2.

Architecture:  TopK SAE -- keeps only the top-k encoder activations per input,
               giving direct control over L0 sparsity without needing to tune L1.
Input:         sae-basics/activations/activations.npy  (1M x 768)
Output:        sae-basics/sae_model/                   (trained weights + training log)
"""

import json
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens import HookedTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
D_MODEL = 768
EXPANSION_FACTOR = 8              # 8x -> 6144 features
D_SAE = D_MODEL * EXPANSION_FACTOR  # 6144

TOP_K = 50                        # number of active features per input (directly controls L0)
LR = 3e-4                         # learning rate
WARMUP_STEPS = 1000
BATCH_SIZE = 4096
N_EPOCHS = 10                     # passes over the 1M dataset (~244 steps/epoch)

# Dead feature resampling
RESAMPLE_EVERY = 2000             # steps between resampling checks
DEAD_THRESHOLD = 200              # feature is "dead" if not active for this many steps

ACTIVATIONS_DIR = pathlib.Path(__file__).parent / "activations"
OUTPUT_DIR = pathlib.Path(__file__).parent / "sae_model"

HOOK_LAYER = 7
HOOK_POINT = f"blocks.{HOOK_LAYER}.hook_resid_post"


# ---------------------------------------------------------------------------
# SAE Module
# ---------------------------------------------------------------------------

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, d_sae: int, top_k: int):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.top_k = top_k

        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # Kaiming init
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)

        # Initialize decoder rows to unit norm
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse feature activations using TopK."""
        x_centered = x - self.b_dec
        pre_acts = x_centered @ self.W_enc + self.b_enc  # (batch, d_sae)

        # TopK: keep only top-k activations, zero the rest
        topk_vals, topk_idx = torch.topk(pre_acts, self.top_k, dim=1)
        h = torch.zeros_like(pre_acts)
        h.scatter_(1, topk_idx, F.relu(topk_vals))
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to input space."""
        return h @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, feature_activations)."""
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h

    @torch.no_grad()
    def normalize_decoder(self):
        """Constrain decoder rows to unit norm."""
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def get_device() -> str:
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return "cuda"
        except RuntimeError:
            return "cpu"
    return "cpu"


def load_activations() -> np.ndarray:
    path = ACTIVATIONS_DIR / "activations.npy"
    print(f"[INFO] Loading activations from {path}")
    acts = np.load(str(path))
    print(f"[INFO] Loaded {acts.shape[0]:,} vectors, shape {acts.shape}")
    return acts


def get_lr(step: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr


def train_sae(activations: np.ndarray, device: str) -> tuple[SparseAutoencoder, list[dict]]:
    n_samples = activations.shape[0]
    steps_per_epoch = n_samples // BATCH_SIZE
    total_steps = N_EPOCHS * steps_per_epoch

    print(f"[INFO] Training config:")
    print(f"  d_model={D_MODEL}, d_sae={D_SAE} ({EXPANSION_FACTOR}x expansion)")
    print(f"  TopK={TOP_K}, LR={LR}, batch_size={BATCH_SIZE}")
    print(f"  {N_EPOCHS} epochs x {steps_per_epoch} steps = {total_steps} total steps")

    # Compute data variance for explained variance metric
    data_var = np.var(activations, axis=0).sum()
    print(f"  Data variance (sum): {data_var:.2f}")

    sae = SparseAutoencoder(D_MODEL, D_SAE, TOP_K).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=LR)

    # Track when each feature last fired (for dead feature detection)
    last_fired = torch.zeros(D_SAE, dtype=torch.long, device=device)

    log = []
    global_step = 0
    t0 = time.time()

    for epoch in range(N_EPOCHS):
        perm = np.random.permutation(n_samples)

        epoch_mse = 0.0
        epoch_l0 = 0.0

        for batch_idx in range(steps_per_epoch):
            idx = perm[batch_idx * BATCH_SIZE : (batch_idx + 1) * BATCH_SIZE]
            x = torch.tensor(activations[idx], dtype=torch.float32, device=device)

            # LR warmup
            lr = get_lr(global_step, WARMUP_STEPS, LR)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward
            x_hat, h = sae(x)

            # Loss: pure reconstruction (TopK handles sparsity)
            loss = F.mse_loss(x_hat, x)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Normalize decoder after each step
            sae.normalize_decoder()

            # Track metrics
            with torch.no_grad():
                l0 = (h > 0).float().sum(dim=1).mean().item()
                active_mask = (h > 0).any(dim=0)
                last_fired[active_mask] = global_step
                dead_frac = ((global_step - last_fired) > DEAD_THRESHOLD).float().mean().item() if global_step > DEAD_THRESHOLD else 0.0

            epoch_mse += loss.item()
            epoch_l0 += l0

            # Dead feature resampling
            if global_step > 0 and global_step % RESAMPLE_EVERY == 0:
                n_dead = resample_dead_features(sae, activations, device, last_fired, global_step)
                if n_dead > 0:
                    print(f"  [RESAMPLE] step {global_step}: reinitialized {n_dead} dead features")

            # Log every 200 steps
            if global_step % 200 == 0:
                explained_var = 1.0 - (loss.item() * D_MODEL / data_var)
                entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "mse": float(loss.item()),
                    "l0": float(l0),
                    "dead_frac": float(dead_frac),
                    "explained_var": float(explained_var),
                    "lr": float(lr),
                }
                log.append(entry)

                elapsed = time.time() - t0
                print(
                    f"  step {global_step:>6d}/{total_steps} | "
                    f"mse={loss.item():.4f} | "
                    f"L0={l0:.1f} dead={dead_frac:.2%} EV={explained_var:.3f} | "
                    f"{elapsed:.0f}s"
                )

                if torch.isnan(loss):
                    print("[FAIL] Loss is NaN -- stopping training")
                    return sae, log

            global_step += 1

        avg_mse = epoch_mse / steps_per_epoch
        avg_l0 = epoch_l0 / steps_per_epoch
        print(f"\n  === Epoch {epoch+1}/{N_EPOCHS} done === "
              f"avg_mse={avg_mse:.4f} avg_L0={avg_l0:.1f}\n")

    total_time = time.time() - t0
    print(f"[INFO] Training finished in {total_time:.0f}s ({total_time/60:.1f}min)")
    return sae, log


def resample_dead_features(
    sae: SparseAutoencoder,
    activations: np.ndarray,
    device: str,
    last_fired: torch.Tensor,
    current_step: int,
) -> int:
    """Reinitialize dead features toward poorly-reconstructed examples."""
    dead_mask = (current_step - last_fired) > DEAD_THRESHOLD
    n_dead = dead_mask.sum().item()
    if n_dead == 0:
        return 0

    idx = np.random.choice(len(activations), size=min(BATCH_SIZE * 4, len(activations)), replace=False)
    x = torch.tensor(activations[idx], dtype=torch.float32, device=device)

    with torch.no_grad():
        x_hat, _ = sae(x)
        errors = (x - x_hat).pow(2).sum(dim=1)
        probs = errors / errors.sum()
        resample_idx = torch.multinomial(probs, num_samples=min(n_dead, len(x)), replacement=False)
        resampled_vecs = x[resample_idx] - x_hat[resample_idx]

    dead_indices = dead_mask.nonzero().squeeze(-1)[:len(resample_idx)]

    with torch.no_grad():
        for i, feat_idx in enumerate(dead_indices):
            if i >= len(resampled_vecs):
                break
            direction = F.normalize(resampled_vecs[i], dim=0)
            sae.W_enc.data[:, feat_idx] = direction * 0.1
            sae.W_dec.data[feat_idx] = direction
            sae.b_enc.data[feat_idx] = 0.0
        last_fired[dead_indices] = current_step

    sae.normalize_decoder()
    return len(dead_indices)


# ---------------------------------------------------------------------------
# Post-training sanity checks
# ---------------------------------------------------------------------------

def sanity_check_training_metrics(log: list[dict]):
    """Check that training metrics are in expected ranges."""
    print("\n--- Training Metric Checks ---")

    early_mse = np.mean([e["mse"] for e in log[:3]])
    late_mse = np.mean([e["mse"] for e in log[-3:]])
    if late_mse < early_mse:
        print(f"[PASS] MSE decreased: {early_mse:.4f} -> {late_mse:.4f}")
    else:
        print(f"[WARN] MSE did not decrease: {early_mse:.4f} -> {late_mse:.4f}")

    final_l0 = log[-1]["l0"]
    if 10 <= final_l0 <= 100:
        print(f"[PASS] Final L0 = {final_l0:.1f} (in range 10-100)")
    elif 5 <= final_l0 <= 200:
        print(f"[WARN] Final L0 = {final_l0:.1f} (outside ideal 10-100, but acceptable)")
    else:
        print(f"[FAIL] Final L0 = {final_l0:.1f} (outside range 5-200)")

    final_dead = log[-1]["dead_frac"]
    if final_dead < 0.5:
        print(f"[PASS] Dead feature fraction = {final_dead:.2%} (< 50%)")
    else:
        print(f"[FAIL] Dead feature fraction = {final_dead:.2%} (>= 50%)")

    final_ev = log[-1]["explained_var"]
    if final_ev > 0.95:
        print(f"[PASS] Explained variance = {final_ev:.3f} (> 0.95, excellent)")
    elif final_ev > 0.85:
        print(f"[PASS] Explained variance = {final_ev:.3f} (> 0.85)")
    else:
        print(f"[WARN] Explained variance = {final_ev:.3f} (< 0.85)")

    any_nan = any(np.isnan(e["loss"]) for e in log)
    if not any_nan:
        print("[PASS] No NaN losses during training")
    else:
        print("[FAIL] NaN losses detected during training")


def sanity_check_decoder_norms(sae: SparseAutoencoder):
    """Check decoder row norms are close to 1.0."""
    with torch.no_grad():
        norms = sae.W_dec.data.norm(dim=1)
        mean_norm = norms.mean().item()
        std_norm = norms.std().item()
    print(f"[INFO] Decoder norms: mean={mean_norm:.4f}, std={std_norm:.4f}")
    if abs(mean_norm - 1.0) < 0.05:
        print("[PASS] Decoder norms close to 1.0")
    else:
        print(f"[WARN] Decoder norms deviate from 1.0: mean={mean_norm:.4f}")


def sanity_check_decoder_diversity(sae: SparseAutoencoder):
    """Check that decoder directions are not all identical."""
    with torch.no_grad():
        W = sae.W_dec.data
        n = min(200, W.shape[0])
        idx = torch.randperm(W.shape[0])[:n]
        subset = F.normalize(W[idx], dim=1)
        cos_sim = subset @ subset.T
        mask = ~torch.eye(n, dtype=torch.bool, device=cos_sim.device)
        off_diag = cos_sim[mask]
        mean_cos = off_diag.mean().item()
        max_cos = off_diag.max().item()

    print(f"[INFO] Decoder cosine similarity (200 random pairs): mean={mean_cos:.4f}, max={max_cos:.4f}")
    if mean_cos < 0.5:
        print("[PASS] Decoder directions are diverse (mean cosine < 0.5)")
    else:
        print("[WARN] Decoder directions may be too similar")


def sanity_check_sparsity(sae: SparseAutoencoder, activations: np.ndarray, device: str):
    """Check that most features are sparse (activate on <1% of inputs)."""
    n_check = min(50_000, len(activations))
    x = torch.tensor(activations[:n_check], dtype=torch.float32, device=device)

    with torch.no_grad():
        chunk_size = 4096
        activation_counts = torch.zeros(D_SAE, device=device)
        for start in range(0, n_check, chunk_size):
            batch = x[start : start + chunk_size]
            h = sae.encode(batch)
            activation_counts += (h > 0).float().sum(dim=0)

    activation_rates = activation_counts / n_check
    sparse_frac = (activation_rates < 0.01).float().mean().item()
    median_rate = activation_rates.median().item()

    print(f"[INFO] Feature activation rates: median={median_rate:.4f}, "
          f"{sparse_frac:.1%} of features activate on <1% of inputs")
    if sparse_frac > 0.5:
        print("[PASS] Majority of features are sparse")
    else:
        print("[WARN] Features may not be sparse enough")


def sanity_check_substitution(sae: SparseAutoencoder, device: str):
    """
    Substitution test: replace real activations with SAE reconstructions
    and check that cross-entropy loss increases only modestly.
    """
    print("\n--- Substitution Test ---")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()

    prompts = [
        "The quick brown fox jumps over the lazy dog",
        "In a distant galaxy far far away there lived",
        "The president of the United States announced that",
        "Machine learning has revolutionized the field of",
        "When Mary and John went to the store John gave a drink to",
    ]

    ce_original_list = []
    ce_reconstructed_list = []

    for prompt in prompts:
        tokens = model.to_tokens(prompt).to(device)
        with torch.no_grad():
            original_logits = model(tokens)
            original_loss = F.cross_entropy(
                original_logits[0, :-1], tokens[0, 1:],
            ).item()
            ce_original_list.append(original_loss)

            _, cache = model.run_with_cache(tokens, names_filter=HOOK_POINT)
            real_acts = cache[HOOK_POINT]

            flat = real_acts.reshape(-1, D_MODEL)
            recon_flat, _ = sae(flat)
            recon_acts = recon_flat.reshape(real_acts.shape)

            resid = recon_acts
            for layer_idx in range(HOOK_LAYER + 1, model.cfg.n_layers):
                resid = model.blocks[layer_idx](resid)
            resid = model.ln_final(resid)
            recon_logits = model.unembed(resid)

            recon_loss = F.cross_entropy(
                recon_logits[0, :-1], tokens[0, 1:],
            ).item()
            ce_reconstructed_list.append(recon_loss)

    mean_orig = np.mean(ce_original_list)
    mean_recon = np.mean(ce_reconstructed_list)
    increase = mean_recon - mean_orig

    print(f"[INFO] Mean CE loss: original={mean_orig:.4f}, reconstructed={mean_recon:.4f}, "
          f"increase={increase:.4f} nats")

    if increase < 0.1:
        print("[PASS] CE increase < 0.1 nats (excellent)")
    elif increase < 0.2:
        print("[PASS] CE increase < 0.2 nats (acceptable)")
    elif increase < 0.5:
        print("[WARN] CE increase < 0.5 nats (marginal)")
    else:
        print("[FAIL] CE increase >= 0.5 nats (SAE is losing critical information)")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_model(sae: SparseAutoencoder, log: list[dict], output_dir: pathlib.Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / "sae_weights.pt"
    torch.save(sae.state_dict(), str(weights_path))
    print(f"[INFO] Saved SAE weights to {weights_path}")

    config = {
        "d_model": D_MODEL,
        "d_sae": D_SAE,
        "expansion_factor": EXPANSION_FACTOR,
        "top_k": TOP_K,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS,
        "hook_point": HOOK_POINT,
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[INFO] Saved config to {config_path}")

    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[INFO] Saved training log to {log_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Phase 3: SAE Architecture & Training")
    print("=" * 60)

    device = get_device()
    print(f"\n[INFO] Using device: {device}")

    activations = load_activations()

    print("\n--- Training ---")
    sae, log = train_sae(activations, device)

    print("\n--- Post-Training Sanity Checks ---")
    sanity_check_training_metrics(log)
    sanity_check_decoder_norms(sae)
    sanity_check_decoder_diversity(sae)
    sanity_check_sparsity(sae, activations, device)
    sanity_check_substitution(sae, device)

    print("\n--- Saving ---")
    save_model(sae, log, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Phase 3 complete!")
    print(f"SAE model saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
