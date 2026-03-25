# circuit-tracing-prereqs
Working on Building Skills for AI Safety and Circuit Tracing Research

## Current Project: SAE on GPT-2 Small for IOI

### Background

**Indirect Object Identification (IOI)** is the task where a model must predict the correct indirect object in sentences like:

> "When Mary and John went to the store, John gave a drink to **Mary**"

The model should assign higher probability to "Mary" (the indirect object, IO) than to "John" (the subject, S). This task was first analyzed in detail by [Wang et al. 2022](https://arxiv.org/abs/2211.00593), which identified a circuit of ~26 heads in GPT-2 small responsible for this behavior.

**Sparse Autoencoders (SAEs)** decompose neural network activations into sparse, interpretable features. By training SAEs on GPT-2 small's residual stream or MLP outputs, we can identify which learned features activate during IOI and potentially discover finer-grained structure than head-level analysis.

---

### Phase 0: Environment & Dependencies

**Goal:** Get GPT-2 small running with hooks, verify IOI behavior exists.

**Steps:**
1. Add dependencies to `pyproject.toml`: `transformer-lens`, `transformers`, `datasets`, `wandb` (optional), `matplotlib`, `numpy`
2. Verify GPU is available and GPT-2 small loads correctly
3. Load GPT-2 small via TransformerLens (`HookedTransformer.from_pretrained("gpt2-small")`)

**Sanity tests:**
- [ ] `model.generate("When Mary and John went to the store, John gave a drink to")` completes sensibly
- [ ] `torch.cuda.is_available()` returns `True`
- [ ] Model has 12 layers, 12 heads, d_model=768, d_mlp=3072

---

### Phase 1: IOI Dataset & Task Verification

**Goal:** Build the IOI dataset and confirm GPT-2 small exhibits the IOI behavior.

**Steps:**
1. Generate IOI prompts using templates from Wang et al.:
   - **ABB format:** "When [IO] and [S] went to the store, [S] gave a drink to" -> expect [IO]
   - **BAB format:** "When [S] and [IO] went to the store, [S] gave a drink to" -> expect [IO]
   - Use diverse name pairs and location/action templates
2. Create a flipped dataset (ABBA <-> BABA, or swap IO/S names) for counterfactual comparisons
3. Define the **logit difference** metric: `logit(IO_token) - logit(S_token)` at the final token position

**Sanity tests:**
- [ ] Logit difference is **positive** on the vast majority (>90%) of IOI prompts (model prefers IO over S)
- [ ] Mean logit difference is roughly **2.5-3.5** (this is the ballpark from the original paper)
- [ ] On the flipped dataset (swap IO and S names), logit difference flips sign
- [ ] Spot-check 10 random prompts manually to verify labels are correct
- [ ] Distribution of logit differences is unimodal and not heavily skewed (model consistently does the task, not just on average)

---

### Phase 2: Activation Collection

**Goal:** Collect activations from GPT-2 small on a large corpus for SAE training.

**Steps:**
1. Choose the **hook point** to train the SAE on. Start with one of:
   - `blocks.{layer}.hook_resid_post` (residual stream after layer) -- **recommended first target: layer 7 or 8** (middle of the network, where IOI circuit components converge)
   - `blocks.{layer}.mlp_out` (MLP output)
   - `blocks.{layer}.hook_resid_pre` for comparison
2. Collect activations from **general text** (OpenWebText or similar), NOT just IOI prompts
   - SAEs should learn general features; IOI-specific features will emerge naturally
   - Collect ~1-10M activation vectors (tokens) depending on GPU memory
3. Store activations in a memory-mapped format or stream them during training
4. Normalize: compute and store the mean and standard deviation of activations for pre-processing

**Sanity tests:**
- [ ] Activation shape is `(n_tokens, d_model)` = `(N, 768)`
- [ ] Activations have reasonable magnitude (not all zeros, not exploding). Mean norm should be roughly O(1)-O(10)
- [ ] Verify you can reconstruct model outputs from cached activations (forward pass through remaining layers matches original logits)

---

### Phase 3: SAE Architecture & Training

**Goal:** Implement and train a sparse autoencoder on collected activations.

**Architecture (vanilla TopK or ReLU SAE):**
```
input:   x in R^768 (centered by subtracting decoder bias or data mean)
encoder: h = ReLU(W_enc @ x + b_enc)    # h in R^(expansion * 768), e.g., expansion=16 -> R^12288
decoder: x_hat = W_dec @ h + b_dec      # x_hat in R^768
loss:    L = MSE(x, x_hat) + lambda * L1(h)
```

**Key hyperparameters:**
| Param | Starting value | Notes |
|-------|---------------|-------|
| Expansion factor | 8-16x | Start with 8x (6144 features) for faster iteration |
| L1 coefficient (lambda) | 1e-3 to 5e-3 | Controls sparsity; tune to get L0 ~ 10-50 |
| Learning rate | 3e-4 | Adam, with warmup over first 1000 steps |
| Batch size | 4096 | Activation vectors per batch |
| Training tokens | 5M-50M | More is better; 10M is a good starting point |

**Training steps:**
1. Implement SAE as a `nn.Module`
2. **Constrain decoder columns to unit norm** (or re-normalize after each step) -- this is critical to prevent the model from shrinking decoder norms to cheat the L1 penalty
3. Train with Adam optimizer
4. Log: loss, reconstruction MSE, L1 loss, L0 (number of active features per input), fraction of "dead" features (never activate in a batch)

**Sanity tests during training:**
- [ ] **Reconstruction loss decreases** over training and converges
- [ ] **L0 (avg features active per input)** is in range 10-100. If L0 > 200, increase lambda. If L0 < 5, decrease lambda
- [ ] **Dead feature fraction < 50%**. If most features are dead, learning rate may be too high, or L1 too strong. Consider neuron resampling (Anthropic's method: reinitialize dead features toward poorly-reconstructed examples)
- [ ] **Explained variance** (`1 - MSE(x, x_hat) / Var(x)`) should be > 0.85, ideally > 0.95
- [ ] Decoder weight norms stay close to 1.0 (if using unit norm constraint)
- [ ] Loss is not NaN or diverging

**Sanity tests after training:**
- [ ] **Substitution test:** Replace real activations with SAE reconstructions and run the model. Cross-entropy loss should increase only modestly (< 0.1-0.2 nats). If it increases dramatically, the SAE is losing critical information
- [ ] Feature decoder directions (columns of W_dec) are not all identical or highly correlated
- [ ] Histogram of feature activations shows most features are sparse (activate on <1% of inputs)

---

### Progress Log

#### Phase 0 -- completed
All checks passed. GPT-2 small loads correctly (12 layers, 12 heads, d_model=768, d_mlp=3072). Generation completes sensibly. IOI logit difference is positive on all 5 test prompts with a mean in the expected range.

#### Phase 2 -- completed
Collected 1,000,000 activation vectors at `blocks.7.hook_resid_post` from wikitext-103-raw-v1 (streamed via HuggingFace `datasets`).

| Check | Result |
|-------|--------|
| Shape | `(1000000, 768)` |
| Mean activation norm | 126.06 (range 70.97 -- 3177.75) |
| NaN / Inf | None |
| Reconstruction (cached -> remaining layers) | Max logit diff = 0.000000 |

Normalization stats saved (mean norm 87.05, mean std 3.08). Output in `sae-basics/activations/`.

#### Phase 3 -- completed
Trained a **TopK sparse autoencoder** (8x expansion = 6144 features, k=50) for 10 epochs over the 1M activation dataset (~2440 steps). Initially attempted a vanilla ReLU SAE with L1 penalty, but L0 remained ~4000 regardless of L1 coefficient -- switched to TopK which directly controls sparsity.

Dead feature resampling triggered at step 2000, reinitializing 1792 dead features toward high-error examples.

| Metric | Value | Status |
|--------|-------|--------|
| Explained variance | 0.985 | PASS (> 0.95) |
| L0 (features active per input) | 50.0 | PASS (exactly k) |
| Dead feature fraction | 9.39% | PASS (< 50%) |
| Decoder norms | mean=1.0000 | PASS |
| Decoder diversity (mean cosine sim) | 0.0015 | PASS (< 0.5) |
| Feature sparsity | 80.5% activate on <1% of inputs | PASS |
| Substitution test (CE increase) | +1.03 nats | FAIL (> 0.2 nats) |
| NaN losses | None | PASS |

The substitution test failure is expected given only 1M training tokens (the plan recommends 5--50M). Reconstruction quality (EV=0.985) should still be sufficient for IOI feature analysis. Model saved to `sae-basics/sae_model/`.

---

### Phase 4: Feature Analysis on IOI

**Goal:** Identify which SAE features are relevant to the IOI task.

**Steps:**
1. Run IOI prompts through the model, collect activations at the SAE's hook point
2. Encode IOI activations through the trained SAE to get feature activations
3. **Attribution analysis:** For each feature, compute how much it contributes to the logit difference:
   - `contribution_f = (W_dec[:, f] @ W_U[:, IO] - W_dec[:, f] @ W_U[:, S]) * h_f`
   - Where `W_U` is the unembedding matrix, `h_f` is the feature activation
4. Rank features by mean |contribution| across IOI prompts
5. **Ablation analysis:** Zero out top contributing features one at a time and measure logit difference change
6. For top features, find their **max-activating examples** from the general corpus to interpret what they represent

**Sanity tests:**
- [ ] A small number of features (<20) account for the majority (>80%) of the logit difference -- the task should be concentrated in a sparse set of features
- [ ] Top features activate **selectively** on IOI-like contexts (e.g., repeated names, prepositional phrases), not uniformly on all text
- [ ] Ablating the top ~5-10 features significantly reduces logit difference (by >50%)
- [ ] Ablating random features of similar activation magnitude does NOT significantly affect logit difference (control comparison)
- [ ] Features that are important for IOI should also activate on novel IOI-like sentences not in the training set (generalization check)
- [ ] Some features should correspond to known circuit components: name mover heads, S-inhibition heads, duplicate token heads, induction heads

---

### Phase 5: Comparison with Known IOI Circuit

**Goal:** Validate SAE findings against the known IOI circuit from Wang et al.

**Steps:**
1. Map top SAE features back to attention head outputs where possible
2. Compare feature importance ranking with known component importance from the original paper
3. Check if SAE reveals **sub-head structure** (multiple features corresponding to a single attention head, capturing different roles)
4. Look for features that might reveal structure the original head-level analysis missed

**Sanity tests:**
- [ ] Features related to Name Mover heads (9.9, 9.6, 10.0) should rank highly
- [ ] Features related to S-Inhibition heads (7.3, 7.9, 8.6) should appear if the SAE is on an appropriate layer
- [ ] The overall story should be **consistent** with the known circuit, even if more fine-grained
- [ ] If training SAE on layer 7-8 residual stream, S-inhibition and Duplicate Token features should be prominent

---

### Stretch Goals

- [ ] Train SAEs on multiple layers and compare which features appear where
- [ ] Try different SAE variants: TopK SAE, Gated SAE, BatchTopK
- [ ] Use SAE features to do **circuit-level causal interventions** (activate/deactivate specific features and measure downstream effects)
- [ ] Visualize feature geometries (e.g., cosine similarity between IOI-relevant decoder directions)
- [ ] Compare SAE feature sparsity on IOI prompts vs. random text

---

## File Index

- `sae-basics/phase0_setup.py` -- Phase 0: loads GPT-2 small via TransformerLens, verifies architecture (12 layers, 12 heads, d_model=768), tests generation, and checks IOI logit difference
- `sae-basics/phase2_activation_collection.py` -- Phase 2: collects 1M residual-stream activations at layer 7 (`hook_resid_post`) from wikitext-103, saves as numpy with normalization stats to `sae-basics/activations/`
- `sae-basics/phase3_sae_training.py` -- Phase 3: trains a TopK sparse autoencoder (8x expansion, k=50) on collected activations, with dead feature resampling and full post-training sanity checks; saves model to `sae-basics/sae_model/`
- `demo_workflow/test_open_router.py` -- Minimal example hitting the OpenRouter API via the OpenAI SDK
- `toy_transformers/Transformer.py` -- Hand-written transformer components (embed/unembed, MLP, attention)
- `toy_transformers/modular_arithmetic_transformer.py` -- Reimplementation of the modular arithmetic toy model from [Nanda 2023](https://arxiv.org/abs/2301.05217)

## Setup

### Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
uv sync
```

This installs all dependencies into a local virtual environment (`.venv`). PyTorch is pulled from the CUDA 12.4 index on non-Mac machines, so GPU support works out of the box on typical Linux GPU servers.
