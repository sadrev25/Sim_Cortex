# Sim_Cortex — Plan → Score → Execute

**Failure-aware trajectory selection for robot manipulation using world model latent space.**

No real robot data. No expensive fine-tuning. Failure signatures learned entirely in simulation — by probing the internal representations of a frozen 2B-parameter world model.

---

## The Core Idea

Most robot policies act first and fail silently. Sim_Cortex inverts this: before any action is executed, it imagines K=3 possible futures using a frozen Cosmos-Predict2 world model, scores each future for predicted failure likelihood using layer-19 hidden states, and selects the safest trajectory.

The key insight: **layer 19 of 30 in Cosmos-Predict2 encodes physical causal structure** — what will happen next, not just what the scene looks like. A lightweight 1.6M-parameter MLP trained on simulated failure data is enough to read this signal.

```
MuJoCo (5 frames) → mimic-video tokenizer → Cosmos-Predict2 DiT (2B, frozen)
  → K=3 imagined futures → layer-19 hidden states (2048-dim each)
  → PRO scorer (3 heads: progress / risk / efficiency)
  → safest trajectory → action decoder → Franka Panda executes
```

---

## Results

Different failure modes produce different best trajectories — the scorer correctly discriminates between them without any real robot data:

| Scenario        | Best trajectory | PRO score | vs. random selection |
|-----------------|-----------------|-----------|----------------------|
| Normal physics  | K=3             | 0.4155    | ✓ correct            |
| Slippery object | K=1             | 0.4474    | ✓ correct            |
| Heavy object    | K=2             | 0.4071    | ✓ correct            |

All three failure modes resolve to different candidates — confirming the scorer is reading distinct physical signatures, not guessing.

---

## Architecture

### Simulation & Data
- **MuJoCo + robosuite** — 4 failure modes: normal, slippery, heavy, extreme pose
- Failure labels generated fully in simulation; zero real robot demonstrations required

### World Model Backbone
- **mimic-video** (ETH Zürich + MIT) — Cosmos-Predict2 2B as video world model
- Model is **frozen** — no fine-tuning; hidden states extracted at layer 19 of 30

### PRO Scorer
- **3-head MLP** (1.6M parameters) trained on simulated failure data
- Heads: progress prediction, risk estimation, efficiency scoring
- Input: 2048-dim latent vector from layer-19 hidden state

### Action Decoder
- **World2ActionPipeline** — converts selected latent trajectory to Franka joint commands

---

## Quick Start

```bash
# Clone
git clone https://github.com/sadrev25/Sim_Cortex.git
cd Sim_Cortex

# Setup (see mimic-video repo for model weights)
source /path/to/mimic-video/.venv/bin/activate

# Generate failure dataset
python data/label_generator_mimic.py

# Train PRO scorer
python pro_scorer/train.py

# Run K=3 scoring experiment
python experiments/k3_scoring_experiment.py

# Full pipeline demo
python experiments/full_demo.py
```

---

## Built With

- [mimic-video](https://github.com/nicklashansen/mimic-video) — ETH Zürich + MIT
- MuJoCo + robosuite
- Cosmos-Predict2 2B (NVIDIA)
- Franka Panda (target hardware)

---

## What's Next

### Short term
- [ ] PCA visualisation of layer-19 latent clusters by failure mode
- [ ] LIBERO fine-tuned action decoder for better pick-and-place
- [ ] Full demo video

### Research direction — latent space fine-tuning
The central open question this project raises: **can we fine-tune directly at layer 19 rather than at the action decoder?**

Standard approaches (LoRA, full fine-tuning) update the action output layer — the "muscle" of the model. But if layer 19 is where physical causal reasoning lives, updating the representation at that layer should generalise across tasks with far less data.

Current hypothesis: a **ReFT-style intervention** (a low-rank residual added to the layer-19 hidden state, ~16K parameters) should adapt robot behaviour to new domains using 10–20 demonstrations — compared to 200–500 demonstrations for LoRA fine-tuning of the action decoder.

This is the research direction I'm actively pursuing as a PhD proposal.

---

## Author

**Mukesh Sadasivam**  
Erasmus Mundus Master's in Intelligent Robotics — University of Stuttgart, University of Toulon, UJI  
[LinkedIn](https://linkedin.com/in/mukesh-sadasivam) · [GitHub](https://github.com/sadrev25)m/in/mukesh-sadasivam)
