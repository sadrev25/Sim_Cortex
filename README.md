# sim-cortex — Plan → Score → Execute

Open-source replication of Sereact's Cortex 2.0 PRO scoring architecture in simulation.

**No real robot data needed. Failure signatures learned entirely in simulation.**

## What It Does

Robots fail at things nobody prepared them for. This project systematically generates failure data in simulation and trains a PRO scorer to recognize failure signatures in latent space — before any action is executed.

Pipeline:
MuJoCo (5 frames) → mimic-video tokenizer → Cosmos-Predict2 DiT (2B)
→ K=3 imagined futures (2048-dim latents each)
→ PRO scorer picks safest trajectory
→ Action decoder → Franka Panda executes

## Results

| Scenario | Best candidate | PRO score |
|----------|---------------|-----------|
| Normal physics | K=3 | 0.4155 |
| Slippery object | K=1 | 0.4474 |
| Heavy object | K=2 | 0.4071 |

Different failure modes → different best trajectories selected.

## Architecture

- **MuJoCo + robosuite** — 4 failure modes: normal, slippery, heavy, extreme pose
- **mimic-video** (ETH Zürich + MIT) — Cosmos-Predict2 2B as world model backbone
- **PRO scorer** — 3 MLP heads (progress, risk, efficiency) trained on sim failure data
- **Action decoder** — World2ActionPipeline converting latents to joint commands

## Quick Start

```bash
# Clone
git clone https://github.com/sadrev25/Sim_Cortex.git
cd Sim_Cortex

# Setup mimic-video (see mimic-video repo for weights)
source /path/to/mimic-video/.venv/bin/activate

# Generate dataset
python data/label_generator_mimic.py

# Train PRO scorer
python pro_scorer/train.py

# Run K=3 experiment
python experiments/k3_scoring_experiment.py

# Run full pipeline demo
python experiments/full_demo.py
```

## Built With

- [mimic-video](https://github.com/agi-collective/mimic-video) — ETH Zürich + MIT
- [MuJoCo](https://mujoco.org/) + [robosuite](https://robosuite.ai/)
- Cosmos-Predict2 2B backbone
- Inspired by [Sereact Cortex 2.0](https://sereact.ai/)

## What's Next

- [ ] Full failure visualization with PCA latent clusters
- [ ] LIBERO fine-tuned action decoder for better picking
- [ ] G1 humanoid integration
- [ ] Full demo video before Embodied AI Europe Summit

## Author

Mukesh Sadasivam — Erasmus Mundus Master's student at University of Stuttgart

[LinkedIn](https://linkedin.com/in/mukesh-sadasivam)
