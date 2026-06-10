"""
K=3 PRO scoring experiment:
Same scene → 3 imagined futures → PRO picks safest
This is the core Cortex 2.0 replication!
"""
import sys, os
sys.path.insert(0, '/home/itm/msadasivam/mimic-video/model')
sys.path.insert(0, '/home/itm/msadasivam/simcortex')
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.5'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
import robosuite as suite
from einops import rearrange
import cv2
import json

from cosmos_predict2.configs.config import make_config
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.conditioner import DataType
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import override
from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder
from pro_scorer.scorer import PROScorer

DEVICE = 'cuda'

def get_text_emb(config):
    t5 = CosmosT5TextEncoder(
        config=config.model.config.video_pipe_config.text_encoder.t5,
        device='cpu', torch_dtype=None,
    )
    emb, _ = t5.encode_prompts(["pick up the object"], return_mask=True)
    del t5; torch.cuda.empty_cache()
    return emb

def tokenize(config, frames):
    tok = instantiate(config.model.config.video_pipe_config.tokenizer)
    tok.to(device=DEVICE, dtype=torch.bfloat16)
    processed = []
    for f in frames:
        f = cv2.resize(f, (640, 480))
        f = 2.0*(f.astype(np.float32)/255.0-0.5)
        f = rearrange(f, 'h w c -> c h w')
        processed.append(f)
    video = torch.from_numpy(np.stack(processed,axis=1)).unsqueeze(0).bfloat16().to(DEVICE)
    with torch.no_grad():
        tokens = tok.encode(video).clone()
    del tok; torch.cuda.empty_cache()
    return tokens

def get_k_latents(config, tokens, text_emb, K=3):
    """Generate K different futuristic latents — sequential loading."""
    latents = []
    for k in range(K):
        # load DiT fresh for each candidate
        pipe = Video2WorldPipeline.from_config(
            config=config.model.config.video_pipe_config,
            dit_path='/home/itm/msadasivam/mimic-video/model/checkpoints/video_backbone/v2w_pretrained_cosmos.pt',
            device=DEVICE, torch_dtype=torch.bfloat16,
            load_ema_to_reg=False, offload_text_encoder=True,
        )
        torch.manual_seed(k * 42)
        noise = torch.randn_like(tokens)
        B,C,T,H,W = noise.shape
        with torch.no_grad():
            result = pipe.dit(
                x_B_C_T_H_W=noise,
                timesteps_B_T=torch.ones(B,T,device=DEVICE).bfloat16(),
                crossattn_emb=text_emb.to(DEVICE).bfloat16(),
                condition_video_input_mask_B_C_T_H_W=None,
                padding_mask=torch.ones(B,H,W,device=DEVICE).bfloat16(),
                data_type=DataType.IMAGE,
                return_only_hidden_states_up_to=19,
            )
        latent = result[1][19].float().mean(dim=(0,1,2,3))
        latents.append(latent.cpu())
        del pipe, result
        torch.cuda.empty_cache()
        print(f"  K={k+1} latent done")
    return latents

def run_experiment(perturbation='normal', K=3):
    print(f"\n{'='*50}")
    print(f"K={K} Scoring Experiment — {perturbation}")
    print(f"{'='*50}")

    config = make_config()
    config = override(config, ['--',
        'experiment=w2a_bridge_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz128'])
    config.model.config.video_pipe_config.guardrail_config.enabled = False

    text_emb = get_text_emb(config)

    # load trained PRO
    pro = PROScorer(latent_dim=2048).to(DEVICE)
    weights_path = '/home/itm/msadasivam/simcortex/pro_scorer/pro_weights_mimic.pt'
    if os.path.exists(weights_path):
        pro.load_state_dict(torch.load(weights_path))
        print("Loaded trained PRO weights!")
    else:
        print("WARNING: Using untrained PRO!")
    pro.eval()

    # setup env
    env = suite.make(
        env_name='Lift', robots='Panda',
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names='agentview',
        camera_heights=480, camera_widths=640,
        control_freq=20, horizon=200,
    )

    obs = env.reset()

    # apply failure
    if perturbation == 'slippery':
        for i in range(env.sim.model.ngeom):
            name = env.sim.model.geom_id2name(i) or ''
            if any(o in name for o in ['cube','object']):
                env.sim.model.geom_friction[i] = [0.05, 0.005, 0.0001]
    elif perturbation == 'heavy':
        for i in range(env.sim.model.nbody):
            name = env.sim.model.body_id2name(i) or ''
            if any(o in name for o in ['cube','object']):
                env.sim.model.body_mass[i] = 5.0

    # collect 5 frames
    frame_buffer = []
    for _ in range(5):
        action = np.random.uniform(env.action_spec[0], env.action_spec[1])
        obs, _, _, _ = env.step(action)
        frame_buffer.append(np.ascontiguousarray(obs['agentview_image']))

    # tokenize once
    tokens = tokenize(config, frame_buffer)

    # get K futuristic latents
    print(f"\nGenerating K={K} futuristic latents...")
    latents = get_k_latents(config, tokens, text_emb, K=K)

    # PRO scores each
    print(f"\nPRO scoring K={K} imagined futures:")
    print(f"{'Candidate':>10} | {'Progress':>8} | {'Risk':>8} | {'Efficiency':>10} | {'PRO Score':>10}")
    print("-"*55)

    scores = []
    for k, latent in enumerate(latents):
        with torch.no_grad():
            result = pro(latent.unsqueeze(0).to(DEVICE))
        prog = result['progress'].item()
        risk = result['risk'].item()
        eff = result['efficiency'].item()
        score = result['score'].item()
        scores.append(score)
        print(f"K={k+1:>8} | {prog:>8.3f} | {risk:>8.3f} | {eff:>10.3f} | {score:>10.3f}")

    best_k = np.argmax(scores)
    print(f"\nBest candidate: K={best_k+1} (score={scores[best_k]:.3f})")
    print(f"Worst candidate: K={np.argmin(scores)+1} (score={min(scores):.3f})")
    print(f"Score advantage: {max(scores)-min(scores):.3f}")

    env.close()
    return scores

if __name__ == '__main__':
    print("K=3 PRO Scoring Experiment")
    print("Replicating Cortex 2.0 Plan→Score→Execute")
    print()

    results = {}
    for pert in ['normal', 'slippery', 'heavy']:
        scores = run_experiment(perturbation=pert, K=3)
        results[pert] = scores

    print("\n" + "="*50)
    print("SUMMARY — PRO scores by failure mode:")
    print("="*50)
    for pert, scores in results.items():
        print(f"{pert:>12}: K1={scores[0]:.3f} K2={scores[1]:.3f} K3={scores[2]:.3f} best={max(scores):.3f}")
