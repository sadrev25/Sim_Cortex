import sys
sys.path.insert(0, '/home/itm/msadasivam/mimic-video/model')
sys.path.insert(0, '/home/itm/msadasivam/simcortex')
import os
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.5'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import pickle
from tqdm import tqdm
import robosuite as suite
from utils.physics_utils import compute_risk, compute_progress, compute_efficiency

# mimic-video imports
from cosmos_predict2.configs.config import make_config
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.conditioner import DataType
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import override
from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder
from einops import rearrange
import cv2

DEVICE = 'cuda'
LANGUAGE = "pick up the object and place it in the bin"


def load_config():
    config = make_config()
    config = override(config, [
        '--',
        'experiment=w2a_bridge_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz128'
    ])
    config.model.config.video_pipe_config.guardrail_config.enabled = False
    return config


def get_text_embedding(config):
    print("Loading T5 text encoder...")
    t5 = CosmosT5TextEncoder(
        config=config.model.config.video_pipe_config.text_encoder.t5,
        device='cpu',
        torch_dtype=None,
    )
    text_emb, _ = t5.encode_prompts([LANGUAGE], return_mask=True)
    del t5
    torch.cuda.empty_cache()
    return text_emb


def tokenize_frames(config, frames_5):
    """Tokenize 5 frames on GPU then free tokenizer."""
    tokenizer = instantiate(config.model.config.video_pipe_config.tokenizer)
    tokenizer.to(device=DEVICE, dtype=torch.bfloat16)

    processed = []
    for f in frames_5:
        f = cv2.resize(f, (640, 480))
        f = 2.0 * (f.astype(np.float32) / 255.0 - 0.5)
        f = rearrange(f, 'h w c -> c h w')
        processed.append(f)

    video = np.stack(processed, axis=1)
    video = torch.from_numpy(video).unsqueeze(0).bfloat16().to(DEVICE)

    with torch.no_grad():
        tokens = tokenizer.encode(video).clone()

    del tokenizer
    torch.cuda.empty_cache()
    return tokens


def encode_with_backbone(config, tokens, text_emb):
    """Run DiT backbone to get layer 19 hidden states."""
    pipe = Video2WorldPipeline.from_config(
        config=config.model.config.video_pipe_config,
        dit_path='/home/itm/msadasivam/mimic-video/model/checkpoints/video_backbone/v2w_pretrained_cosmos.pt',
        device=DEVICE,
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=False,
        offload_text_encoder=True,
    )

    noise = torch.randn_like(tokens)
    B, C, T, H, W = noise.shape
    timesteps = torch.ones(B, T, device=DEVICE).bfloat16()
    padding_mask = torch.ones(B, H, W, device=DEVICE).bfloat16()
    text_emb_cuda = text_emb.to(DEVICE).bfloat16()

    with torch.no_grad():
        result = pipe.dit(
            x_B_C_T_H_W=noise,
            timesteps_B_T=timesteps,
            crossattn_emb=text_emb_cuda,
            condition_video_input_mask_B_C_T_H_W=None,
            padding_mask=padding_mask,
            data_type=DataType.IMAGE,
            return_only_hidden_states_up_to=19,
        )

    hidden = result[1][19]  # (1, 2, 30, 40, 2048)
    latent = hidden.float().mean(dim=(0, 1, 2, 3))  # (2048,)

    del pipe, result
    torch.cuda.empty_cache()
    return latent.cpu()


def make_env():
    return suite.make(
        env_name='PickPlace',
        robots='Panda',
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names='agentview',
        camera_heights=480,
        camera_widths=640,
        reward_shaping=True,
        control_freq=20,
        horizon=200,
    )


def apply_perturbation(env, perturbation):
    if perturbation == 'slippery':
        for i in range(env.sim.model.ngeom):
            name = env.sim.model.geom_id2name(i) or ''
            if any(o in name for o in ['Milk','Bread','Cereal','Can']):
                env.sim.model.geom_friction[i] = [0.05, 0.005, 0.0001]
    elif perturbation == 'heavy':
        for i in range(env.sim.model.nbody):
            name = env.sim.model.body_id2name(i) or ''
            if any(o in name for o in ['Milk','Bread','Cereal','Can']):
                env.sim.model.body_mass[i] = 5.0
    elif perturbation == 'extreme_pose':
        noise = np.random.uniform(-0.3, 0.3, 7)
        env.sim.data.qpos[:7] += noise
        env.sim.forward()


def collect_episode(env, config, text_emb, perturbation, steps=50):
    obs = env.reset()
    apply_perturbation(env, perturbation)

    episode_data = []
    frame_buffer = []

    try:
        eef = obs['robot0_eef_pos']
        obj = obs['object-state'][:3]
        init_dist = float(np.linalg.norm(eef - obj))
    except:
        init_dist = 1.0

    for step in range(steps):
        # collect frame
        frame = obs['agentview_image']
        frame_buffer.append(frame)
        if len(frame_buffer) > 5:
            frame_buffer = frame_buffer[-5:]

        # need 5 frames for mimic-video
        if len(frame_buffer) < 5:
            action = np.random.uniform(env.action_spec[0], env.action_spec[1])
            obs, _, _, _ = env.step(action)
            continue

        # compute physics labels
        try:
            eef = obs['robot0_eef_pos']
            obj = obs['object-state'][:3]
            dist = float(np.linalg.norm(eef - obj))
            progress = float(np.clip(1 - dist/init_dist, 0, 1))
        except:
            progress = 0.0

        qvel = env.sim.data.qvel[:7]
        risk = float(np.clip(
            (np.any(np.abs(qvel) > 2.0) * 0.4) +
            (0.4 if perturbation == 'slippery' else 0) +
            (0.3 if perturbation == 'heavy' else 0) +
            (0.5 if perturbation == 'extreme_pose' else 0),
            0, 1
        ))
        efficiency = float(np.clip(1 - step/steps, 0, 1))

        # encode with mimic-video
        tokens = tokenize_frames(config, frame_buffer)
        latent = encode_with_backbone(config, tokens, text_emb)

        episode_data.append({
            'latent': latent.numpy(),
            'progress': progress,
            'risk': risk,
            'efficiency': efficiency,
            'perturbation': perturbation,
            'step': step,
        })

        # random action
        action = np.random.uniform(env.action_spec[0], env.action_spec[1])
        obs, _, done, _ = env.step(action)
        if done:
            break

    return episode_data


def main():
    print("=" * 60)
    print("Generating mimic-video PRO dataset")
    print("=" * 60)

    config = load_config()
    text_emb = get_text_embedding(config)
    print(f"Text embedding: {text_emb.shape}")

    perturbations = ['normal', 'slippery', 'heavy', 'extreme_pose']
    episodes_per_mode = 3  # 3 episodes per mode = 12 total
    steps_per_episode = 8  # only 8 steps per episode

    all_data = []
    env = make_env()

    for pert in perturbations:
        print(f"\n--- Perturbation: {pert} ---")
        for ep in range(episodes_per_mode):
            print(f"  Episode {ep+1}/{episodes_per_mode}...")
            try:
                data = collect_episode(
                    env, config, text_emb, pert,
                    steps=steps_per_episode
                )
                all_data.extend(data)
                print(f"  Got {len(data)} samples. Total: {len(all_data)}")
            except Exception as e:
                print(f"  Error: {e}")
                continue

    env.close()

    # save dataset
    save_path = '/home/itm/msadasivam/simcortex/data/pro_dataset_mimic.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(all_data, f)

    print(f"\nDataset saved: {save_path}")
    print(f"Total samples: {len(all_data)}")
    print(f"Sample keys: {all_data[0].keys()}")
    print(f"Latent shape: {all_data[0]['latent'].shape}")
    print("\nDone!")


if __name__ == '__main__':
    main()
