"""
Full sim-cortex demo:
frames → mimic-video backbone → action decoder → MuJoCo execution
"""
import sys
import os
sys.path.insert(0, '/home/itm/msadasivam/mimic-video/model')
sys.path.insert(0, '/home/itm/msadasivam/simcortex')
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.5'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
import json
import time
from einops import rearrange
from scipy.spatial.transform import Rotation
import robosuite as suite
import imageio
import os

from cosmos_predict2.configs.config import make_config
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.conditioner import DataType
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import override
from cosmos_predict2.data.action.utils import extract_normalization_types


def get_proprioception(obs):
    """Convert MuJoCo obs to mimic-video 10-dim state."""
    eef_pos = obs['robot0_eef_pos']  # (3,)
    eef_quat = obs['robot0_eef_quat']  # (4,) xyzw
    gripper = obs['robot0_gripper_qpos'][0]  # scalar

    # convert quat to 6D rotation
    rot_matrix = Rotation.from_quat(eef_quat).as_matrix()
    rot_6d = rot_matrix[:2].reshape(6)  # top 2 rows = 6D

    state = np.concatenate([eef_pos, rot_6d, [gripper]])  # (10,)
    return state


def convert_action_to_mujoco(action_10d):
    """Convert mimic-video 10-dim action to MuJoCo 7-dim action."""
    delta_pos = action_10d[:3]
    rot_6d = action_10d[3:9]
    gripper = action_10d[9]

    # reconstruct rotation matrix from 6D
    r1 = rot_6d[:3]
    r2 = rot_6d[3:6]
    r1 = r1 / (np.linalg.norm(r1) + 1e-9)
    r2 = r2 - np.dot(r2, r1) * r1
    r2 = r2 / (np.linalg.norm(r2) + 1e-9)
    r3 = np.cross(r1, r2)
    rot_matrix = np.stack([r1, r2, r3])

    rot_vec = Rotation.from_matrix(rot_matrix).as_rotvec()
    gripper_cmd = 1.0 if gripper > 0 else -1.0

    return np.concatenate([delta_pos, rot_vec, [gripper_cmd]])


def tokenize_frames(config, frames):
    """Tokenize 5 frames on GPU then free tokenizer."""
    tokenizer = instantiate(config.model.config.video_pipe_config.tokenizer)
    tokenizer.to(device='cuda', dtype=torch.bfloat16)

    processed = []
    for f in frames:
        import cv2
        f = cv2.resize(f, (640, 480))
        f = 2.0 * (f.astype(np.float32) / 255.0 - 0.5)
        f = rearrange(f, 'h w c -> c h w')
        processed.append(f)

    video = np.stack(processed, axis=1)
    video = torch.from_numpy(video).unsqueeze(0).bfloat16().cuda()

    with torch.no_grad():
        tokens = tokenizer.encode(video).clone()

    del tokenizer
    torch.cuda.empty_cache()
    return tokens


def get_hidden_states(config, tokens, text_emb):
    """Run backbone DiT to get layer 19 hidden states."""
    backbone_pipe = Video2WorldPipeline.from_config(
        config=config.model.config.video_pipe_config,
        dit_path='/home/itm/msadasivam/mimic-video/model/checkpoints/video_backbone/v2w_pretrained_cosmos.pt',
        device='cuda',
        torch_dtype=torch.bfloat16,
        load_ema_to_reg=False,
        offload_text_encoder=True,
    )

    noise = torch.randn_like(tokens)
    B, C, T, H, W = noise.shape
    timesteps = torch.ones(B, T, device='cuda').bfloat16()
    padding_mask = torch.ones(B, H, W, device='cuda').bfloat16()
    text_emb_cuda = text_emb.to('cuda').bfloat16()

    with torch.no_grad():
        result = backbone_pipe.dit(
            x_B_C_T_H_W=noise,
            timesteps_B_T=timesteps,
            crossattn_emb=text_emb_cuda,
            condition_video_input_mask_B_C_T_H_W=None,
            padding_mask=padding_mask,
            data_type=DataType.IMAGE,
            return_only_hidden_states_up_to=19,
        )

    hidden = result[1][19].clone()
    del backbone_pipe, result
    torch.cuda.empty_cache()
    return hidden


def decode_actions(config, hidden_states, proprio, seed=0):
    """Decode hidden states to robot actions."""
    world2action_pipe = World2ActionPipeline.from_config(
        config.model.config.pipe_config,
        dit_path='/home/itm/msadasivam/mimic-video-weights/action_decoder/w2a_bridge_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz256_iter_000014112.pt',
        device='cuda',
        dtype=torch.bfloat16,
    )

    with open('/home/itm/msadasivam/mimic-video-weights/dataset_statistics/bridge.json') as f:
        stats = json.load(f)

    data_config = instantiate(config.data_config)
    world2action_pipe.normalizer.build_from_stats(
        stats,
        normalization_types=extract_normalization_types(data_config.policy_io.policy_io),
        concat_groups=data_config.policy_io.concat_groups,
        device='cuda',
        dtype=torch.bfloat16,
    )

    B_h, T_h, H_h, W_h, D_h = hidden_states.shape
    crossattn = hidden_states.reshape(B_h, T_h*H_h*W_h, D_h).bfloat16()

    proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).unsqueeze(0).bfloat16().cuda()
    context_timestep = torch.ones(1, 1, device='cuda').bfloat16()

    with torch.no_grad():
        actions = world2action_pipe(
            state_B_HO_O=proprio_tensor,
            crossattn_emb=crossattn,
            context_timesteps_B_1=context_timestep,
            seed=seed,
        )

    del world2action_pipe
    torch.cuda.empty_cache()
    return actions[0].float().cpu().numpy()  # (15, 10)


def main():
    print("=" * 60)
    print("sim-cortex Full Demo")
    print("frames → mimic-video → action decoder → MuJoCo")
    print("=" * 60)

    # load config once
    config = make_config()
    config = override(config, [
        '--',
        'experiment=w2a_bridge_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz128'
    ])
    config.model.config.video_pipe_config.guardrail_config.enabled = False

    # pre-compute text embedding on CPU
    print("\nPre-computing text embedding...")
    from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder
    t5 = CosmosT5TextEncoder(
        config=config.model.config.video_pipe_config.text_encoder.t5,
        device='cpu',
        torch_dtype=None,
    )
    text_emb, _ = t5.encode_prompts(
        ["pick up the object"],
        return_mask=True,
    )
    del t5
    torch.cuda.empty_cache()
    print(f"Text embedding shape: {text_emb.shape}")

    # setup MuJoCo
    print("\nSetting up MuJoCo environment...")
    env = suite.make(
        env_name='Lift',
        robots='Panda',
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names='agentview',
        camera_heights=480,
        camera_widths=640,
        control_freq=20,
        horizon=500,
    )

    obs = env.reset()
    frame_buffer = []
    total_reward = 0
    action_count = 0

    print("\nRunning simulation...")
    print(f"{'Step':>5} | {'Reward':>8} | {'EEF pos':>20} | Status")
    print("-" * 60)

    for episode_step in range(20):
        # collect current frame
        frame = obs['agentview_image']
        frame_buffer.append(frame)

        # keep only last 5 frames
        if len(frame_buffer) > 5:
            frame_buffer = frame_buffer[-5:]

        # need 5 frames to run pipeline
        if len(frame_buffer) < 5:
            # random action until we have 5 frames
            action = np.random.uniform(
                env.action_spec[0], env.action_spec[1]
            )
            obs, reward, done, info = env.step(action)
            continue

        # get proprioception
        proprio = get_proprioception(obs)

        print(f"\nRunning mimic-video pipeline (step {episode_step})...")

        # Step 1: tokenize
        tokens = tokenize_frames(config, frame_buffer)

        # Step 2: backbone → hidden states
        hidden = get_hidden_states(config, tokens, text_emb)
        print(f"  Hidden states: {hidden.shape}")

        # Step 3: decode actions
        actions_15 = decode_actions(config, hidden, proprio, seed=episode_step)
        print(f"  Actions: {actions_15.shape} → executing {len(actions_15)} steps")

        # Step 4: execute actions in MuJoCo
        for i, action_10d in enumerate(actions_15):
            mujoco_action = convert_action_to_mujoco(action_10d)
            obs, reward, done, info = env.step(mujoco_action)
            total_reward += reward
            action_count += 1

            eef = obs['robot0_eef_pos']
            status = "SUCCESS!" if reward > 0 else ""
            print(f"{action_count:>5} | {reward:>8.3f} | "
                  f"{eef[0]:.3f},{eef[1]:.3f},{eef[2]:.3f} | {status}")

            if reward > 0:
                print("\n🎉 Robot successfully completed the task!")

            if done:
                break

        if done:
            break

    print(f"\nTotal reward: {total_reward:.3f}")
    print(f"Total actions: {action_count}")
    env.close()


if __name__ == '__main__':
    main()
