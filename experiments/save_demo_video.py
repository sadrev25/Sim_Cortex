"""
Save demo video of mimic-video pipeline controlling Franka Panda.
"""
import sys, os
sys.path.insert(0, '/home/itm/msadasivam/mimic-video/model')
sys.path.insert(0, '/home/itm/msadasivam/simcortex')
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.5'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
import json
import imageio
from einops import rearrange
from scipy.spatial.transform import Rotation
import robosuite as suite

from cosmos_predict2.configs.config import make_config
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.conditioner import DataType
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import override
from cosmos_predict2.data.action.utils import extract_normalization_types
from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder

def get_proprioception(obs):
    eef_pos = obs['robot0_eef_pos']
    eef_quat = obs['robot0_eef_quat']
    gripper = obs['robot0_gripper_qpos'][0]
    rot_matrix = Rotation.from_quat(eef_quat).as_matrix()
    rot_6d = rot_matrix[:2].reshape(6)
    return np.concatenate([eef_pos, rot_6d, [gripper]])

def convert_action(action_10d):
    delta_pos = action_10d[:3]
    rot_6d = action_10d[3:9]
    gripper = action_10d[9]
    r1 = rot_6d[:3] / (np.linalg.norm(rot_6d[:3]) + 1e-9)
    r2 = rot_6d[3:6]
    r2 = r2 - np.dot(r2, r1) * r1
    r2 = r2 / (np.linalg.norm(r2) + 1e-9)
    r3 = np.cross(r1, r2)
    rot_vec = Rotation.from_matrix(np.stack([r1,r2,r3])).as_rotvec()
    return np.concatenate([delta_pos, rot_vec, [1.0 if gripper > 0 else -1.0]])

def tokenize(config, frames):
    tok = instantiate(config.model.config.video_pipe_config.tokenizer)
    tok.to(device='cuda', dtype=torch.bfloat16)
    processed = []
    import cv2
    for f in frames:
        f = cv2.resize(f, (640, 480))
        f = 2.0*(f.astype(np.float32)/255.0-0.5)
        f = rearrange(f,'h w c -> c h w')
        processed.append(f)
    video = torch.from_numpy(np.stack(processed,axis=1)).unsqueeze(0).bfloat16().cuda()
    with torch.no_grad():
        tokens = tok.encode(video).clone()
    del tok
    torch.cuda.empty_cache()
    return tokens

def get_hidden(config, tokens, text_emb):
    pipe = Video2WorldPipeline.from_config(
        config=config.model.config.video_pipe_config,
        dit_path='/home/itm/msadasivam/mimic-video/model/checkpoints/video_backbone/v2w_pretrained_cosmos.pt',
        device='cuda', torch_dtype=torch.bfloat16,
        load_ema_to_reg=False, offload_text_encoder=True,
    )
    noise = torch.randn_like(tokens)
    B,C,T,H,W = noise.shape
    with torch.no_grad():
        result = pipe.dit(
            x_B_C_T_H_W=noise,
            timesteps_B_T=torch.ones(B,T,device='cuda').bfloat16(),
            crossattn_emb=text_emb.to('cuda').bfloat16(),
            condition_video_input_mask_B_C_T_H_W=None,
            padding_mask=torch.ones(B,H,W,device='cuda').bfloat16(),
            data_type=DataType.IMAGE,
            return_only_hidden_states_up_to=19,
        )
    hidden = result[1][19].clone()
    del pipe, result
    torch.cuda.empty_cache()
    return hidden

def decode_actions(config, hidden, proprio, seed=0):
    w2a = World2ActionPipeline.from_config(
        config.model.config.pipe_config,
        dit_path='/home/itm/msadasivam/mimic-video-weights/action_decoder/w2a_bridge_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz256_iter_000014112.pt',
        device='cuda', dtype=torch.bfloat16,
    )
    with open('/home/itm/msadasivam/mimic-video-weights/dataset_statistics/bridge.json') as f:
        stats = json.load(f)
    data_config = instantiate(config.data_config)
    w2a.normalizer.build_from_stats(
        stats,
        normalization_types=extract_normalization_types(data_config.policy_io.policy_io),
        concat_groups=data_config.policy_io.concat_groups,
        device='cuda', dtype=torch.bfloat16,
    )
    B,T,H,W,D = hidden.shape
    crossattn = hidden.reshape(B,T*H*W,D).bfloat16()
    proprio_t = torch.from_numpy(proprio).unsqueeze(0).unsqueeze(0).bfloat16().cuda()
    with torch.no_grad():
        actions = w2a(
            state_B_HO_O=proprio_t,
            crossattn_emb=crossattn,
            context_timesteps_B_1=torch.ones(1,1,device='cuda').bfloat16(),
            seed=seed,
        )
    del w2a
    torch.cuda.empty_cache()
    return actions[0].float().cpu().numpy()

def main():
    print("sim-cortex Demo Video Generator")
    print("="*50)

    config = make_config()
    config = override(config, ['--',
        'experiment=w2a_bridge_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz128'])
    config.model.config.video_pipe_config.guardrail_config.enabled = False

    print("Loading T5...")
    t5 = CosmosT5TextEncoder(
        config=config.model.config.video_pipe_config.text_encoder.t5,
        device='cpu', torch_dtype=None,
    )
    text_emb, _ = t5.encode_prompts(["pick up the object"], return_mask=True)
    del t5
    torch.cuda.empty_cache()

    print("Setting up MuJoCo...")
    env = suite.make(
        env_name='Lift', robots='Panda',
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names='agentview',
        camera_heights=480, camera_widths=640,
        control_freq=20, horizon=500,
    )

    obs = env.reset()
    frame_buffer = []
    all_frames = []
    step_count = 0

    print("Running and saving frames...")
    os.makedirs('/home/itm/msadasivam/simcortex/demo_frames', exist_ok=True)

    for episode_step in range(15):
        frame = obs['agentview_image']
        frame_buffer.append(frame)
        # save every frame for video
        all_frames.append(frame.copy())
        imageio.imwrite(
            f'/home/itm/msadasivam/simcortex/demo_frames/frame_{len(all_frames):04d}.png',
            frame
        )

        if len(frame_buffer) > 5:
            frame_buffer = frame_buffer[-5:]

        if len(frame_buffer) < 5:
            action = np.random.uniform(env.action_spec[0], env.action_spec[1])
            obs, _, _, _ = env.step(action)
            continue

        proprio = get_proprioception(obs)
        print(f"\nStep {episode_step}: running pipeline...")

        tokens = tokenize(config, frame_buffer)
        hidden = get_hidden(config, tokens, text_emb)
        actions_15 = decode_actions(config, hidden, proprio, seed=episode_step)

        for action_10d in actions_15:
            mujoco_action = convert_action(action_10d)
            obs, reward, done, _ = env.step(mujoco_action)
            step_count += 1

            frame = obs['agentview_image']
            all_frames.append(frame.copy())
            imageio.imwrite(
                f'/home/itm/msadasivam/simcortex/demo_frames/frame_{len(all_frames):04d}.png',
                frame
            )

            eef = obs['robot0_eef_pos']
            print(f"  {step_count}: eef={eef.round(3)} reward={reward:.3f}")

            if reward > 0:
                print("SUCCESS!")
            if done:
                break

    env.close()

    # save video
    print(f"\nSaving video ({len(all_frames)} frames)...")
    imageio.mimsave(
        '/home/itm/msadasivam/simcortex/demo_video.mp4',
        all_frames, fps=15
    )
    print("Video saved: ~/simcortex/demo_video.mp4")
    print(f"Frames saved: ~/simcortex/demo_frames/")

if __name__ == '__main__':
    main()
