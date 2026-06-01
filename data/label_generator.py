import sys
sys.path.insert(0, '/home/mukesh/simcortex')
import numpy as np
import torch
import pickle
import os
from tqdm import tqdm
import robosuite as suite
from world_model.backbone import VisualEncoder
from utils.physics_utils import compute_risk, compute_progress, compute_efficiency

def make_env_with_camera():
    return suite.make(
        env_name='PickPlace',
        robots='Panda',
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names='agentview',
        camera_heights=84,
        camera_widths=84,
        reward_shaping=True,
        control_freq=20,
        horizon=200,
    )

def collect_episode(env, encoder, perturbation, device, steps_per_episode=50):
    obs = env.reset()
    episode_data = []

    # apply perturbation
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
        import mujoco
        noise = np.random.uniform(-0.3, 0.3, 7)
        env.sim.data.qpos[:7] += noise
        env.sim.forward()

    # initial distance
    try:
        eef_pos = obs['robot0_eef_pos']
        obj_pos = obs['object-state'][:3]
        initial_distance = float(np.linalg.norm(eef_pos - obj_pos))
    except Exception:
        initial_distance = 1.0
    if initial_distance < 1e-6:
        initial_distance = 1.0

    for step in range(steps_per_episode):
        # encode image to latent
        image = obs['agentview_image']
        with torch.no_grad():
            latent = encoder.encode(image).cpu().numpy()

        # compute labels
        progress, distance = compute_progress(obs, initial_distance)
        risk       = compute_risk(env.sim, obs, perturbation)
        efficiency = compute_efficiency(step, steps_per_episode)

        episode_data.append({
            'latent':       latent,
            'progress':     progress,
            'risk':         risk,
            'efficiency':   efficiency,
            'perturbation': perturbation or 'normal',
            'step':         step,
        })

        # random action
        low, high = env.action_spec
        action = np.random.uniform(low, high)
        obs, reward, done, info = env.step(action)
        if done:
            break

    return episode_data


def generate_dataset(
    n_episodes=200,
    steps_per_episode=50,
    save_path='/home/mukesh/simcortex/data/pro_dataset.pkl',
    device='cuda'
):
    print("=" * 50)
    print("PRO Label Generator")
    print("=" * 50)

    print("\nLoading DINOv2 encoder...")
    encoder = VisualEncoder(device=device)

    perturbations = (
        [None]           * (n_episodes // 4) +
        ['slippery']     * (n_episodes // 4) +
        ['heavy']        * (n_episodes // 4) +
        ['extreme_pose'] * (n_episodes // 4)
    )
    np.random.shuffle(perturbations)

    print(f"\nGenerating {n_episodes} episodes...")
    print(f"  Normal:       {perturbations.count(None)}")
    print(f"  Slippery:     {perturbations.count('slippery')}")
    print(f"  Heavy:        {perturbations.count('heavy')}")
    print(f"  Extreme pose: {perturbations.count('extreme_pose')}")

    all_data = []
    stats = {'normal': 0, 'slippery': 0, 'heavy': 0, 'extreme_pose': 0}

    for ep_idx in tqdm(range(n_episodes), desc="Collecting"):
        perturbation = perturbations[ep_idx]
        try:
            env = make_env_with_camera()
            episode_data = collect_episode(
                env, encoder, perturbation, device, steps_per_episode
            )
            all_data.extend(episode_data)
            stats[perturbation or 'normal'] += 1
            env.close()
        except Exception as e:
            print(f"\nEpisode {ep_idx} failed: {e}")
            try: env.close()
            except: pass
            continue

    # save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset = {
        'data':       all_data,
        'stats':      stats,
        'n_episodes': n_episodes,
        'latent_dim': 384,
        'encoder':    'DINOv2-small',
    }
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)

    # summary
    print("\n" + "=" * 50)
    print("Dataset Generated!")
    print("=" * 50)
    print(f"Total timesteps: {len(all_data)}")
    print(f"Episodes by type: {stats}")

    progresses   = [d['progress']   for d in all_data]
    risks        = [d['risk']       for d in all_data]
    efficiencies = [d['efficiency'] for d in all_data]

    print(f"\nLabel statistics:")
    print(f"  Progress:   mean={np.mean(progresses):.3f} std={np.std(progresses):.3f}")
    print(f"  Risk:       mean={np.mean(risks):.3f} std={np.std(risks):.3f}")
    print(f"  Efficiency: mean={np.mean(efficiencies):.3f} std={np.std(efficiencies):.3f}")
    print(f"\nDataset size: {os.path.getsize(save_path)/1024/1024:.1f} MB")
    print("Ready for PRO training!")
    return all_data


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generate_dataset(n_episodes=40, steps_per_episode=50, device=device)
