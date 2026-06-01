import sys
sys.path.insert(0, '/home/mukesh/simcortex')
import numpy as np
import torch
import time
import robosuite as suite
from world_model.backbone import VisualEncoder
from pro_scorer.scorer import PROScorer

def run_failure_mode(
    perturbation_name,
    friction=None,
    mass=None,
    extreme_pose=False,
    encoder=None,
    scorer=None,
    device='cuda',
    steps=150,
    render=True,
    slow=True,
):
    print("\n" + "=" * 60)
    print(f"FAILURE MODE: {perturbation_name}")
    print("=" * 60)

    env = suite.make(
        env_name='PickPlace',
        robots='Panda',
        has_renderer=render,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names='agentview',
        camera_heights=84,
        camera_widths=84,
        control_freq=20,
        horizon=300,
    )
    obs = env.reset()

    # apply perturbation
    if friction:
        count = 0
        for i in range(env.sim.model.ngeom):
            name = env.sim.model.geom_id2name(i) or ''
            if any(o in name for o in ['Milk','Bread','Cereal','Can']):
                env.sim.model.geom_friction[i] = friction
                count += 1
        print(f"  Applied friction {friction} to {count} object geoms")

    if mass:
        count = 0
        for i in range(env.sim.model.nbody):
            name = env.sim.model.body_id2name(i) or ''
            if any(o in name for o in ['Milk','Bread','Cereal','Can']):
                env.sim.model.body_mass[i] = mass
                count += 1
        print(f"  Applied mass {mass}kg to {count} object bodies")

    if extreme_pose:
        import mujoco
        noise = np.random.uniform(-0.4, 0.4, 7)
        env.sim.data.qpos[:7] += noise
        env.sim.forward()
        print(f"  Applied extreme pose noise: {noise.round(2)}")

    # initial distance
    try:
        eef_pos = obs['robot0_eef_pos']
        obj_pos = obs['object-state'][:3]
        initial_distance = float(np.linalg.norm(eef_pos - obj_pos))
    except:
        initial_distance = 1.0

    print(f"\n{'Step':>5} | {'Progress':>8} | {'Risk':>6} | {'Efficiency':>10} | {'PRO Score':>9} | Physics State")
    print("-" * 80)

    prev_obj_pos = None
    episode_scores = []

    for step in range(steps):
        # get image and encode
        image = obs['agentview_image']
        with torch.no_grad():
            latent = encoder.encode(image)

        # compute physics labels
        try:
            eef_pos  = obs['robot0_eef_pos']
            obj_pos  = obs['object-state'][:3]
            distance = float(np.linalg.norm(eef_pos - obj_pos))
            progress = float(np.clip(
                1.0 - distance / initial_distance, 0.0, 1.0
            ))
        except:
            progress = 0.0
            obj_pos  = np.zeros(3)

        # object velocity — temporal signal
        try:
            obj_vel = np.linalg.norm(env.sim.data.qvel[9:12])
        except:
            obj_vel = 0.0

        # joint limit proximity
        qpos      = env.sim.data.qpos[:7]
        qlim_low  = env.sim.model.jnt_range[:7, 0]
        qlim_high = env.sim.model.jnt_range[:7, 1]
        range_sz  = qlim_high - qlim_low + 1e-6
        proximity = np.minimum(
            (qpos - qlim_low) / range_sz,
            (qlim_high - qpos) / range_sz
        )
        near_limit = np.any(proximity < 0.1)

        # risk signal
        risk = 0.0
        if near_limit:              risk += 0.4
        if obj_vel > 0.5:           risk += 0.3
        if np.any(np.abs(env.sim.data.qvel[:7]) > 2.0): risk += 0.3
        if friction:                risk += 0.1
        if mass:                    risk += 0.1
        if extreme_pose:            risk += 0.2
        risk = float(np.clip(risk, 0.0, 1.0))

        efficiency = float(np.clip(1.0 - step / steps, 0.0, 1.0))

        # PRO score
        with torch.no_grad():
            result = scorer(latent.unsqueeze(0))
            pro_score = result['score'].item()

        episode_scores.append(pro_score)

        # physics state description
        state_desc = []
        if near_limit:   state_desc.append('JOINT_LIMIT!')
        if obj_vel > 0.5: state_desc.append(f'OBJ_MOVING({obj_vel:.1f}m/s)')
        if risk > 0.5:   state_desc.append('HIGH_RISK!')
        if progress > 0.3: state_desc.append('PROGRESS!')
        state_str = ' '.join(state_desc) if state_desc else 'normal'

        # print every 10 steps
        if step % 10 == 0:
            print(
                f"{step:>5} | "
                f"{progress:>8.3f} | "
                f"{risk:>6.3f} | "
                f"{efficiency:>10.3f} | "
                f"{pro_score:>9.4f} | "
                f"{state_str}"
            )

        if render:
            env.render()
        if slow:
            time.sleep(0.03)

        # random action
        low, high = env.action_spec
        action = np.random.uniform(low, high)
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    # episode summary
    print("-" * 80)
    print(f"SUMMARY — {perturbation_name}:")
    print(f"  Avg PRO score:  {np.mean(episode_scores):.4f}")
    print(f"  Min PRO score:  {np.min(episode_scores):.4f}")
    print(f"  Max PRO score:  {np.max(episode_scores):.4f}")
    print(f"  Std PRO score:  {np.std(episode_scores):.4f}")

    env.close()
    time.sleep(2)
    return np.mean(episode_scores)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # load encoder and scorer
    print("Loading DINOv2 encoder...")
    encoder = VisualEncoder(device=device)

    print("Loading PRO scorer...")
    scorer = PROScorer(latent_dim=384).to(device)

    # try loading trained weights if available
    import os
    model_path = '/home/mukesh/simcortex/data/pro_model.pt'
    if os.path.exists(model_path):
        scorer.load_state_dict(torch.load(model_path))
        print("Loaded trained PRO weights!")
    else:
        print("Using untrained PRO (train first for better scores)")
    scorer.eval()

    print("\nWatching all failure modes...")
    print("Watch the window + read terminal scores simultaneously!")
    input("\nPress Enter to start...")

    results = {}

    results['normal'] = run_failure_mode(
        'NORMAL (baseline)',
        encoder=encoder, scorer=scorer, device=device
    )
    input("\nPress Enter for next failure mode...")

    results['slippery'] = run_failure_mode(
        'SLIPPERY OBJECTS (friction=0.05)',
        friction=[0.05, 0.005, 0.0001],
        encoder=encoder, scorer=scorer, device=device
    )
    input("\nPress Enter for next failure mode...")

    results['heavy'] = run_failure_mode(
        'HEAVY OBJECTS (mass=5kg)',
        mass=5.0,
        encoder=encoder, scorer=scorer, device=device
    )
    input("\nPress Enter for next failure mode...")

    results['extreme'] = run_failure_mode(
        'EXTREME POSE (joint noise)',
        extreme_pose=True,
        encoder=encoder, scorer=scorer, device=device
    )

    # final comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON — Avg PRO Scores By Failure Mode")
    print("=" * 60)
    for mode, score in results.items():
        bar = '█' * int(score * 30)
        print(f"  {mode:>10}: {score:>7.4f} |{bar}")
    print("\nLower score = PRO thinks trajectory is riskier")
    print("Higher score = PRO thinks trajectory is safer")
