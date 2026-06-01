import numpy as np

def compute_risk(env_sim, obs, perturbation=None):
    """
    Risk signal based on multiple physics indicators.
    More meaningful than raw collision detection.
    
    Risk factors:
    1. Object moving unexpectedly (slipping)
    2. Robot joints near limits
    3. Gripper applying force but no progress
    4. Perturbation-specific risk
    """
    model = env_sim.model
    data  = env_sim.data
    risk  = 0.0

    # Factor 1: joint limit proximity
    # joints near limits = risky configuration
    qpos = data.qpos[:7]
    qlim_low  = model.jnt_range[:7, 0]
    qlim_high = model.jnt_range[:7, 1]
    range_size = qlim_high - qlim_low + 1e-6
    proximity  = np.minimum(
        (qpos - qlim_low) / range_size,
        (qlim_high - qpos) / range_size
    )
    # if any joint within 10% of limit
    if np.any(proximity < 0.1):
        risk += 0.4

    # Factor 2: high joint velocity = erratic motion
    qvel = data.qvel[:7]
    if np.any(np.abs(qvel) > 2.0):
        risk += 0.3

    # Factor 3: object velocity = slipping/falling
    try:
        obj_vel = np.linalg.norm(data.qvel[9:12])
        if obj_vel > 0.5:
            risk += 0.3
    except Exception:
        pass

    # Factor 4: perturbation-specific boost
    if perturbation == 'slippery':
        risk += 0.1
    elif perturbation == 'heavy':
        risk += 0.1
    elif perturbation == 'extreme_pose':
        risk += 0.2

    return float(np.clip(risk, 0.0, 1.0))


def compute_progress(obs, initial_distance):
    """Progress toward goal from observation dict."""
    try:
        eef_pos  = obs['robot0_eef_pos']
        obj_pos  = obs['object-state'][:3]
        distance = float(np.linalg.norm(eef_pos - obj_pos))
        progress = float(np.clip(
            1.0 - distance / (initial_distance + 1e-6),
            0.0, 1.0
        ))
        return progress, distance
    except Exception:
        return 0.0, initial_distance


def compute_efficiency(step, max_steps):
    """Efficiency — steps remaining ratio."""
    return float(np.clip(1.0 - step / max_steps, 0.0, 1.0))
