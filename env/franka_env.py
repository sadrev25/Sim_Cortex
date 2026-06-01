import numpy as np
import robosuite as suite
import mujoco

class FrankaPickPlaceEnv:
    def __init__(self, perturbation=None, render=False):
        self.perturbation = perturbation
        self.env = suite.make(
            env_name='PickPlace',
            robots='Panda',
            has_renderer=render,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
            horizon=200,
        )
        self.model = self.env.sim.model
        self.data = self.env.sim.data
        self.initial_distance = None
        self.steps_taken = 0
        self.max_steps = 200

    def reset(self):
        obs = self.env.reset()
        self.steps_taken = 0
        if self.perturbation:
            self._apply_perturbation()
        self.initial_distance = self._get_distance_to_goal()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.steps_taken += 1
        return obs, reward, done, info

    def get_labels(self):
        distance = self._get_distance_to_goal()
        in_collision = self._check_collision()
        progress = float(np.clip(1.0 - (distance / (self.initial_distance + 1e-6)), 0.0, 1.0))
        risk = 1.0 if in_collision else 0.0
        efficiency = float(np.clip(1.0 - (self.steps_taken / self.max_steps), 0.0, 1.0))
        return {
            'progress': progress,
            'risk': risk,
            'efficiency': efficiency,
            'distance': distance,
            'steps': self.steps_taken,
        }

    def get_observation_vector(self):
        obs = self.env._get_observations()
        return np.concatenate([
            obs['robot0_eef_pos'],
            obs['robot0_eef_quat'],
            obs['robot0_gripper_qpos'],
            obs['robot0_joint_pos'],
            obs.get('object-state', np.zeros(14)),
        ])

    def _apply_perturbation(self):
        if self.perturbation == 'slippery':
            for i in range(self.model.ngeom):
                name = self.model.geom_id2name(i) or ''
                if 'object' in name or 'cube' in name:
                    self.model.geom_friction[i] = [0.05, 0.005, 0.0001]
        elif self.perturbation == 'heavy':
            for i in range(self.model.nbody):
                name = self.model.body_id2name(i) or ''
                if 'object' in name or 'cube' in name:
                    self.model.body_mass[i] = 5.0
        elif self.perturbation == 'extreme_pose':
            noise = np.random.uniform(-0.3, 0.3, 7)
            self.data.qpos[:7] += noise
            mujoco.mj_forward(self.model._model, self.data._data)

    def _get_distance_to_goal(self):
        try:
            obs = self.env._get_observations()
            eef_pos = obs['robot0_eef_pos']
            obj_pos = obs['object-state'][:3]
            return float(np.linalg.norm(eef_pos - obj_pos))
        except Exception:
            return 1.0

    def _check_collision(self):
        return self.data.ncon > 0

    def close(self):
        self.env.close()


if __name__ == '__main__':
    print("Testing normal environment...")
    env = FrankaPickPlaceEnv(perturbation=None)
    obs = env.reset()
    print(f"  Initial distance to goal: {env.initial_distance:.3f}")
    for step in range(5):
        action = np.random.uniform(env.env.action_spec[0], env.env.action_spec[1])
        obs, reward, done, info = env.step(action)
        labels = env.get_labels()
        print(f"  Step {step+1}: progress={labels['progress']:.3f} risk={labels['risk']:.0f} efficiency={labels['efficiency']:.3f}")
    env.close()

    print("\nTesting slippery perturbation...")
    env = FrankaPickPlaceEnv(perturbation='slippery')
    obs = env.reset()
    print(f"  Slippery env ready, distance: {env.initial_distance:.3f}")
    env.close()

    print("\nAll environment tests passed!")
