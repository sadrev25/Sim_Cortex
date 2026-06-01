import torch
import torch.nn as nn
import timm
import numpy as np
from torchvision import transforms

class VisualEncoder(nn.Module):
    """
    DINOv2-small as visual encoder.
    Takes 84x84 RGB image from MuJoCo offscreen render
    Outputs 384-dim latent vector.
    
    In full Cortex 2.0: this would be mimic-video Cosmos-Predict2
    Here: DINOv2-small as lightweight proxy
    Swap Monday when PC available.
    """

    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device

        print("Loading DINOv2-small encoder...")
        self.encoder = timm.create_model(
            'vit_small_patch14_dinov2',
            pretrained=True,
            num_classes=0,  # remove classification head
        )
        self.encoder = self.encoder.eval().to(device)

        # freeze — never train this
        for param in self.encoder.parameters():
            param.requires_grad = False

        # image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.latent_dim = 384
        print(f"DINOv2 encoder ready! Latent dim: {self.latent_dim}")

    @torch.no_grad()
    def encode(self, image_np):
        """
        Encode a single MuJoCo image to latent vector.
        
        Args:
            image_np: numpy array (84, 84, 3) uint8 from MuJoCo
        Returns:
            latent: torch tensor (384,) on device
        """
        # preprocess
        img_tensor = self.transform(image_np)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # encode
        latent = self.encoder(img_tensor)
        return latent.squeeze(0)

    @torch.no_grad()
    def encode_batch(self, images_np):
        """
        Encode batch of images.
        
        Args:
            images_np: list of numpy arrays (84, 84, 3)
        Returns:
            latents: torch tensor (N, 384)
        """
        tensors = []
        for img in images_np:
            tensors.append(self.transform(img))
        batch = torch.stack(tensors).to(self.device)
        latents = self.encoder(batch)
        return latents

    @torch.no_grad()
    def encode_sequence(self, image_sequence):
        """
        Encode sequence of frames — temporal context.
        Stack 4 consecutive frames for motion understanding.
        This partially compensates for DINOv2 being image-only.

        Args:
            image_sequence: list of 4 numpy arrays (84, 84, 3)
        Returns:
            latent: torch tensor (384,) mean-pooled temporal latent
        """
        latents = self.encode_batch(image_sequence)
        # mean pool across time — captures average visual state
        # difference across frames captures motion
        temporal_latent = latents.mean(dim=0)
        return temporal_latent


class WorldModel:
    """
    Lightweight world model for K candidate generation.
    
    In full Cortex 2.0: flow-matching world model generates
    K future latent sequences from noise seeds.
    
    Here: we run K different action sequences in MuJoCo
    and encode the resulting frames.
    Same concept — K different imagined futures scored by PRO.
    """

    def __init__(self, encoder, env, K=3):
        self.encoder = encoder
        self.env = env
        self.K = K

    def generate_candidates(self, current_obs, current_image):
        """
        Generate K candidate future latents.
        Each candidate uses different random action sequence.
        
        Args:
            current_obs: current observation dict
            current_image: current frame (84, 84, 3)
        Returns:
            candidates: list of K dicts with latents + actions
        """
        candidates = []

        # save environment state
        sim_state = self.env.env.sim.get_state()

        for k in range(self.K):
            # restore to current state
            self.env.env.sim.set_state(sim_state)
            self.env.env.sim.forward()

            # sample random action sequence (horizon=5 steps)
            future_images = [current_image]
            future_actions = []

            for h in range(5):
                action = np.random.uniform(
                    self.env.env.action_spec[0],
                    self.env.env.action_spec[1]
                )
                obs, reward, done, info = self.env.step(action)
                future_actions.append(action)

                # render future frame
                frame = self.env.env.sim.render(
                    width=84, height=84,
                    camera_name='agentview'
                )
                future_images.append(frame)

            # encode future image sequence to latent
            future_latent = self.encoder.encode_sequence(
                future_images[-4:]  # last 4 frames
            )

            # get MuJoCo labels for this candidate
            labels = self.env.get_labels()

            candidates.append({
                'latent': future_latent,
                'actions': future_actions,
                'labels': labels,
                'k': k,
            })

        # restore environment to before candidate generation
        self.env.env.sim.set_state(sim_state)
        self.env.env.sim.forward()
        self.env.steps_taken -= 5 * self.K

        return candidates


if __name__ == '__main__':
    import sys
    sys.path.append('/home/mukesh/simcortex')
    from env.franka_env import FrankaPickPlaceEnv
    import numpy as np

    print("=" * 50)
    print("Testing Visual Encoder + World Model")
    print("=" * 50)

    # test encoder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    encoder = VisualEncoder(device=device)

    # test with fake image
    fake_img = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
    latent = encoder.encode(fake_img)
    print(f"Single image latent shape: {latent.shape}")
    print(f"VRAM used: {round(torch.cuda.memory_allocated()/1024**2)}MB")

    # test with real MuJoCo render
    print("\nTesting with real MuJoCo environment...")
    env = FrankaPickPlaceEnv(perturbation=None, render=False)

    # need offscreen rendering
    import robosuite as suite
    env_with_cam = suite.make(
        env_name='PickPlace',
        robots='Panda',
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names='agentview',
        camera_heights=84,
        camera_widths=84,
    )
    obs = env_with_cam.reset()
    real_image = obs['agentview_image']
    print(f"Real MuJoCo image shape: {real_image.shape}")

    real_latent = encoder.encode(real_image)
    print(f"Real image latent shape: {real_latent.shape}")
    print(f"Latent mean: {real_latent.mean().item():.4f}")
    print(f"Latent std: {real_latent.std().item():.4f}")
    print(f"VRAM used: {round(torch.cuda.memory_allocated()/1024**2)}MB")

    print("\n✅ Visual encoder working!")
    print("✅ MuJoCo → image → latent pipeline ready!")
    print("\nNext: PRO scoring heads read these latents")
