import sys
import os
sys.path.insert(0, '/home/itm/msadasivam/mimic-video/model')
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.5'
os.environ['CUDA_PATH'] = '/usr/local/cuda-12.5'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
from einops import rearrange
import cv2

from cosmos_predict2.configs.config import make_config
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.conditioner import DataType
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import override


class MimicVideoEncoder:
    """
    mimic-video Cosmos-Predict2 visual encoder.
    
    Two-stage pipeline to fit in 6GB VRAM:
    Stage 1: Tokenizer on GPU (0.24GB) -> tokens -> free
    Stage 2: DiT on GPU (3.64GB) -> layer 19 hidden states
    
    Layer 19 hidden states = rich temporal video latent
    Used by PRO scorer to score K candidate trajectories.
    """

    def __init__(
        self,
        device="cuda",
        language="pick up the object and place it in the bin",
        layer_idx=19,
    ):
        self.device = device
        self.language = language
        self.layer_idx = layer_idx
        self.latent_dim = 2048

        print("Loading mimic-video encoder...")
        self.config = make_config()
        self.config = override(self.config, [
            "--",
            "experiment=w2a_bridge_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz128"
        ])
        self.config.model.config.video_pipe_config.guardrail_config.enabled = False
        self.pipe = None  # loaded on demand

        print("mimic-video config ready!")

    def _load_dit(self):
        """Load DiT pipeline on demand."""
        if self.pipe is not None:
            return
        self.pipe = Video2WorldPipeline.from_config(
            config=self.config.model.config.video_pipe_config,
            dit_path="/home/itm/msadasivam/mimic-video/model/checkpoints/video_backbone/v2w_pretrained_cosmos.pt",
            device=self.device,
            torch_dtype=torch.bfloat16,
            load_ema_to_reg=False,
            offload_text_encoder=True,
        )
        for param in self.pipe.dit.parameters():
            param.requires_grad = False
        # pre-compute text embedding once on CPU
        self.text_emb, _ = self.pipe.text_encoder.encode_prompts(
            [self.language], return_mask=True,
        )
        self.text_emb = self.text_emb.to(self.device).bfloat16()
        print(f"DiT ready! VRAM: {round(torch.cuda.memory_allocated()/1024**3,2)}GB")

    def preprocess_frames(self, frames):
        """Convert list of numpy frames to video tensor."""
        processed = []
        for frame in frames:
            frame = cv2.resize(frame, (640, 480))
            frame = 2.0 * (frame.astype(np.float32) / 255.0 - 0.5)
            frame = rearrange(frame, "h w c -> c h w")
            processed.append(frame)
        video = np.stack(processed, axis=1)
        return torch.from_numpy(video).unsqueeze(0).bfloat16().to(self.device)

    def _unload_dit(self):
        """Unload DiT to free VRAM for tokenizer."""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()

    def tokenize_frames(self, frames):
        """
        Stage 1: Load tokenizer, tokenize, free tokenizer.
        Keeps VRAM free for DiT.
        """
        video = self.preprocess_frames(frames)

        tokenizer = instantiate(
            self.config.model.config.video_pipe_config.tokenizer
        )
        tokenizer.to(device=self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            tokens = tokenizer.encode(video).clone()

        del tokenizer
        torch.cuda.empty_cache()
        return tokens

    @torch.no_grad()
    def encode(self, frames_5, seed=None):
        """
        Encode 5 frames to 2048-dim latent vector.
        Stage 1: tokenize (no DiT loaded)
        Stage 2: load DiT, run forward, get latent
        """
        # Stage 1: unload DiT, tokenize, free tokenizer
        self._unload_dit()
        tokens = self.tokenize_frames(frames_5)
        # Stage 2: load DiT, run forward
        self._load_dit()

        B, C, T, H, W = tokens.shape
        if seed is not None:
            torch.manual_seed(seed)
        noise = torch.randn_like(tokens)
        timesteps = torch.ones(B, T, device=self.device).bfloat16()
        padding_mask = torch.ones(B, H, W, device=self.device).bfloat16()

        result = self.pipe.dit(
            x_B_C_T_H_W=noise,
            timesteps_B_T=timesteps,
            crossattn_emb=self.text_emb,
            condition_video_input_mask_B_C_T_H_W=None,
            padding_mask=padding_mask,
            data_type=DataType.IMAGE,
            return_only_hidden_states_up_to=self.layer_idx,
        )

        # result[1] = list of hidden states, [layer_idx] = our latent
        layer_hidden = result[1][self.layer_idx]  # (1, T, H, W, 2048)
        latent = layer_hidden.float().mean(dim=(0, 1, 2, 3))  # (2048,)
        return latent

    def generate_k_candidates(self, frames_5, K=3):
        """
        Generate K different latents from same frames.
        Different seeds = different imagined futures.
        This is the core of Plan->Score->Execute.
        
        Args:
            frames_5: list of 5 numpy arrays
            K: number of candidates
        Returns:
            latents: list of K tensors (2048,)
        """
        latents = []
        for k in range(K):
            latent = self.encode(frames_5, seed=k * 42)
            latents.append(latent)
        return latents


if __name__ == "__main__":
    print("Testing MimicVideoEncoder...")
    encoder = MimicVideoEncoder(device="cuda")

    fake_frames = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(5)
    ]

    print("\nEncoding 5 frames...")
    latent = encoder.encode(fake_frames)
    print(f"Latent shape: {latent.shape}")
    print(f"Latent mean: {latent.mean().item():.4f}")
    print(f"Latent std: {latent.std().item():.4f}")

    print("\nGenerating K=3 candidates...")
    latents = encoder.generate_k_candidates(fake_frames, K=3)
    for k, lat in enumerate(latents):
        print(f"  Candidate {k+1}: norm={lat.norm().item():.3f}")

    diff = (latents[0] - latents[1]).norm().item()
    print(f"\nDiff k1 vs k2: {diff:.4f}")
    print("Different futures! ✅" if diff > 0 else "Same ❌")
    print("\nMimicVideoEncoder ready!")
