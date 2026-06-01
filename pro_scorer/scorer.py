import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/mukesh/simcortex')
from pro_scorer.progress_head import ProgressHead
from pro_scorer.risk_head import RiskHead
from pro_scorer.efficiency_head import EfficiencyHead

class PROScorer(nn.Module):
    def __init__(self, latent_dim=384, lambda_risk=0.5, beta_efficiency=0.3):
        super().__init__()
        self.progress_head   = ProgressHead(latent_dim)
        self.risk_head       = RiskHead(latent_dim)
        self.efficiency_head = EfficiencyHead(latent_dim)
        self.lambda_risk     = lambda_risk
        self.beta_efficiency = beta_efficiency

    def forward(self, latent):
        progress   = self.progress_head(latent)
        risk       = self.risk_head(latent)
        efficiency = self.efficiency_head(latent)
        score = progress - self.lambda_risk * risk + self.beta_efficiency * efficiency
        return {
            'progress':   progress,
            'risk':       risk,
            'efficiency': efficiency,
            'score':      score,
        }

    def score_candidates(self, candidate_latents):
        scores = []
        for latent in candidate_latents:
            result = self.forward(latent)
            scores.append(result['score'])
        scores_tensor = torch.stack(scores)
        best_idx      = scores_tensor.argmax().item()
        best_score    = scores_tensor[best_idx]
        advantage     = best_score - scores_tensor.mean()
        return {
            'best_idx':   best_idx,
            'best_score': best_score.item(),
            'advantage':  advantage.item(),
            'all_scores': scores_tensor.tolist(),
        }

if __name__ == '__main__':
    print("Testing PRO Scorer...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scorer = PROScorer(latent_dim=384).to(device)
    K = 3
    fake_latents = [torch.randn(384).to(device) for _ in range(K)]
    result = scorer.score_candidates(fake_latents)
    print(f"K={K} candidates scored!")
    print(f"All scores: {[f'{s:.4f}' for s in result['all_scores']]}")
    print(f"Best candidate: K={result['best_idx']}")
    print(f"Best score: {result['best_score']:.4f}")
    print(f"Advantage: {result['advantage']:.4f}")
    total_params = sum(p.numel() for p in scorer.parameters())
    print(f"Total PRO parameters: {total_params:,}")
    print(f"Model size: {total_params*4/1024/1024:.2f} MB")
    print("PRO Scorer ready!")
