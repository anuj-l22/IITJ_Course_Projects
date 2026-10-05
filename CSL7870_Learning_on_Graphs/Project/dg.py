# algos/dg.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def _rbf_mmd(x, y, sigma=None, eps=1e-8):
    # x,y: 1D tensors of per-sample risks
    x = x.view(-1, 1)
    y = y.view(-1, 1)
    z = torch.cat([x, y], 0)
    with torch.no_grad():
        if sigma is None:
            pdist = torch.pdist(z, p=2)
            sigma = torch.median(pdist) + eps if pdist.numel() > 0 else torch.tensor(1.0, device=z.device)
    gamma = 1.0 / (2 * sigma**2 + eps)
    Kxx = torch.exp(-gamma * (x - x.t()).pow(2))
    Kyy = torch.exp(-gamma * (y - y.t()).pow(2))
    Kxy = torch.exp(-gamma * (x - y.t()).pow(2))
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

class RDM:
    """Risk Distribution Matching (approx variant): worst-domain vs aggregate with MMD."""
    def __init__(self, weight=1.0):
        self.weight = weight

    def loss(self, batch, per_sample_loss, env_attr='env_id', approx=True):
        if not hasattr(batch, env_attr):
            return per_sample_loss.new_tensor(0.0), {'rdm_mmd': 0.0, 'rdm_envs': 0}
        env = getattr(batch, env_attr)
        envs = torch.unique(env)
        losses_by_env = [per_sample_loss[env == e] for e in envs]
        if len(losses_by_env) < 2:
            return per_sample_loss.new_tensor(0.0), {'rdm_mmd': 0.0, 'rdm_envs': len(losses_by_env)}
        means = [l.mean() for l in losses_by_env]
        worst_idx = int(torch.argmax(torch.stack(means)))
        agg = torch.cat(losses_by_env, dim=0)
        if approx:
            mmd = _rbf_mmd(losses_by_env[worst_idx], agg)
        else:
            mmd = per_sample_loss.new_tensor(0.0)
            for i in range(len(losses_by_env)):
                for j in range(i + 1, len(losses_by_env)):
                    mmd = mmd + _rbf_mmd(losses_by_env[i], losses_by_env[j])
            mmd = mmd / (len(losses_by_env) * (len(losses_by_env) - 1) / 2.0)
        return self.weight * mmd, {'rdm_mmd': float(mmd.detach().cpu()), 'rdm_envs': len(losses_by_env)}

class URMDiscriminator(nn.Module):
    """Adversarial discriminator for URM: distinguish uniform noise vs. encoder features."""
    def __init__(self, feat_dim=300, hidden=None):
        super().__init__()
        hidden = hidden or 2 * feat_dim
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, z):  # z: [B,D]
        return self.net(z).squeeze(-1)

class URMHook:
    """
    URM via adversarial distribution matching (features -> Uniform[-1,1]^D).
    Encoder tries to fool D; D distinguishes uniform noise (real=1) vs features (fake=0).
    """
    def __init__(self, feat_dim, device, d_lr=1e-4, d_steps=1):
        self.D = URMDiscriminator(feat_dim).to(device)
        self.optD = torch.optim.Adam(self.D.parameters(), lr=d_lr, betas=(0.5, 0.999))
        self.device = device
        self.d_steps = d_steps

    def step_D(self, z_batch):
        B, D = z_batch.shape
        ones = torch.ones(B, device=self.device)
        zeros = torch.zeros(B, device=self.device)
        u = torch.empty(B, D, device=self.device).uniform_(-1.0, 1.0)  # target uniform
        logits_u = self.D(u)
        logits_z = self.D(z_batch.detach())
        loss_D = F.binary_cross_entropy_with_logits(logits_u, ones) \
               + F.binary_cross_entropy_with_logits(logits_z, zeros)
        self.optD.zero_grad()
        loss_D.backward()
        self.optD.step()
        with torch.no_grad():
            acc_u = (torch.sigmoid(logits_u) > 0.5).float().mean()
            acc_z = (torch.sigmoid(logits_z) < 0.5).float().mean()
        return loss_D.detach(), float(acc_u.cpu()), float(acc_z.cpu())

    def gen_loss(self, z_batch, weight=1.0):
        B = z_batch.size(0)
        ones = torch.ones(B, device=self.device)
        logits_z = self.D(z_batch)
        loss_G = F.binary_cross_entropy_with_logits(logits_z, ones)
        return weight * loss_G, {'urm_g_loss': float(loss_G.detach().cpu())}
