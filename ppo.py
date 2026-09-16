"""从零实现的 PPO：MLP Actor(高斯)/Critic，GAE + clipped surrogate + 熵正则。"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import config as C


def _mlp(sizes):
    layers = []
    for j in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[j], sizes[j + 1]))
        if j < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = _mlp([obs_dim] + hidden + [act_dim])
        # 【v25】固定 std≈0.4，不可学习。v21(ENT_COEF=0.005) 让 std 漂到 0.607 崩步态，v24(0.001) 让 std
        #   漂到 0.25 探索不足学不会。甜蜜点 ~0.37-0.45，直接固定 0.4 消除漂移。
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.9), requires_grad=False)

    def forward(self, obs):
        mean = self.net(obs)
        std = torch.exp(self.log_std)
        return mean, std


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.net = _mlp([obs_dim] + hidden + [1])

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class PPO:
    def __init__(self, obs_dim, act_dim, hidden=None, lr=None, seed=None):
        hidden = hidden or C.HIDDEN
        lr = lr or C.LR
        torch.manual_seed(seed if seed is not None else C.SEED)
        np.random.seed(seed if seed is not None else C.SEED)
        self.actor = Actor(obs_dim, act_dim, hidden)
        self.critic = Critic(obs_dim, hidden)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32)
            mean, std = self.actor(o)
            dist = Normal(mean, std)
            a = mean if deterministic else dist.sample()
            a = torch.clamp(a, -1.0, 1.0)
            logp = dist.log_prob(a).sum(-1)
            val = self.critic(o)
        return a.numpy(), float(logp), float(val)

    def value(self, obs):
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32)
            return float(self.critic(o))

    def update(self, obs_np, act_np, logp_np, ret_np, adv_np):
        obs = torch.as_tensor(obs_np, dtype=torch.float32)
        act = torch.as_tensor(act_np, dtype=torch.float32)
        logp_old = torch.as_tensor(logp_np, dtype=torch.float32)
        ret = torch.as_tensor(ret_np, dtype=torch.float32)
        adv = torch.as_tensor(adv_np, dtype=torch.float32)

        n = obs.shape[0]
        idxs = np.arange(n)
        stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        for _ in range(C.EPOCHS):
            np.random.shuffle(idxs)
            for start in range(0, n, C.MINIBATCH):
                idx = idxs[start:start + C.MINIBATCH]
                o, a, lp_old, rt, ad = obs[idx], act[idx], logp_old[idx], ret[idx], adv[idx]

                mean, std = self.actor(o)
                dist = Normal(mean, std)
                logp = dist.log_prob(a).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                logratio = torch.clamp(logp - lp_old, -2.0, 2.0)
                ratio = torch.exp(logratio)
                surr1 = ratio * ad
                surr2 = torch.clamp(ratio, 1.0 - C.CLIP, 1.0 + C.CLIP) * ad
                actor_loss = -torch.min(surr1, surr2).mean() - C.ENT_COEF * entropy

                val = self.critic(o)
                critic_loss = F.mse_loss(val, rt)

                loss = actor_loss + 0.5 * critic_loss
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    C.MAX_GRAD_NORM,
                )
                self.opt.step()

                stats["actor_loss"] += actor_loss.item()
                stats["critic_loss"] += critic_loss.item()
                stats["entropy"] += entropy.item()

        k = (n // C.MINIBATCH) * C.EPOCHS
        for key in stats:
            stats[key] /= max(1, k)
        return stats

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
