"""训练主循环：rollout -> GAE -> PPO 更新 -> 按迭代推进课程。"""
import math
import os

import numpy as np

import config as C
from env import WallClimbEnv
from ppo import PPO


def compute_gae(rew, val, done, last_val):
    """广义优势估计 (GAE)，done 掩码处理 episode 边界。"""
    T = len(rew)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    next_val = last_val
    for t in reversed(range(T)):
        nonterminal = 1.0 - done[t]
        delta = rew[t] + C.GAMMA * next_val * nonterminal - val[t]
        lastgaelam = delta + C.GAMMA * C.LAMBDA * nonterminal * lastgaelam
        adv[t] = lastgaelam
        next_val = val[t]
    return adv, adv + val


def train(render=False, save_dir=C.CHECKPOINT_DIR, total_iters=C.TOTAL_ITERS):
    env = WallClimbEnv()
    ppo = PPO(C.OBS_DIM, C.ACT_DIM)
    os.makedirs(save_dir, exist_ok=True)

    renderer = None
    if render:
        from render import Renderer
        renderer = Renderer()

    obs = env.reset()
    ep_returns, ep_climbs, ep_success = [], [], []

    for it in range(total_iters):
        env.set_curriculum(it)

        obs_buf, act_buf, rew_buf, done_buf, logp_buf, val_buf = [], [], [], [], [], []
        for _ in range(C.ROLLOUT_STEPS):
            act, logp, val = ppo.act(obs)
            next_obs, rew, done, info = env.step(act)
            obs_buf.append(obs); act_buf.append(act); rew_buf.append(rew)
            done_buf.append(float(done)); logp_buf.append(logp); val_buf.append(val)
            if renderer is not None:
                renderer.draw(env)
            obs = next_obs
            if done:
                ep_returns.append(env.ep_reward)
                ep_climbs.append(info["climb"])
                ep_success.append(1.0 if info["survived"] else 0.0)
                obs = env.reset()

        last_val = ppo.value(obs)
        obs_np = np.array(obs_buf, dtype=np.float32)
        act_np = np.array(act_buf, dtype=np.float32)
        rew_np = np.array(rew_buf, dtype=np.float32)
        done_np = np.array(done_buf, dtype=np.float32)
        logp_np = np.array(logp_buf, dtype=np.float32)
        val_np = np.array(val_buf, dtype=np.float32)

        adv, ret = compute_gae(rew_np, val_np, done_np, last_val)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = np.clip(adv, -10.0, 10.0)   # 防优势爆炸（回报退化 -> std~0 时除出巨值）

        stats = ppo.update(obs_np, act_np, logp_np, ret, adv)

        # 最近 50 个 episode 的滑动平均
        n = min(50, len(ep_returns)) if ep_returns else 0
        avg_ret = float(np.mean(ep_returns[-n:])) if n else 0.0
        avg_climb = float(np.mean(ep_climbs[-n:])) if n else 0.0
        succ = float(np.mean(ep_success[-n:])) if n else 0.0

        if it % 10 == 0 or it == total_iters - 1:
            print(
                f"iter {it:5d} | theta {math.degrees(env.theta):5.1f} | "
                f"p_attach {env.p_attach:.3f} | ret {avg_ret:7.2f} | "
                f"climb {avg_climb:5.3f}m | surv {succ:6.1%} | "
                f"a_loss {stats['actor_loss']:.3f} c_loss {stats['critic_loss']:.3f}"
            )

        if it % 200 == 0 and it > 0:
            ppo.save(os.path.join(save_dir, f"ckpt_{it}.pt"))

    ppo.save(os.path.join(save_dir, "final.pt"))
    print(f"training finished, final checkpoint -> {os.path.join(save_dir, 'final.pt')}")
