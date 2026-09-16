"""入口：python main.py --train | --play [--checkpoint PATH] [--render]"""
import argparse
import os

import pygame

import config as C
from env import WallClimbEnv
from ppo import PPO


def play(checkpoint, seed):
    from render import Renderer

    env = WallClimbEnv()
    env.set_curriculum(C.ITER_FAIL_END)   # theta=90, p_attach=0.85（Phase 3 挑战）
    ppo = PPO(C.OBS_DIM, C.ACT_DIM, seed=seed)
    if checkpoint and os.path.exists(checkpoint):
        ppo.load(checkpoint)
        print(f"loaded {checkpoint}")
    else:
        print("no checkpoint provided -> random policy (won't climb well)")

    renderer = Renderer()
    obs = env.reset()
    running = True
    while running:
        quit_, force, reset = renderer.poll()
        if quit_:
            running = False
            break
        if reset:
            obs = env.reset()
            continue
        if force:
            # 人为让一只足脱落，演示恢复能力
            i = int(env.rng.integers(0, env.foot_count))
            env.a[i] = 0.0
            env.q[i] = 0.0

        act, _, _ = ppo.act(obs, deterministic=True)
        obs, _, done, _ = env.step(act)
        renderer.draw(env)
        if done:
            obs = env.reset()

    pygame.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="训练策略")
    ap.add_argument("--play", action="store_true", help="加载策略并渲染")
    ap.add_argument("--checkpoint", type=str, default=None, help="checkpoint 路径")
    ap.add_argument("--render", action="store_true", help="训练时同步渲染")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    if args.play:
        play(args.checkpoint, args.seed)
    else:
        from train import train
        train(render=args.render)


if __name__ == "__main__":
    main()
