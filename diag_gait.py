"""诊断：脚本化「谨慎步态」在 theta=90 时能否爬升。验证环境动力学是否支持竖直爬墙。"""
import numpy as np
import config as C
from env import WallClimbEnv


def run_scripted(theta_deg, move_speed=0.3, half_seconds=0.5):
    env = WallClimbEnv()
    # 手动设 theta（弧度），p_attach=1.0
    env.theta = np.radians(theta_deg)
    env.p_attach = 1.0
    env.reset()

    half = max(1, int(half_seconds / C.DT))
    ys = []
    for step in range(C.MAX_STEPS):
        swing = (step // half) % 2  # 0 -> pair A (feet 0,1) 摆动, 1 -> pair B (feet 2,3)
        action = np.zeros(8, dtype=np.float32)
        for i in range(4):
            pair = 0 if i < 2 else 1
            if pair == swing:
                action[2 * i] = move_speed      # 缓慢上移（保 q 高）
                action[2 * i + 1] = 0.0         # 磁关 -> 摆动相
            else:
                action[2 * i] = 0.0             # 不动
                action[2 * i + 1] = 1.0         # 磁开 -> 保持/尝试吸附
        obs, rew, done, info = env.step(action)
        ys.append(env.y_b)
        if done:
            break

    ymax = max(ys)
    print(f"theta={theta_deg:3d}  move={move_speed:.1f}  half={half_seconds:.1f}s | "
          f"steps={len(ys):4d}  y_start={ys[0]:.3f}  y_max={ymax:.3f}  y_end={ys[-1]:.3f}  "
          f"climb={ymax - ys[0]:+.3f}m  survived={info.get('survived', False)}")


if __name__ == "__main__":
    for th in [0, 30, 60, 75, 90]:
        for mv in [0.2, 0.5]:
            run_scripted(th, move_speed=mv)
