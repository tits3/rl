"""2D 磁吸附爬壁环境（gym 风格，但不依赖 gym）。

把论文的 4 条件黏附门控 (Eq.1-4) 抽象到 2D：
  1. 接触识别      : 足处于摆动相（未吸附）
  2. 磁激活命令    : a_magnet >= 0.5
  3. 几何对齐      : 接触质量 q（移动越快 q 越低，对应"气隙/部分接触"）
  4. 随机成功      : 吸附成功概率 = p_attach * q
"""
import math
import numpy as np

import config as C


class WallClimbEnv:
    def __init__(self):
        self.foot_count = C.FOOT_COUNT
        self.obs_dim = C.OBS_DIM
        self.act_dim = C.ACT_DIM
        self.iter = 0
        self.theta = 0.0
        self.p_attach = 1.0
        self.rng = np.random.default_rng(C.SEED)
        self.reset()

    # ---- 课程 ----
    def set_curriculum(self, it):
        self.iter = it
        self.theta = C.theta_schedule(it)
        self.p_attach = C.p_attach_schedule(it)

    # ---- 重置 ----
    def reset(self):
        self.y_b = C.BODY_INIT_Y
        self.v_b = 0.0
        # 4 足初始吸附，围绕身体略有高低差，形成爬行起始构型
        self.y = np.full(self.foot_count, C.BODY_INIT_Y, dtype=np.float32)
        for i in range(self.foot_count):
            self.y[i] += 0.08 * (i - (self.foot_count - 1) / 2.0)
        self.anchor = self.y.copy()                             # 【v19】各足最近一次吸附的锚点高度
        self.a = np.ones(self.foot_count, dtype=np.float32)      # 1=吸附
        self.q = np.ones(self.foot_count, dtype=np.float32)      # 接触质量
        self.phase = 0.0
        self.step_count = 0
        self.stuck_steps = 0                                    # 【新增】静滞计数器
        self.prev_action = np.zeros(self.act_dim, dtype=np.float32)
        self.ep_reward = 0.0
        self.start_y = self.y_b
        self.best_y = self.y_b          # 【v28】本 episode 最高身体高度（高度进展奖励基准）
        return self._obs()

    # ---- 观测 ----
    def _obs(self):
        o = []
        o.append(float(np.clip(self.y_b / C.WALL_HEIGHT, 0.0, 1.2)))   # 归一化高度
        o.append(float(np.clip(self.v_b, -2.0, 2.0)))                  # 速度
        for i in range(self.foot_count):
            o.append(float(np.clip(self.y[i] - self.y_b, -1.0, 1.0)))  # 相对足高
            o.append(float(self.a[i]))                                 # 吸附态
        # 【新增】接触质量 q：对应论文 Eq.4 的几何对齐/气隙敏感度。
        # 让策略能"预判"某足即将脱附（q 低 -> 吸附易失败），从而学会提前补救。
        for i in range(self.foot_count):
            o.append(float(self.q[i]))
        o.append(math.sin(self.theta))                                 # 重力方向
        o.append(math.cos(self.theta))
        # 【改】论文 8 维逐腿步态时钟，替换原 2 维单相位：
        #   φ_i = 2πt/T + (π/2)·i（i=0..3），四腿相位各差 1/4 周期，
        #   给策略提供"此刻该动哪条腿"的强归纳偏置（这是原代码缺失的关键）。
        for i in range(self.foot_count):
            phi_i = self.phase + (math.pi / 2.0) * i
            o.append(math.sin(phi_i))
            o.append(math.cos(phi_i))
        o.extend(self.prev_action)                                    # 上一步动作
        return np.array(o, dtype=np.float32)

    # ---- 步进 ----
    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.prev_action = action.copy()
        move = action[0::2]      # 4 足移动命令
        magnet = action[1::2]    # 4 足磁命令
        g_t = C.G   # 【改 v21】坠落速度与 θ 无关（去掉 MIN_SIN 弱下限）：Phase1 起"<2 足"就快速下滑，
                    #   逼策略从平地就练成"永不 <2 足"的稳健支撑。v20 崩塌根因：低 θ 弱下滑学出 sloppy 步态。

        # --- 足动力学 + 黏附门控 ---
        r_attach = 0.0                                          # 【v19】本步重新吸附更高的累计增量
        for i in range(self.foot_count):
            if self.a[i] == 1.0:
                self.q[i] = 1.0
                if magnet[i] < 0.5:            # 磁关 -> 释放
                    self.a[i] = 0.0
                    self.q[i] = 0.5
            else:
                # 摆动相：足沿墙移动
                self.y[i] += move[i] * C.V_FOOT_MAX * C.DT
                # 接触质量：慢移对齐(q->1)，快移劣化(q->0)
                self.q[i] = float(np.clip(
                    self.q[i] + C.ALIGN_RATE * (1.0 - 2.0 * abs(move[i])) * C.DT,
                    0.0, 1.0))
                # 磁命令 ON -> 尝试吸附
                if magnet[i] >= 0.5:
                    if self.rng.random() < self.p_attach * self.q[i]:
                        self.a[i] = 1.0
                        self.q[i] = 1.0
                        # 【v19】重新吸附到比上次锚点更高 → 正回报（爬升原子）
                        r_attach += max(0.0, self.y[i] - self.anchor[i])
                        self.anchor[i] = self.y[i]
                    else:
                        self.a[i] = 0.0
                        self.q[i] = 0.0    # 滑脱，失去接触
            self.y[i] = float(np.clip(self.y[i], 0.0, C.WALL_HEIGHT))

        # --- 身体动力学 ---
        attached = self.a == 1.0
        n_attached = int(attached.sum())
        if n_attached >= C.SUPPORT_FEET:
            # 磁摩擦抵消重力，身体跟随已吸附足均值（爬升由此驱动）
            target = float(self.y[attached].mean())
            self.v_b = C.FOLLOW_GAIN * (target - self.y_b)
        else:
            # 【改 v6】支撑不足（< SUPPORT_FEET=2 足）：沿墙下滑 + 阻尼。
            #   四足需 ≥2 足（对角）稳定支撑；0/1 足时身体下滑。配合 g_t 下限，
            #   使 θ=0 时"掉足"也不再免费，逼出"始终 ≥2 足吸附"的支撑习惯（θ=90 的关键前提）。
            self.v_b = C.FREE_DAMP * self.v_b - g_t * C.DT
        self.v_b = float(np.clip(self.v_b, -1.0, 1.0))   # 限制速度，避免尖峰
        self.y_b += self.v_b * C.DT
        self.y_b = float(np.clip(self.y_b, 0.0, C.WALL_HEIGHT))

        # --- 步态时钟推进 ---
        self.phase += 2.0 * math.pi * C.DT / C.GAIT_PERIOD

        # --- 奖励（v20：稠密身体速度 + 稀疏重吸原子 + 坠落惩罚）---
        #   R_tot = W_CLIMB·v_b + W_ATTACH·r_attach + R_support
        # v19 教训：只留 R_attach 太稀疏，无法 bootstrap。v20 补回稠密 R_climb（身体真实爬升速度，
        #   只有足重吸更高时 v_b 才 >0，不可钻空子），与 R_attach（重吸瞬间强化爬升原子）互补。
        # 【改 v28】高度进展奖励（单调，不可刷）+ 坠落支撑惩罚。
        #   只有身体达到"比本集以往更高"才给正回报；原地蠕动/回落 0 分，直接把回报绑到净爬升。
        r_progress = C.W_PROGRESS * max(0.0, self.y_b - self.best_y)
        r_support = -C.W_SUPPORT * max(0.0, C.SUPPORT_FEET - n_attached)
        self.best_y = max(self.best_y, self.y_b)

        reward = r_progress + r_support

        # --- 终止（v13：无终端奖励，早终止 + "survived" 成功判定）---
        self.step_count += 1
        # 静滞检测（论文早终止）：全足吸附且身体无移动 5s -> 结束。
        # 这是"爬升压力"的来源：贴墙不动会提前终止（回报低），替代 R_TOP 登顶目标。
        if n_attached == 4 and abs(self.v_b) < 0.01:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
        stuck = self.stuck_steps >= C.STUCK_STEPS
        fell = self.y_b < C.MIN_HEIGHT
        timeout = self.step_count >= C.MAX_STEPS
        done = fell or timeout or stuck
        # 【改 v27】卡死终止罚单：真正贴住不动 5s 时一次性重罚，堵住"永不脱附"风险厌恶退化。
        #   只在 stuck（250 步连续静止）触发，前驱"短暂吸附"不受罚（区别于 v26 每步惩罚）。
        if stuck:
            reward += -C.PEN_STUCK
        # 成功 = 撑满整个 episode（未坠落、未静滞）——对应论文"10s 不掉不静止"
        survived = timeout

        self.ep_reward += reward
        info = {
            "n_attached": n_attached,
            "fell": fell,
            "timeout": timeout,
            "survived": survived,
            "stuck": stuck,
            "theta": self.theta,
            "p_attach": self.p_attach,
            "climb": self.y_b - self.start_y,
        }
        return self._obs(), reward, done, info

    # ---- 供渲染/调试读取 ----
    def state_dict(self):
        return {
            "y_b": self.y_b, "v_b": self.v_b,
            "y": self.y.copy(), "a": self.a.copy(), "q": self.q.copy(),
            "theta": self.theta, "p_attach": self.p_attach,
            "phase": self.phase, "step_count": self.step_count,
            "ep_reward": self.ep_reward, "climb": self.y_b - self.start_y,
        }
