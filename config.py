"""配置：物理常量、黏附模型、三相课程时间表、PPO 超参。

课程结构与论文 Eq.5-7 对齐：
  Phase 1 (iter <= ITER_FLAT_END)      : theta=0,        p_attach=1.0  学爬行步态
  Phase 2 (ITER_FLAT_END..ITER_TILT_END): theta 0->90,   p_attach=1.0  激活吸附
  Phase 3 (iter > ITER_TILT_END)        : theta=90,      p_attach 1.0->0.85  注入随机脱落
"""
import math

# ---- 物理 ----
DT = 0.02                # 仿真步长 (s)
G = 9.81                 # 重力加速度
FOOT_COUNT = 4           # 磁足数量
V_FOOT_MAX = 0.6         # 足沿墙最大移动速度 (m/s)
WALL_HEIGHT = 3.0        # 可爬墙高 (m)
GAIT_PERIOD = 1.2        # 步态周期 (s)，论文 T=1.2
FOLLOW_GAIN = 10.0       # 身体跟随已吸附足均值的增益 (1/s)

BODY_INIT_Y = 0.5        # 初始身体高度
MIN_HEIGHT = 0.05        # 低于此高度 = 坠落
MAX_STEPS = 1000         # 单 episode 最大步数 (1000*0.02=20s)
FREE_DAMP = 0.5          # 0 足吸附时身体速度阻尼（θ=0 时避免惯性滑行漏步态）
SUPPORT_FEET = 2         # 【新增 v6】支撑所需最小吸附足数：四足需 ≥2 足（对角）才能稳定支撑，
                         #   否则身体沿墙下滑。诊断发现根因：平地步态把"0/1 足"当免费，θ 一倾斜就坠。
# 【改 v21】移除 MIN_SIN：v20 崩塌根因 = 低 θ 时 g_t 弱（2.94）学出"偶尔掉到 <2 足"的 sloppy 步态，
#   高 θ 时 g_t 强（9.81）直接摔死。现 g_t 恒为 G，Phase1 起"<2 足"就快速下滑，逼出稳健支撑。

# ---- 黏附模型 ----
ALIGN_RATE = 2.0         # 接触质量 q 恢复速率（足缓慢移动时 q->1，快速移动时 q 下降）

# ---- 三相课程时间表（按 PPO 迭代计数，数值按 2D 缩比） ----
ITER_FLAT_END = 500      # 【改 v12】150 -> 500：延长 Phase1。v11 证明 TOP_FRAC=0.50 解锁了 success（峰值
                          #   42%），但 succ 在 θ 倾斜后 42%->0% 塌方——根因是 Phase1 太短，策略在 θ=0
                          #   尚未收敛（iter 150 时 succ 仍在上升）就被迫倾角。延长到 500，给足纯平地
                          #   收敛时间，让"登顶步态"先稳固，再进入倾角转移。Phase2 斜率略变陡（1700 iter），
                          #   但换取 3.3x 的 Phase1 收敛余量。
ITER_TILT_END = 2200     # Phase2 结束（theta 到达 90）
ITER_FAIL_END = 4200     # Phase3 中 p_attach 降到 0.85 完成
                         #   【改】3000 -> 4200：拉长 Phase3（学随机失败下的恢复），
                         #   对应论文 Phase3 占全程 ~39% 的比例（原仅 ~25%）
P_ATTACH_MIN = 0.85      # 论文中 p_attach 下限

# ---- 奖励权重（v20：稠密 R_climb 身体速度 + 稀疏 R_attach 重吸原子，两者均不可钻空子）----
#   R_tot = W_CLIMB·v_b + W_ATTACH·r_attach + R_support
# v19 失败诊断：只留 R_attach（重吸更高）太稀疏——"脱附→上移→重吸"整段只有重吸那一刻有回报，
#   策略无法 bootstrap，iter 350 仍在坠底。
# v20 根修：补回稠密的 R_climb=W·v_b（身体真实爬升速度，每步都有信号，且不可钻空子——
#   身体只有足重吸更高时才会上升），与稀疏的 R_attach 互补：R_climb 提供稠密梯度引导，
#   R_attach 在重吸瞬间直接强化爬升原子。噪声修复（v19）保留。
W_PROGRESS = 50.0        # 【改 v28】稠密高度进展奖励：W_PROGRESS * max(0, y_b - best_y)，只奖励"达到新高度"。
                          #   替换 v20 的 W_CLIMB(速度)+W_ATTACH(重吸)。诊断根因：v27 的"弱爬升+0.013m"其实是
                          #   "爆发式蠕动"骗过速度型 stuck 检测+刷速度奖励，净爬升≈0。高度进展单调不可刷
                          #   （同一高度不能重复赚），把回报直接绑到"净爬升"，逼策略真正往上爬。
                          #   全程 2.5m 满爬≈125 分，远大于抱死 -50 / 坠落 ~-100，消除风险厌恶与蠕动退化。
W_ATTACH = 0.0           # 【改 v28】禁用重吸奖励：高度进展已隐含"重吸更高→身体爬升"，r_attach 冗余且可刷。
W_SUPPORT = 1.0          # 坠落惩罚：-W_SUPPORT*max(0, SUPPORT_FEET-n_attached)，掉到 <2 足才扣分
STUCK_STEPS = 250        # 全足吸附且无移动 5s（250 步）判静滞（论文早终止条件）
PEN_STUCK = 50.0         # 【改 v27】卡死终止罚单：episode 因 stuck 结束时一次性 -PEN_STUCK（而非 v26 的每步惩罚）。
                          #   诊断 ckpt_400 确认崩溃根因 = 策略学成"磁铁常开永不脱附"的风险厌恶局部最优
                          #   （坠落 -550 太惨，贴住不动 0 回报成为安全洼地）。每步 W_STUCK 会误伤"先学会吸附"前驱，
                          #   v26 因此连吸附都没学会。这里改成只在真正卡死(250步连续静止)时扣大分，使"贴住不动"
                          #   明确为 -50，逼策略去爬（爬顶 +50）而非抱死。前驱"短暂吸附"不触发，不受罚。

# ---- PPO ----
OBS_DIM = 32             # 【改】22 -> 32：新增 4 维接触质量 q + 8 维逐腿时钟（替换原 2 维单相位）
ACT_DIM = 8
HIDDEN = [128, 64]       # 【改】[64,64] -> [128,64]：逼近论文 [256,128,64] 的容量，适配更多观测
LR = 1e-4
GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
EPOCHS = 10              # 【改 v24】恢复 v21 的 10 轮。v22(4轮)/v23(KL早停) 都误诊：真根因不是 epoch 数，
                          #   而是 ENT_COEF 把 log_std 推到 clamp 上限（std=0.607）→ 动作噪声失控毁掉步态。
                          #   诊断：ckpt_200 的 log_std 还分散(-0.6~-1.4)，ckpt_400 就有 6/8 维卡在 -0.499。
MINIBATCH = 64
ENT_COEF = 0.0           # 【改 v25】std 已固定（ppo.Actor 里 log_std requires_grad=False），熵项对 std 无梯度，
                         #   设 0 消除死代码。v24 教训：0.001 让 std 漂到 0.25（下夹）探索不足，学不会爬。
MAX_GRAD_NORM = 0.5
ROLLOUT_STEPS = 2048     # 每次 rollout 收集的步数
TOTAL_ITERS = 4400       # 【改】3200 -> 4400：为拉长 Phase3（学恢复）相应增加总迭代
SEED = 0

CHECKPOINT_DIR = "checkpoints_v28"


def theta_schedule(iter):
    """重力倾斜角：0(地面) -> pi/2(竖直墙)。论文 Eq.5 结构。"""
    if iter <= ITER_FLAT_END:
        return 0.0
    if iter <= ITER_TILT_END:
        return (math.pi / 2) * (iter - ITER_FLAT_END) / (ITER_TILT_END - ITER_FLAT_END)
    return math.pi / 2


def p_attach_schedule(iter):
    """吸附成功率：1.0 -> 0.85。论文 Eq.7 结构。"""
    if iter <= ITER_TILT_END:
        return 1.0
    span = ITER_FAIL_END - ITER_TILT_END
    frac = min(max(iter - ITER_TILT_END, 0), span) / span
    return 1.0 - (1.0 - P_ATTACH_MIN) * frac
