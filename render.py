"""pygame 渲染：墙、身体、4 足(绿=吸附/红=未吸附)、重力箭头、读数。"""
import math

import pygame

import config as C


class Renderer:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Wall Climbing RL (magnetic adhesion)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self.width, self.height = width, height
        self.wall_x = 80
        self.wall_w = 40
        self.body_x = 260          # 身体中心 x（对应离墙约 0.5m 的视觉偏移）
        self.margin = 50
        self.scale = (height - 2 * self.margin) / C.WALL_HEIGHT   # px per meter

    def _sy(self, y):
        """world y (0=底, 向上) -> screen y (向下)。"""
        return self.height - self.margin - float(y) * self.scale

    def poll(self):
        quit_ = force = reset = False
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                quit_ = True
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    quit_ = True
                elif e.key == pygame.K_f:
                    force = True
                elif e.key == pygame.K_r:
                    reset = True
        return quit_, force, reset

    def draw(self, env):
        s = env.state_dict()
        self.screen.fill((18, 20, 28))

        # 墙
        pygame.draw.rect(
            self.screen, (105, 108, 118),
            (self.wall_x, self.margin, self.wall_w, self.height - 2 * self.margin),
        )
        # 高度刻度
        for y in range(0, int(C.WALL_HEIGHT) + 1):
            sy = self._sy(y)
            pygame.draw.line(self.screen, (70, 72, 80),
                             (self.wall_x, sy), (self.wall_x + self.wall_w, sy), 1)
            lbl = self.font.render(f"{y}m", True, (150, 150, 150))
            self.screen.blit(lbl, (self.wall_x - 38, sy - 8))

        # 身体
        by = self._sy(s["y_b"])
        body = pygame.Rect(self.body_x - 20, by - 30, 40, 60)
        pygame.draw.rect(self.screen, (210, 185, 70), body, border_radius=8)

        # 重力箭头（从身体中心）：法向分量 -cosθ(左)，切向分量 +sinθ(下)
        gx, gy = -math.cos(s["theta"]), math.sin(s["theta"])
        cx, cy = self.body_x, by
        L = 70
        ex, ey = cx + gx * L, cy + gy * L
        pygame.draw.line(self.screen, (255, 120, 110), (cx, cy), (ex, ey), 4)
        # 箭头头部
        if abs(gx) + abs(gy) > 1e-6:
            ux, uy = gx, gy
            px, py = -uy, ux
            pygame.draw.polygon(self.screen, (255, 120, 110), [
                (ex, ey),
                (ex - ux * 10 + px * 5, ey - uy * 10 + py * 5),
                (ex - ux * 10 - px * 5, ey - uy * 10 - py * 5),
            ])

        # 足 + 腿
        for i in range(env.foot_count):
            fy = self._sy(s["y"][i])
            attached = s["a"][i] == 1
            color = (90, 220, 90) if attached else (230, 70, 70)
            fx = self.wall_x + self.wall_w + 8
            pygame.draw.line(self.screen, (150, 150, 150),
                             (self.body_x - 12, by), (fx, fy), 3)
            pygame.draw.circle(self.screen, color, (fx, fy), 11)
            pygame.draw.circle(self.screen, (30, 30, 30), (fx, fy), 11, 2)

        # 读数
        lines = [
            f"theta   : {math.degrees(s['theta']):5.1f} deg",
            f"p_attach: {s['p_attach']:.3f}",
            f"climb   : {s['climb']:+.3f} m",
            f"attached: {int(s['a'].sum())}/{env.foot_count}",
            f"ep_reward: {s['ep_reward']:+.1f}",
            f"step    : {s['step_count']}",
        ]
        for i, txt in enumerate(lines):
            self.screen.blit(self.font.render(txt, True, (230, 230, 230)),
                             (self.wall_x + self.wall_w + 60, self.margin + i * 22))

        pygame.display.flip()
        self.clock.tick(60)
