import cv2
import numpy as np
import math
import time
import random

import mediapipe as mp

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
WINDOW_W = 1280
WINDOW_H = 720
PINCH_THRESHOLD     = 0.06
PUNCH_VEL_THRESHOLD = 25
MAX_FISH            = 20

# Deep sea color palette (BGR)
SEA_DEEP    = np.array([80, 40, 10],   dtype=np.float32)   # deep navy
SEA_MID     = np.array([120, 80, 20],  dtype=np.float32)
SEA_SHALLOW = np.array([160, 130, 40], dtype=np.float32)
LIGHT_RAYS  = (200, 170, 60)

# ─────────────────────────────────────────────
#  Gradient sea background
# ─────────────────────────────────────────────
def make_background():
    bg = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
    for y in range(WINDOW_H):
        t = y / WINDOW_H
        if t < 0.5:
            c = SEA_SHALLOW * (1 - t*2) + SEA_MID * (t*2)
        else:
            c = SEA_MID * (1 - (t-0.5)*2) + SEA_DEEP * ((t-0.5)*2)
        bg[y, :] = c.astype(np.uint8)
    return bg

BG_BASE = make_background()

# ─────────────────────────────────────────────
#  Light rays (pre-computed angles)
# ─────────────────────────────────────────────
RAYS = [(random.randint(100, WINDOW_W-100), random.uniform(0.3, 0.7), random.uniform(20, 60)) for _ in range(6)]

def draw_light_rays(frame, t):
    overlay = frame.copy()
    for (rx, strength, width) in RAYS:
        sway = int(math.sin(t * 0.4 + rx) * 30)
        pts = np.array([
            [rx + sway - int(width), 0],
            [rx + sway + int(width), 0],
            [rx + int(width*2.5), WINDOW_H],
            [rx - int(width*2.5), WINDOW_H],
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (180, 160, 60))
    cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)

# ─────────────────────────────────────────────
#  Bubbles
# ─────────────────────────────────────────────
class Bubble:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = random.uniform(0, WINDOW_W)
        self.y = random.uniform(WINDOW_H * 0.6, WINDOW_H)
        self.r = random.randint(3, 10)
        self.vy = random.uniform(-1.2, -0.4)
        self.vx = random.uniform(-0.3, 0.3)
        self.alpha = random.uniform(0.3, 0.7)
        self.wobble = random.uniform(0, math.pi * 2)

    def update(self):
        self.wobble += 0.05
        self.x += self.vx + math.sin(self.wobble) * 0.4
        self.y += self.vy
        if self.y < -20:
            self.reset()
            self.y = WINDOW_H + 5

    def draw(self, frame):
        x, y, r = int(self.x), int(self.y), self.r
        if 0 <= x < WINDOW_W and 0 <= y < WINDOW_H:
            cv2.circle(frame, (x, y), r, (220, 210, 160), 1, cv2.LINE_AA)
            # shine
            sx, sy = x - r//3, y - r//3
            if 0 <= sx < WINDOW_W and 0 <= sy < WINDOW_H:
                cv2.circle(frame, (sx, sy), max(1, r//4), (240, 235, 200), -1, cv2.LINE_AA)

BUBBLES = [Bubble() for _ in range(60)]

# ─────────────────────────────────────────────
#  Kelp
# ─────────────────────────────────────────────
class KelpStrand:
    def __init__(self, base_x):
        self.base_x = base_x
        self.segments = random.randint(8, 14)
        self.seg_len  = random.randint(35, 55)
        self.phase    = random.uniform(0, math.pi * 2)
        self.color1   = (random.randint(20, 50), random.randint(100, 160), random.randint(10, 40))
        self.color2   = (random.randint(30, 70), random.randint(140, 200), random.randint(20, 60))
        self.thickness = random.randint(3, 7)

    def draw(self, frame, t):
        x, y = self.base_x, WINDOW_H
        for i in range(self.segments):
            sway = math.sin(t * 0.6 + self.phase + i * 0.4) * (8 + i * 2)
            nx = int(x + sway)
            ny = int(y - self.seg_len)
            depth = i / self.segments
            color = tuple(int(self.color1[c] * (1-depth) + self.color2[c] * depth) for c in range(3))
            cv2.line(frame, (int(x), int(y)), (nx, ny), color, self.thickness, cv2.LINE_AA)
            x, y = nx, ny

KELP = [KelpStrand(x) for x in range(60, WINDOW_W, random.randint(80, 140)) for _ in range(1)]
# spread more naturally
KELP = [KelpStrand(random.randint(30, WINDOW_W-30)) for _ in range(18)]

# ─────────────────────────────────────────────
#  Corals
# ─────────────────────────────────────────────
class Coral:
    def __init__(self):
        self.x = random.randint(40, WINDOW_W - 40)
        self.y = WINDOW_H - random.randint(10, 40)
        self.branches = random.randint(4, 9)
        self.height   = random.randint(40, 100)
        self.color    = random.choice([
            (60, 80, 220),   # red-orange
            (100, 60, 200),  # pink
            (200, 100, 60),  # teal
            (80, 180, 220),  # yellow
            (160, 60, 180),  # violet
        ])
        self.kind = random.choice(['branching', 'fan', 'tube'])
        self.phase = random.uniform(0, math.pi*2)

    def draw(self, frame, t):
        if self.kind == 'branching':
            self._draw_branching(frame, self.x, self.y, -math.pi/2, self.height, 5, t)
        elif self.kind == 'fan':
            self._draw_fan(frame, t)
        else:
            self._draw_tubes(frame, t)

    def _draw_branching(self, frame, x, y, angle, length, depth, t):
        if depth == 0 or length < 6: return
        sway = math.sin(t * 0.5 + self.phase + depth) * 0.08
        angle += sway
        ex = int(x + math.cos(angle) * length)
        ey = int(y + math.sin(angle) * length)
        thick = max(1, depth // 2)
        bright = min(255, int(self.color[0] + depth * 10))
        col = (min(255, self.color[0] + depth*8), min(255, self.color[1] + depth*5), min(255, self.color[2]))
        cv2.line(frame, (int(x), int(y)), (ex, ey), col, thick, cv2.LINE_AA)
        spread = math.pi / (3 + depth * 0.3)
        self._draw_branching(frame, ex, ey, angle - spread, length * 0.7, depth - 1, t)
        self._draw_branching(frame, ex, ey, angle + spread, length * 0.7, depth - 1, t)

    def _draw_fan(self, frame, t):
        sway = math.sin(t * 0.4 + self.phase) * 5
        for i in range(self.branches * 3):
            a = -math.pi + (i / (self.branches * 3)) * math.pi + sway * 0.05
            ex = int(self.x + math.cos(a) * self.height)
            ey = int(self.y + math.sin(a) * self.height)
            cv2.line(frame, (self.x, self.y), (ex, ey), self.color, 1, cv2.LINE_AA)
        cv2.circle(frame, (self.x, self.y), 5, self.color, -1)

    def _draw_tubes(self, frame, t):
        for i in range(self.branches):
            ox = self.x + (i - self.branches//2) * 12
            h  = self.height - abs(i - self.branches//2) * 8
            sway = int(math.sin(t * 0.5 + self.phase + i) * 3)
            cv2.line(frame, (ox, self.y), (ox + sway, self.y - h), self.color, 4, cv2.LINE_AA)
            cv2.circle(frame, (ox + sway, self.y - h), 6, self.color, -1, cv2.LINE_AA)

CORALS = [Coral() for _ in range(22)]

# ─────────────────────────────────────────────
#  Sand floor
# ─────────────────────────────────────────────
SAND_Y = WINDOW_H - 60

def draw_floor(frame):
    # sand gradient
    for dy in range(60):
        t = dy / 60
        c = int(60 + t * 40), int(100 + t * 30), int(120 + t * 20)
        cv2.line(frame, (0, SAND_Y + dy), (WINDOW_W, SAND_Y + dy), c, 1)
    # pebbles
    random.seed(42)
    for _ in range(80):
        px2 = random.randint(0, WINDOW_W)
        py2 = random.randint(SAND_Y, WINDOW_H)
        pr  = random.randint(2, 6)
        pc  = random.randint(80, 130)
        cv2.circle(frame, (px2, py2), pr, (pc, pc+10, pc+20), -1)
    random.seed()

FLOOR_CACHE = BG_BASE.copy()
draw_floor(FLOOR_CACHE)

# ─────────────────────────────────────────────
#  Tropical Fish
# ─────────────────────────────────────────────
FISH_SPECS = [
    # (body_color, stripe_color, fin_color, name)
    ((30, 180, 255),  (255, 255, 255), (0, 120, 255),   "clownfish"),
    ((0, 220, 255),   (0, 80, 180),   (0, 200, 100),   "tang"),
    ((60, 180, 60),   (255, 255, 0),  (30, 200, 80),   "parrotfish"),
    ((200, 100, 255), (255, 200, 0),  (180, 60, 255),  "angelfish"),
    ((0, 160, 255),   (0, 0, 180),    (0, 200, 255),   "damselfish"),
    ((30, 220, 180),  (255, 255, 255),(20, 180, 140),  "moorish idol"),
]

class Fish:
    def __init__(self, x, y):
        spec = random.choice(FISH_SPECS)
        self.body_color  = spec[0]
        self.stripe_color= spec[1]
        self.fin_color   = spec[2]
        self.name        = spec[3]
        self.x = float(x)
        self.y = float(y)
        self.vx = random.uniform(-1.5, 1.5)
        self.vy = random.uniform(-0.5, 0.5)
        self.size = random.randint(22, 38)
        self.phase = random.uniform(0, math.pi * 2)
        self.tail_phase = random.uniform(0, math.pi * 2)
        self.target = None
        self.grabbed = False
        self.facing = 1 if self.vx >= 0 else -1

    def set_target(self, tx, ty):
        self.target = (tx, ty)
        self.grabbed = True

    def release(self):
        self.target = None
        self.grabbed = False

    def update(self, t):
        self.tail_phase += 0.18
        self.phase += 0.02

        if self.grabbed and self.target:
            dx = self.target[0] - self.x
            dy = self.target[1] - self.y
            self.vx += dx * 0.15
            self.vy += dy * 0.15
            self.vx *= 0.75
            self.vy *= 0.75
        else:
            # natural swim
            self.vx += random.uniform(-0.08, 0.08)
            self.vy += random.uniform(-0.04, 0.04)
            self.vy += math.sin(self.phase) * 0.05  # bob
            self.vx = max(-2.5, min(2.5, self.vx))
            self.vy = max(-1.2, min(1.2, self.vy))
            # soft wall bounce
            if self.x < 60:  self.vx += 0.3
            if self.x > WINDOW_W - 60: self.vx -= 0.3
            if self.y < 60:  self.vy += 0.3
            if self.y > SAND_Y - 30: self.vy -= 0.4

        self.x += self.vx
        self.y += self.vy
        self.x = max(20, min(WINDOW_W - 20, self.x))
        self.y = max(20, min(SAND_Y - 20, self.y))

        if abs(self.vx) > 0.1:
            self.facing = 1 if self.vx > 0 else -1

    def draw(self, frame, t):
        x, y = int(self.x), int(self.y)
        s = self.size
        f = self.facing
        tail_wag = int(math.sin(self.tail_phase) * s * 0.5)

        # tail
        tail_pts = np.array([
            [x - f * int(s*0.6), y],
            [x - f * int(s*1.2), y - int(s*0.5) + tail_wag],
            [x - f * int(s*1.2), y + int(s*0.5) + tail_wag],
        ], np.int32)
        cv2.fillPoly(frame, [tail_pts], self.fin_color)

        # body ellipse
        axes = (s, int(s * 0.55))
        cv2.ellipse(frame, (x, y), axes, 0, 0, 360, self.body_color, -1, cv2.LINE_AA)

        # stripe
        stripe_pts = np.array([
            [x + f * int(s*0.1), y - int(s*0.54)],
            [x + f * int(s*0.1), y + int(s*0.54)],
            [x - f * int(s*0.1), y + int(s*0.54)],
            [x - f * int(s*0.1), y - int(s*0.54)],
        ], np.int32)
        cv2.fillPoly(frame, [stripe_pts], self.stripe_color)
        # re-draw body border
        cv2.ellipse(frame, (x, y), axes, 0, 0, 360, self.body_color, 2, cv2.LINE_AA)

        # dorsal fin
        fin_pts = np.array([
            [x, y - int(s*0.55)],
            [x + f * int(s*0.5), y - int(s*0.9)],
            [x - f * int(s*0.2), y - int(s*0.55)],
        ], np.int32)
        cv2.fillPoly(frame, [fin_pts], self.fin_color)

        # eye
        ex = x + f * int(s * 0.45)
        ey = y - int(s * 0.15)
        cv2.circle(frame, (ex, ey), max(2, s//7), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (ex, ey), max(1, s//10), (20, 20, 20), -1, cv2.LINE_AA)

        # shimmer highlight
        hx = x + f * int(s * 0.2)
        hy = y - int(s * 0.25)
        cv2.ellipse(frame, (hx, hy), (max(1, s//5), max(1, s//8)), -30, 0, 360,
                    (min(255, self.body_color[0]+80), min(255, self.body_color[1]+60), min(255, self.body_color[2]+40)),
                    -1, cv2.LINE_AA)

FISHES: list[Fish] = []
grabbed_fish: Fish | None = None

# ─────────────────────────────────────────────
#  Particles (ambient plankton)
# ─────────────────────────────────────────────
NUM_PARTICLES = 400
rng = np.random.default_rng()
pp_x = rng.uniform(0, WINDOW_W, NUM_PARTICLES).astype(np.float32)
pp_y = rng.uniform(0, WINDOW_H * 0.8, NUM_PARTICLES).astype(np.float32)
pp_vx = rng.uniform(-0.3, 0.3, NUM_PARTICLES).astype(np.float32)
pp_vy = rng.uniform(-0.2, 0.1, NUM_PARTICLES).astype(np.float32)
pp_phase = rng.uniform(0, math.pi * 2, NUM_PARTICLES).astype(np.float32)
pp_size = rng.uniform(1, 2.5, NUM_PARTICLES).astype(np.float32)
pp_color = rng.integers(0, len([ (200,230,255),(180,255,220),(255,240,180),(200,200,255) ]), NUM_PARTICLES)
PLANKTON_COLS = [(200,230,255),(180,255,220),(255,240,180),(200,200,255)]

def update_plankton(t):
    global pp_x, pp_y, pp_vx, pp_vy, pp_phase
    pp_phase += 0.02
    pp_vx += np.sin(pp_phase * 0.5) * 0.01
    pp_vy += -0.01  # drift up
    pp_vx *= 0.99
    pp_vy *= 0.99
    pp_x += pp_vx
    pp_y += pp_vy
    # reset if drifted off top
    reset_mask = pp_y < 0
    pp_x[reset_mask] = rng.uniform(0, WINDOW_W, np.sum(reset_mask))
    pp_y[reset_mask] = rng.uniform(WINDOW_H * 0.5, WINDOW_H * 0.9, np.sum(reset_mask))
    pp_x %= WINDOW_W

def draw_plankton(frame, t):
    pulse = (0.5 + 0.5 * np.sin(pp_phase + t)).astype(np.float32)
    for i in range(NUM_PARTICLES):
        x, y = int(pp_x[i]), int(pp_y[i])
        if 0 <= x < WINDOW_W and 0 <= y < WINDOW_H:
            col = PLANKTON_COLS[pp_color[i]]
            a = float(pulse[i]) * 0.6
            cv2.circle(frame, (x, y), int(pp_size[i]), (int(col[0]*a), int(col[1]*a), int(col[2]*a)), -1)

# ─────────────────────────────────────────────
#  Hand Tracker
# ─────────────────────────────────────────────
class HandTracker:
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    def parse(self, results):
        if not results.hand_landmarks: return []
        return [list(hand) for hand in results.hand_landmarks]

    def pinch_point(self, lm):
        d = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
        if d < PINCH_THRESHOLD:
            return ((lm[4].x + lm[8].x) / 2 * WINDOW_W,
                    (lm[4].y + lm[8].y) / 2 * WINDOW_H)
        return None

# ─────────────────────────────────────────────
#  HUD
# ─────────────────────────────────────────────
def draw_hud(frame, fps, fish_count, pinching):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # top bar
    cv2.rectangle(frame, (0, 0), (WINDOW_W, 38), (0, 0, 0), -1)
    cv2.addWeighted(frame, 1, frame, 0, 0, frame)  # noop, bar drawn above
    info = f"FPS {fps:.0f}   FISH {fish_count}/{MAX_FISH}   {'[PINCH ACTIVE]' if pinching else 'PINCH to spawn/grab fish'}"
    cv2.putText(frame, info, (14, 24), font, 0.52, (160, 230, 200), 1, cv2.LINE_AA)

# ─────────────────────────────────────────────
#  Camera overlay blend
# ─────────────────────────────────────────────
def blend_camera(sea_frame, cam_frame):
    """Blend camera feed as a semi-transparent corner inset."""
    ih, iw = 180, 320
    cam_small = cv2.resize(cam_frame, (iw, ih))
    # mirror
    cam_small = cv2.flip(cam_small, 1)
    x0, y0 = WINDOW_W - iw - 14, 44
    roi = sea_frame[y0:y0+ih, x0:x0+iw]
    blended = cv2.addWeighted(cam_small, 0.7, roi, 0.3, 0)
    sea_frame[y0:y0+ih, x0:x0+iw] = blended
    # border
    cv2.rectangle(sea_frame, (x0, y0), (x0+iw, y0+ih), (120, 200, 160), 1)
    cv2.putText(sea_frame, "YOU", (x0+6, y0+ih-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120,200,160), 1)

# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    global grabbed_fish

    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)

    frame_count = 0
    fps = 0.0
    t_prev = time.time()
    prev_pinch_state = [False, False]  # per hand
    pinch_spawn_cooldown = 0

    print("🐠 Deep Sea Interface | PINCH to spawn & grab fish | Q to quit")

    while True:
        ret, cam_frame = cap.read()
        if not ret: break

        t = time.time()

        # ── sea frame ──
        frame = FLOOR_CACHE.copy()
        draw_light_rays(frame, t)

        # kelp (behind corals)
        for k in KELP:
            k.draw(frame, t)

        # corals
        for c in CORALS:
            c.draw(frame, t)

        # plankton
        update_plankton(t)
        draw_plankton(frame, t)

        # bubbles
        for b in BUBBLES:
            b.update()
            b.draw(frame)

        # ── hand tracking ──
        results = tracker.process(cam_frame)
        hands = tracker.parse(results)

        pinching_any = False
        new_grabbed = None
        pinch_positions = []

        for hi, lm in enumerate(hands):
            pp = tracker.pinch_point(lm)
            is_pinching = pp is not None
            was_pinching = prev_pinch_state[hi] if hi < len(prev_pinch_state) else False

            if is_pinching:
                pinching_any = True
                pinch_positions.append(pp)
                px_sc, py_sc = int(pp[0]), int(pp[1])

                # draw pinch indicator
                cv2.circle(frame, (px_sc, py_sc), 18, (0, 255, 200), 2, cv2.LINE_AA)
                cv2.circle(frame, (px_sc, py_sc), 4,  (0, 255, 200), -1, cv2.LINE_AA)

                # grab nearest fish if just pinched
                if not was_pinching:
                    # spawn new fish if none nearby and under limit
                    nearest = None
                    nearest_d = 80
                    for fish in FISHES:
                        d = math.hypot(fish.x - pp[0], fish.y - pp[1])
                        if d < nearest_d:
                            nearest_d = d
                            nearest = fish

                    if nearest:
                        if grabbed_fish and grabbed_fish is not nearest:
                            grabbed_fish.release()
                        grabbed_fish = nearest
                        grabbed_fish.set_target(*pp)
                    elif len(FISHES) < MAX_FISH and pinch_spawn_cooldown <= 0:
                        FISHES.append(Fish(pp[0], pp[1]))
                        pinch_spawn_cooldown = 20

                elif grabbed_fish and is_pinching:
                    grabbed_fish.set_target(*pp)

            else:
                if was_pinching and grabbed_fish:
                    grabbed_fish.release()
                    grabbed_fish = None

            prev_pinch_state = [is_pinching if hi == 0 else (prev_pinch_state[0] if len(prev_pinch_state) > 0 else False),
                                 is_pinching if hi == 1 else (prev_pinch_state[1] if len(prev_pinch_state) > 1 else False)]

        if pinch_spawn_cooldown > 0:
            pinch_spawn_cooldown -= 1

        # ── update + draw fish ──
        for fish in FISHES:
            fish.update(t)

        # sort by y for depth
        for fish in sorted(FISHES, key=lambda f: f.y):
            fish.draw(frame, t)

        # ── camera inset ──
        blend_camera(frame, cam_frame)

        draw_hud(frame, fps, len(FISHES), pinching_any)

        cv2.imshow("DEEP SEA", frame)

        if cv2.getWindowProperty("DEEP SEA", cv2.WND_PROP_VISIBLE) < 1:
            break
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

        frame_count += 1
        if frame_count % 30 == 0:
            now = time.time()
            fps = 30.0 / max(now - t_prev, 0.001)
            t_prev = now

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
