import cv2
import numpy as np
import math
import time
import sys

import mediapipe as mp

# ─────────────────────────────────────────────
#  Config - TUNED FOR PERFORMANCE
# ─────────────────────────────────────────────
NUM_PARTICLES   = 1500          
WINDOW_W        = 1280
WINDOW_H        = 720
TRAIL_ALPHA     = 0.20          
FLOCK_STRENGTH  = 0.1
SCATTER_FORCE   = 15.0
MAX_SPEED_FAR   = 7.0
MAX_SPEED_NEAR  = 3.0
NEAR_THRESHOLD  = 60

NEON_COLORS_BGR = [
    (204, 255,   0),   # Cyan-Green
    (255,   0, 255),   # Magenta
    (255, 180,   0),   # Sky-Blue
    (  0,  80, 255),   # Orange-Red
    (255,   0, 180),   # Violet
]

FINGERTIP_IDS = [4, 8, 12, 16, 20]

# ─────────────────────────────────────────────
#  Particle System (NumPy Vectorized)
# ─────────────────────────────────────────────
rng = np.random.default_rng()

px = rng.uniform(0, WINDOW_W, NUM_PARTICLES).astype(np.float32)
py = rng.uniform(0, WINDOW_H, NUM_PARTICLES).astype(np.float32)
vx = rng.uniform(-1.0, 1.0, NUM_PARTICLES).astype(np.float32)
vy = rng.uniform(-1.0, 1.0, NUM_PARTICLES).astype(np.float32)
sizes = rng.uniform(1, 3, NUM_PARTICLES).astype(np.float32)
color_idx = (np.arange(NUM_PARTICLES) % len(NEON_COLORS_BGR)).astype(np.int32)
phase = rng.uniform(0, 2 * math.pi, NUM_PARTICLES).astype(np.float32)

def update_particles(fingertips_xy, is_fist):
    global px, py, vx, vy, phase
    phase += 0.03

    if not fingertips_xy or is_fist:
        if is_fist and fingertips_xy:
            cx, cy = np.mean(fingertips_xy, axis=0)
            dx, dy = px - cx, py - cy
            dist = np.sqrt(dx*dx + dy*dy) + 1e-6
            force = np.minimum(SCATTER_FORCE * 2000.0 / (dist*dist), 10.0)
            vx += (dx/dist) * force
            vy += (dy/dist) * force

        vx *= 0.95
        vy *= 0.95
        px += vx
        py += vy
        px %= WINDOW_W
        py %= WINDOW_H
        return

    for ci, (tx, ty) in enumerate(fingertips_xy):
        mask = (color_idx % len(fingertips_xy)) == ci
        dx, dy = tx - px[mask], ty - py[mask]
        dist = np.sqrt(dx*dx + dy*dy) + 1e-6
        pull = np.minimum(100.0 / dist, 5.0)
        vx[mask] += (dx/dist) * pull * FLOCK_STRENGTH
        vy[mask] += (dy/dist) * pull * FLOCK_STRENGTH
        vx[mask] += rng.uniform(-0.2, 0.2, np.sum(mask))
        vy[mask] += rng.uniform(-0.2, 0.2, np.sum(mask))

    vx *= 0.90
    vy *= 0.90
    px += vx
    py += vy
    np.clip(px, 0, WINDOW_W-1, out=px)
    np.clip(py, 0, WINDOW_H-1, out=py)

def draw_particles(frame, frame_count):
    pulse = 0.7 + 0.3 * np.sin(phase + frame_count * 0.05)
    for ci, bgr in enumerate(NEON_COLORS_BGR):
        mask = color_idx == ci
        for x, y, p, s in zip(px[mask].astype(int), py[mask].astype(int), pulse[mask], sizes[mask]):
            alpha = 0.4 + 0.6 * p
            color = (int(bgr[0]*alpha), int(bgr[1]*alpha), int(bgr[2]*alpha))
            cv2.circle(frame, (x, y), int(s), color, -1)

def draw_hud(frame, is_fist, hand_count, fps):
    h, w = frame.shape[:2]
    teal = (204, 255, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    msg = "NEURAL LINK: ACTIVE" if hand_count > 0 else "SCANNING..."
    if hand_count > 0:
        msg += f" | MODE: {'SCATTER' if is_fist else 'FLOCK'}"
    tw = cv2.getTextSize(msg, font, 0.5, 1)[0][0]
    cv2.putText(frame, msg, (w//2 - tw//2, h - 20), font, 0.5, teal, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, h - 20), font, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

# ─────────────────────────────────────────────
#  Hand Tracking Wrapper
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
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.landmarker.detect(mp_image)

    def parse(self, results):
        if not results.hand_landmarks: return []
        return [list(hand) for hand in results.hand_landmarks]

    def detect_fist(self, lm):
        closed = 0
        for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]:
            dist = math.sqrt((lm[tip].x - lm[mcp].x)**2 + (lm[tip].y - lm[mcp].y)**2)
            if dist < 0.12: closed += 1
        return closed >= 3

# ─────────────────────────────────────────────
#  Main Loop
# ─────────────────────────────────────────────
def main():
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_H)

    canvas = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
    frame_count, fps, t_prev = 0, 0.0, time.time()

    print("Starting Neural Interface... Press 'Q' or close window to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        results = tracker.process(frame)
        hands = tracker.parse(results)
        
        fingertips = []
        is_fist = False
        
        for lm in hands:
            if tracker.detect_fist(lm):
                is_fist = True
                fingertips.append((lm[0].x * WINDOW_W, lm[0].y * WINDOW_H))
            else:
                for idx in FINGERTIP_IDS:
                    fingertips.append((lm[idx].x * WINDOW_W, lm[idx].y * WINDOW_H))

        update_particles(fingertips, is_fist)
        canvas = cv2.addWeighted(canvas, 1.0 - TRAIL_ALPHA, np.zeros_like(canvas), 0, 0).astype(np.uint8)
        draw_particles(canvas, frame_count)
        draw_hud(canvas, is_fist, len(hands), fps)

        cv2.imshow("NEURAL PARTICLE INTERFACE", canvas)
        
        if cv2.getWindowProperty("NEURAL PARTICLE INTERFACE", cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            now = time.time()
            fps = 30.0 / (now - t_prev)
            t_prev = now

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()