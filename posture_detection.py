
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
from collections import deque
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
CAMERA_INDEX       = 0
SCORE_HISTORY_LEN  = 120        # frames kept for sparkline graph
ALERT_HOLD_FRAMES  = 60         # how long an alert stays on screen
SCREENSHOT_DIR     = "posture_screenshots"
LOG_FILE           = "posture_log.txt"

# Angle thresholds (degrees) – tuned for a seated desk posture
NECK_GOOD_MAX      = 15         # forward head tilt threshold
SPINE_GOOD_MAX     = 10         # lateral spine lean threshold
SHOULDER_GOOD_MAX  = 8          # shoulder elevation / asymmetry threshold

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def angle_3pts(a, b, c):
    """Return the angle (°) at vertex b formed by points a–b–c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_val = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos_val, -1.0, 1.0)))

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def landmark_to_px(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))

def score_to_grade(score):
    if score >= 90: return "A", (50, 220, 100)
    if score >= 75: return "B", (80, 200, 255)
    if score >= 55: return "C", (0, 200, 255)
    if score >= 35: return "D", (0, 140, 255)
    return "F", (0, 60, 255)

def draw_rounded_rect(img, pt1, pt2, color, alpha=0.55, radius=12):
    """Semi-transparent rounded rectangle panel."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_bar(img, x, y, w, h, value, max_val, color_good, color_bad):
    """Horizontal progress bar with gradient color."""
    ratio = min(value / max_val, 1.0)
    color = color_good if ratio < 0.5 else color_bad
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    cv2.rectangle(img, (x, y), (x + int(w * ratio), y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 120), 1)

def draw_sparkline(img, scores, x, y, w, h):
    """Mini line chart of recent posture scores."""
    if len(scores) < 2:
        return
    pts = []
    for i, s in enumerate(scores):
        px = x + int(i / (len(scores) - 1) * w)
        py = y + h - int(s / 100 * h)
        pts.append((px, py))
    for i in range(1, len(pts)):
        color = (50, 220, 100) if scores[i] > 65 else (0, 140, 255)
        cv2.line(img, pts[i-1], pts[i], color, 1)

# ─────────────────────────────────────────────
#  POSTURE ANALYSER CLASS
# ─────────────────────────────────────────────

class PostureAnalyzer:
    def __init__(self):
        self.mp_pose    = mp.solutions.pose
        self.mp_draw    = mp.solutions.drawing_utils
        self.pose       = self.mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=1
        )

        self.score_history   = deque([100] * SCORE_HISTORY_LEN, maxlen=SCORE_HISTORY_LEN)
        self.alerts          = []       # list of (message, frame_count_remaining, color)
        self.session_start   = time.time()
        self.frame_count     = 0
        self.good_frames     = 0
        self.bad_frames      = 0
        self.last_screenshot = 0
        self.show_skeleton   = True
        self.show_hud        = True

        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        self._init_log()

    def _init_log(self):
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 60}\nSession started: {datetime.now()}\n{'=' * 60}\n")

    def _log(self, msg):
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")

    # ── Core analysis ──────────────────────────────────────────────────────

    def analyse(self, landmarks, w, h):
        """
        Compute neck tilt, spine lean, and shoulder level angles.
        Returns a posture_score (0–100) and a dict of metrics.
        """
        lm = landmarks.landmark
        IDX = self.mp_pose.PoseLandmark

        def px(idx): return landmark_to_px(lm[idx], w, h)

        # Key points
        nose        = px(IDX.NOSE)
        l_eye       = px(IDX.LEFT_EYE)
        r_eye       = px(IDX.RIGHT_EYE)
        l_ear       = px(IDX.LEFT_EAR)
        r_ear       = px(IDX.RIGHT_EAR)
        l_shoulder  = px(IDX.LEFT_SHOULDER)
        r_shoulder  = px(IDX.RIGHT_SHOULDER)
        l_hip       = px(IDX.LEFT_HIP)
        r_hip       = px(IDX.RIGHT_HIP)
        l_elbow     = px(IDX.LEFT_ELBOW)
        r_elbow     = px(IDX.RIGHT_ELBOW)
        l_wrist     = px(IDX.LEFT_WRIST)
        r_wrist     = px(IDX.RIGHT_WRIST)

        mid_shoulder = midpoint(l_shoulder, r_shoulder)
        mid_hip      = midpoint(l_hip, r_hip)
        mid_eye      = midpoint(l_eye, r_eye)

        # ── 1. NECK TILT (forward-head posture) ──────────────────────────
        # Angle between ear–shoulder–hip vertical
        neck_ref   = (mid_shoulder[0], mid_shoulder[1] - 100)   # point directly above shoulder
        neck_angle = angle_3pts(nose, mid_shoulder, neck_ref)
        # Map: 0° = perfect, NECK_GOOD_MAX° = threshold
        neck_dev   = max(0, neck_angle - 5)

        # ── 2. SPINE LEAN (lateral tilt) ─────────────────────────────────
        # Angle of shoulder-midpoint → hip-midpoint vs. vertical axis
        spine_dx   = mid_shoulder[0] - mid_hip[0]
        spine_dy   = mid_shoulder[1] - mid_hip[1]
        spine_angle = abs(math.degrees(math.atan2(spine_dx, -spine_dy)))

        # ── 3. SHOULDER LEVEL (asymmetry) ────────────────────────────────
        shoulder_dy = abs(l_shoulder[1] - r_shoulder[1])
        shoulder_angle = math.degrees(math.atan2(shoulder_dy, abs(l_shoulder[0] - r_shoulder[0]) + 1))

        # ── 4. HEAD TILT (left–right) ─────────────────────────────────────
        head_tilt_deg = abs(math.degrees(math.atan2(
            abs(l_ear[1] - r_ear[1]),
            abs(l_ear[0] - r_ear[0]) + 1
        )))

        # ── Scoring (penalty-based) ────────────────────────────────────────
        penalty_neck      = min(neck_dev      / NECK_GOOD_MAX,      1.0) * 35
        penalty_spine     = min(spine_angle   / SPINE_GOOD_MAX,     1.0) * 35
        penalty_shoulder  = min(shoulder_angle/ SHOULDER_GOOD_MAX,  1.0) * 20
        penalty_head      = min(head_tilt_deg / 12,                 1.0) * 10

        score = max(0, 100 - penalty_neck - penalty_spine - penalty_shoulder - penalty_head)

        metrics = dict(
            neck_angle      = neck_angle,
            neck_dev        = neck_dev,
            spine_angle     = spine_angle,
            shoulder_angle  = shoulder_angle,
            head_tilt       = head_tilt_deg,
            score           = score,
            mid_shoulder    = mid_shoulder,
            mid_hip         = mid_hip,
            nose            = nose,
            l_shoulder      = l_shoulder,
            r_shoulder      = r_shoulder,
        )
        return score, metrics

    # ── Alert generation ───────────────────────────────────────────────────

    def generate_alerts(self, metrics):
        new_alerts = []
        s = metrics["score"]

        if metrics["neck_dev"] > NECK_GOOD_MAX:
            new_alerts.append(("⚠  Pull head back — forward head posture", (0, 100, 255)))
        if metrics["spine_angle"] > SPINE_GOOD_MAX:
            new_alerts.append(("⚠  Straighten spine — lateral lean detected", (0, 140, 255)))
        if metrics["shoulder_angle"] > SHOULDER_GOOD_MAX:
            new_alerts.append(("⚠  Level your shoulders", (0, 160, 220)))
        if metrics["head_tilt"] > 12:
            new_alerts.append(("⚠  Keep head upright — tilt detected", (0, 120, 240)))
        if s > 80 and not new_alerts:
            new_alerts.append(("✔  Excellent posture — keep it up!", (50, 220, 100)))

        # Only update if alerts changed (avoids flickering)
        for msg, color in new_alerts:
            if not any(a[0] == msg for a in self.alerts):
                self.alerts.append((msg, ALERT_HOLD_FRAMES, color))
                self._log(f"ALERT: {msg}  [score={s:.0f}]")

    # ── HUD drawing ────────────────────────────────────────────────────────

    def draw_hud(self, frame, metrics, score):
        h, w = frame.shape[:2]
        grade, grade_color = score_to_grade(score)

        # ── Left panel ───────────────────────────────────────────────────
        draw_rounded_rect(frame, (10, 10), (260, 340), (15, 15, 25))

        # Title
        cv2.putText(frame, "POSTURE ANALYZER", (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 200), 1, cv2.LINE_AA)

        # Big score
        cv2.putText(frame, f"{int(score)}%", (20, 100),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, grade_color, 3, cv2.LINE_AA)
        cv2.putText(frame, f"Grade: {grade}", (170, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, grade_color, 2, cv2.LINE_AA)

        # Metric bars
        metrics_list = [
            ("Neck",     metrics["neck_dev"],        NECK_GOOD_MAX * 2),
            ("Spine",    metrics["spine_angle"],     SPINE_GOOD_MAX * 2),
            ("Shoulder", metrics["shoulder_angle"],  SHOULDER_GOOD_MAX * 2),
            ("Head",     metrics["head_tilt"],       24),
        ]
        y_off = 115
        for label, val, max_v in metrics_list:
            cv2.putText(frame, f"{label}: {val:.1f}°", (20, y_off + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 200), 1, cv2.LINE_AA)
            draw_bar(frame, 110, y_off + 2, 130, 12, val, max_v,
                     (50, 220, 100), (0, 80, 255))
            y_off += 30

        # Session stats
        elapsed   = int(time.time() - self.session_start)
        good_pct  = (self.good_frames / max(self.frame_count, 1)) * 100
        cv2.putText(frame, f"Session: {elapsed//60:02d}:{elapsed%60:02d}", (20, 258),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Good posture: {good_pct:.0f}%", (20, 278),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 160), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, 298),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 120), 1, cv2.LINE_AA)

        # Sparkline
        cv2.putText(frame, "Score history", (20, 318),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 100, 120), 1, cv2.LINE_AA)
        draw_rounded_rect(frame, (10, 322), (260, 355), (10, 10, 18))
        draw_sparkline(frame, list(self.score_history), 18, 325, 234, 26)

        # ── Alerts panel (right side) ────────────────────────────────────
        ax, ay = w - 380, 10
        active = [(m, f, c) for m, f, c in self.alerts if f > 0]
        if active:
            draw_rounded_rect(frame, (ax, ay), (w - 10, ay + 30 * len(active) + 16),
                              (15, 15, 25))
        updated = []
        for i, (msg, frames_left, color) in enumerate(active):
            alpha_text = min(frames_left / 20, 1.0)
            cv2.putText(frame, msg, (ax + 10, ay + 22 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
            updated.append((msg, frames_left - 1, color))
        self.alerts = updated

        # ── Controls hint ────────────────────────────────────────────────
        draw_rounded_rect(frame, (10, h - 50), (320, h - 10), (10, 10, 18))
        hints = "[S] Skeleton  [H] HUD  [P] Screenshot  [Q] Quit"
        cv2.putText(frame, hints, (18, h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 140), 1, cv2.LINE_AA)

    def draw_posture_lines(self, frame, metrics):
        """Draw spine line and neck vector as diagnostic overlays."""
        ms  = tuple(map(int, metrics["mid_shoulder"]))
        mh  = tuple(map(int, metrics["mid_hip"]))
        nos = tuple(map(int, metrics["nose"]))

        # Spine line
        cv2.line(frame, ms, mh, (0, 255, 180), 2, cv2.LINE_AA)
        # Neck line
        cv2.line(frame, nos, ms, (255, 200, 0), 2, cv2.LINE_AA)
        # Shoulder line
        cv2.line(frame,
                 tuple(map(int, metrics["l_shoulder"])),
                 tuple(map(int, metrics["r_shoulder"])),
                 (200, 100, 255), 2, cv2.LINE_AA)

    # ── Main process frame ─────────────────────────────────────────────────

    def process(self, frame):
        h, w = frame.shape[:2]
        self.frame_count += 1

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = self.pose.process(img_rgb)
        img_rgb.flags.writeable = True
        output = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        score = 50   # default when no detection

        if results.pose_landmarks:
            score, metrics = self.analyse(results.pose_landmarks, w, h)

            # Skeleton overlay
            if self.show_skeleton:
                self.mp_draw.draw_landmarks(
                    output,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
                )
                self.draw_posture_lines(output, metrics)

            self.generate_alerts(metrics)

            if score >= 65:
                self.good_frames += 1
            else:
                self.bad_frames += 1
        else:
            self.alerts.append(("  No person detected in frame", ALERT_HOLD_FRAMES, (80, 80, 200)))

        self.score_history.append(score)

        if self.show_hud:
            self.draw_hud(output, metrics if results.pose_landmarks else
                          dict(neck_dev=0, spine_angle=0, shoulder_angle=0, head_tilt=0,
                               score=50, mid_shoulder=(w//2, h//3), mid_hip=(w//2, h//2),
                               nose=(w//2, h//4), l_shoulder=(w//3, h//3),
                               r_shoulder=(2*w//3, h//3)), score)

        return output

    def screenshot(self, frame):
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(SCREENSHOT_DIR, f"posture_{ts}.jpg")
        cv2.imwrite(path, frame)
        self._log(f"Screenshot saved: {path}")
        print(f"[INFO] Screenshot saved → {path}")

    def close(self):
        elapsed = int(time.time() - self.session_start)
        good_pct = (self.good_frames / max(self.frame_count, 1)) * 100
        summary = (
            f"Session ended | Duration: {elapsed//60:02d}:{elapsed%60:02d} | "
            f"Frames: {self.frame_count} | Good posture: {good_pct:.1f}%"
        )
        self._log(summary)
        print(f"\n[SUMMARY] {summary}")
        self.pose.close()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    analyzer = PostureAnalyzer()
    print("=" * 60)
    print("  Smart Posture Analysis System — Running")
    print("  Controls:  S=Skeleton  H=HUD  P=Screenshot  Q=Quit")
    print("=" * 60)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Empty frame — retrying…")
            continue

        output = analyzer.process(frame)
        cv2.imshow("Smart Posture Analyzer — CV Project", output)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            analyzer.show_skeleton = not analyzer.show_skeleton
        elif key == ord('h'):
            analyzer.show_hud = not analyzer.show_hud
        elif key == ord('p'):
            analyzer.screenshot(output)

    cap.release()
    cv2.destroyAllWindows()
    analyzer.close()


if __name__ == "__main__":
    main()