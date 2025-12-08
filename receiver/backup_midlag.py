import cv2
import numpy as np
import threading
import time
import os
import re

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

# ============================================================
# CONFIG (TUNED FOR OLD LAPTOP)
# ============================================================
DEBUG = False          # set True if you want verbose prints
MAX_DIM = 720          # working resolution (try 240 if still laggy, 480 if you want prettier)
MAX_SPEED = 1.5        # cap playback speed multiplier (was 2.0 before)

# ============================================================
# GLOBAL STATE
# ============================================================
current_bpm = 100.0
current_speed = 1.0
current_flux = 0   # expected 0..1000 from flux script
stop_flag = False

screen_w, screen_h = None, None  # unused now but kept for compatibility

# current visual mode (laplacian, morph_gradient, colormap)
mode = "laplacian"   # default

# throttle printing so it doesn't lag
_last_bpm_print = 0.0
_last_flux_print = 0.0
_last_mode_print = 0.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(BASE_DIR, "..", "media")

print(f"[PATH] Looking in: {MEDIA_DIR}")

video_path = None

try:
    files = os.listdir(MEDIA_DIR)
    if DEBUG:
        print(f"[PATH] Found files: {files}")
except Exception as e:
    print(f"[ERROR] Cannot read media directory: {e}")
    exit()

# Regex patterns:
#   video1, video2, video10...
#   video_vertical1, video_vertical12...
pattern_normal   = re.compile(r"^video\d+\.", re.IGNORECASE)
pattern_vertical = re.compile(r"^video_vertical\d+\.", re.IGNORECASE)

for fname in sorted(files):  # sorted → stable order
    lower = fname.lower()

    if pattern_normal.match(lower) or pattern_vertical.match(lower):
        full = os.path.join(MEDIA_DIR, fname)
        if os.path.isfile(full):
            video_path = full
            break

if video_path is None:
    print("[ERROR] No videos found (expected video<num>.* or video_vertical<num>.*)")
    exit()

print(f"[VIDEO] Loaded: {video_path}")

# ============================================================
# FILTER STACK
# ============================================================
active_filters = []

# Precompute kernel for morph gradient (used a lot at high flux)
MORPH_KERNEL = np.ones((5, 5), np.uint8)

# ============================================================
# OSC CALLBACKS
# ============================================================
def got_bpm(addr, bpm_value):
    global current_bpm, current_speed, _last_bpm_print
    try:
        bpm_value = float(bpm_value)
    except:
        return

    current_bpm = bpm_value
    current_speed = 0.0 if bpm_value <= 0 else 1.0 + 0.09 * (bpm_value - 100)
    current_speed = max(0.0, min(current_speed, MAX_SPEED))

    now = time.time()
    if DEBUG and (now - _last_bpm_print > 0.5):
        print(f"[OSC] BPM = {bpm_value:.2f} → speed = {current_speed:.3f}x")
        _last_bpm_print = now


def got_flux(addr, flux_value):
    global current_flux, _last_flux_print
    try:
        current_flux = int(flux_value)
    except:
        return

    now = time.time()
    if DEBUG and (now - _last_flux_print > 0.5):
        print(f"[OSC] FLUX = {current_flux}")
        _last_flux_print = now


def got_mode(addr, mode_value):
    """
    /butcher/mode callback
    1 -> laplacian
    2 -> morph_gradient
    3 -> colormap
    """
    global mode, _last_mode_print
    try:
        m = int(mode_value)
    except:
        if DEBUG:
            print(f"[OSC] MODE invalid value: {mode_value}")
        return

    if m == 1:
        mode = "laplacian"
    elif m == 2:
        mode = "morph_gradient"
    elif m == 3:
        mode = "colormap"
    else:
        if DEBUG:
            print(f"[OSC] MODE unknown int: {m}")
        return

    now = time.time()
    if DEBUG and (now - _last_mode_print > 0.5):
        print(f"[OSC] MODE = {m} → {mode}")
        _last_mode_print = now


# ============================================================
# START OSC SERVER
# ============================================================
osc_server = None

def start_osc():
    global osc_server

    dispatcher = Dispatcher()
    dispatcher.map("/bpm", got_bpm)
    dispatcher.map("/butcher/flux", got_flux)
    dispatcher.map("/butcher/mode", got_mode)   # mode endpoint

    osc_server = BlockingOSCUDPServer(("0.0.0.0", 9000), dispatcher)
    print("[OSC] Listening on port 9000...")

    try:
        osc_server.serve_forever()
    except Exception as e:
        print(f"[OSC] Server stopped: {e}")


def stop_osc():
    global osc_server
    if osc_server:
        try: osc_server.shutdown()
        except: pass
        try: osc_server.server_close()
        except: pass
        osc_server = None
        print("[OSC] Closed port 9000")


osc_thread = threading.Thread(target=start_osc, daemon=True)
osc_thread.start()

# ============================================================
# ORIGINAL BGR FILTER FUNCTIONS (kept if you need them elsewhere)
# ============================================================
def apply_sobel(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag = cv2.normalize(cv2.magnitude(sx, sy), None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(mag.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def apply_scharr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    sy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    mag = cv2.normalize(cv2.magnitude(sx, sy), None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(mag.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def apply_laplacian(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

def apply_morph_gradient(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, MORPH_KERNEL)
    return cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)

def apply_gaussian(frame):
    return cv2.GaussianBlur(frame, (3, 3), 3)

# === grayscale-only helpers for fast multi-pass ===
def laplacian_gray(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    return lap

def sobel_gray(gray):
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag = cv2.magnitude(sx, sy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)

def morph_gradient_gray(gray):
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, MORPH_KERNEL)
    return grad

# ============================================================
# MAPPING (unchanged)
# ============================================================
def count_from_flux(n, mode_name):
    if mode_name == "laplacian":
        if n < 100: return 1
        if n < 200: return 2
        if n < 300: return 3
        if n < 500: return 5
        if n < 700: return 5
        return 5

    if mode_name == "morph_gradient":
        if n < 100: return 1
        if n < 200: return 2
        if n < 300: return 3
        if n < 500: return 7
        return 13

    if mode_name == "sobel":
        if n < 100: return 1
        if n < 300: return 3
        if n < 600: return 6
        return 10

    return 0

# ============================================================
# APPLY FILTER STACK (same behavior, optimized implementation)
# ============================================================
def apply_filter_stack(frame, filters, mode_name):
    out = frame

    # === COLORMAP MODE (unchanged logic) ===
    if mode_name == "colormap":
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        if current_flux < 100:
            out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif current_flux < 200:
            out = cv2.applyColorMap(gray, cv2.COLORMAP_COOL)
        elif current_flux < 300:
            out = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        elif current_flux < 400:
            out = cv2.applyColorMap(gray, cv2.COLORMAP_DEEPGREEN)
        elif current_flux < 500:
            out = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
        elif current_flux < 600:
            out = cv2.applyColorMap(gray, cv2.COLORMAP_COOL)
        elif current_flux < 700:
            out = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
        elif current_flux < 800:
            out = cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)
        elif current_flux < 900:
            out = cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)
        else:
            # 400–1000: turbo
            out = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

        return apply_gaussian(out)

    # === EDGE MODES ===
    # Multi-pass on grayscale, same pass counts as original.
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    passes = count_from_flux(current_flux, mode_name)

    for _ in range(passes):
        if mode_name == "laplacian":
            gray = laplacian_gray(gray)
        elif mode_name == "sobel":
            gray = sobel_gray(gray)
        elif mode_name == "morph_gradient":
            gray = morph_gradient_gray(gray)

    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return out

# ============================================================
# VIDEO LOOP
# ============================================================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("ERROR: Could not open video")
    stop_osc()
    exit()

cv2.namedWindow("Video Filters", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Video Filters", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ---- FPS HANDLING (tuned) ----
fps_raw = cap.get(cv2.CAP_PROP_FPS)
print(f"[VIDEO] Raw FPS from OpenCV: {fps_raw}")

# old laptops: don't try to hit 60 fps
if fps_raw <= 0:
    fps = 24.0
else:
    fps = min(fps_raw, 30.0)

print(f"[VIDEO] Using FPS: {fps}")
frame_time = 1.0 / fps
# -----------------------

last_time = time.time()

print("\n=== VIDEO FILTER ENGINE RUNNING ===")
print("Press Q to quit.")
print("1 = Laplacian mode (manual override)")
print("2 = Morph gradient mode (manual override)")
print("3 = Colormap mode (manual override)")
print("OSC /butcher/mode (1/2/3) also switches mode.\n")

while not stop_flag:
    now = time.time()

    if current_speed > 0:
        if now - last_time >= frame_time / max(current_speed, 0.0001):
            last_time = now

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # -------- downscale BEFORE filters (big win) --------
            h, w = frame.shape[:2]

            if max(h, w) > MAX_DIM:
                scale = MAX_DIM / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame_proc = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                frame_proc = frame
            # ----------------------------------------------------

            filtered = apply_filter_stack(frame_proc, active_filters, mode)
            # Let OpenCV fullscreen scale this, no extra resize
            cv2.imshow("Video Filters", filtered)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        stop_flag = True
    elif key == ord('1'):
        mode = "laplacian"
        if DEBUG:
            print("Mode: Laplacian (manual)")
    elif key == ord('2'):
        mode = "morph_gradient"
        if DEBUG:
            print("Mode: Morph Gradient (manual)")
    elif key == ord('3'):
        mode = "colormap"
        if DEBUG:
            print("Mode: Colormap (manual)")

# ============================================================
# CLEAN EXIT
# ============================================================
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
stop_osc()
time.sleep(0.2)
print("Exited cleanly.")
