import cv2
import numpy as np
import threading
import time
import random
import os
import re

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

# ============================================================
# GLOBAL STATE
# ============================================================
current_bpm = 100.0
current_speed = 1.0
current_flux = 0   # expected 0..1000 from flux script
stop_flag = False

# current visual mode (laplacian, morph_gradient, colormap)
mode = "laplacian"   # default

# Throttling for prints
last_bpm_print = 0.0
last_flux_print = 0.0
last_mode_print = 0.0

# ============================================================
# PATHS / VIDEO SELECTION
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(BASE_DIR, "..", "media")

print(f"[PATH] Looking in: {MEDIA_DIR}")

video_path = None

try:
    files = os.listdir(MEDIA_DIR)
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
# FILTER STACK HOLDER (not heavily used now, but kept for future)
# ============================================================
active_filters = []

# ============================================================
# OSC CALLBACKS
# ============================================================
def got_bpm(addr, bpm_value):
    global current_bpm, current_speed, last_bpm_print
    try:
        bpm_value = float(bpm_value)
    except:
        return

    current_bpm = bpm_value
    # Map BPM to playback speed, then clamp for sanity
    current_speed = 0.0 if bpm_value <= 0 else 1.0 + 0.09 * (bpm_value - 100)
    current_speed = max(0.0, min(current_speed, 2.0))  # never more than 2x

    now = time.time()
    if now - last_bpm_print > 0.5:  # print at most twice per second
        print(f"[OSC] BPM = {bpm_value:.2f} → speed = {current_speed:.3f}x")
        last_bpm_print = now


def got_flux(addr, flux_value):
    global current_flux, last_flux_print
    try:
        current_flux = int(flux_value)
    except:
        return

    now = time.time()
    if now - last_flux_print > 0.5:
        print(f"[OSC] FLUX = {current_flux}")
        last_flux_print = now


def got_mode(addr, mode_value):
    """
    /butcher/mode callback
    1 -> laplacian
    2 -> morph_gradient
    3 -> colormap
    """
    global mode, last_mode_print
    try:
        m = int(mode_value)
    except:
        print(f"[OSC] MODE invalid value: {mode_value}")
        return

    if m == 1:
        mode = "laplacian"
    elif m == 2:
        mode = "morph_gradient"
    elif m == 3:
        mode = "colormap"
    else:
        # ignore unknown modes
        print(f"[OSC] MODE unknown int: {m}")
        return

    now = time.time()
    if now - last_mode_print > 0.5:
        print(f"[OSC] MODE = {m} → {mode}")
        last_mode_print = now


# ============================================================
# START OSC SERVER
# ============================================================
osc_server = None

def start_osc():
    global osc_server

    dispatcher = Dispatcher()
    dispatcher.map("/bpm", got_bpm)
    dispatcher.map("/butcher/flux", got_flux)
    dispatcher.map("/butcher/mode", got_mode)

    osc_server = BlockingOSCUDPServer(("0.0.0.0", 9000), dispatcher)
    print("[OSC] Listening on port 9000...")

    try:
        osc_server.serve_forever()
    except Exception as e:
        print(f"[OSC] Server stopped: {e}")


def stop_osc():
    global osc_server
    if osc_server:
        try:
            osc_server.shutdown()
        except:
            pass
        try:
            osc_server.server_close()
        except:
            pass
        osc_server = None
        print("[OSC] Closed port 9000")


osc_thread = threading.Thread(target=start_osc, daemon=True)
osc_thread.start()

# ============================================================
# FILTER FUNCTIONS (single-pass helpers)
# ============================================================
def apply_sobel_gray(gray):
    sx = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    sy = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    mag = cv2.magnitude(sx.astype(np.float32), sy.astype(np.float32))
    edges = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return edges.astype(np.uint8)

def apply_laplacian_gray(gray):
    edges = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    edges = cv2.convertScaleAbs(edges)
    return edges

def apply_morph_gradient_gray(gray):
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    return edges

# ============================================================
# APPLY FILTER STACK (optimized: single pass, blend by flux)
# ============================================================
def apply_filter_stack(frame, filters, mode_name):
    """
    mode_name:
      - 'laplacian'      -> edge detect + blend
      - 'morph_gradient' -> morph gradient + blend
      - 'sobel'          -> sobel + blend (if used)
      - 'colormap'       -> colormap based on flux
    """
    # COLORMAP MODE (still only one grayscale conversion)
    if mode_name == "colormap":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
            out = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

        # light blur once
        out = cv2.GaussianBlur(out, (3, 3), 3)
        return out

    # EDGE MODES — ONE FILTER PER FRAME
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if mode_name == "laplacian":
        edges = apply_laplacian_gray(gray)
    elif mode_name == "sobel":
        edges = apply_sobel_gray(gray)
    elif mode_name == "morph_gradient":
        edges = apply_morph_gradient_gray(gray)
    else:
        # fallback: no processing
        return frame

    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Use flux to control how strong edges are (0.2–1.0)
    alpha = np.clip(current_flux / 400.0, 0.2, 1.0)
    beta = 1.0 - alpha

    blended = cv2.addWeighted(frame, beta, edges_color, alpha, 0)
    return blended

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

# ---- FPS HANDLING ----
fps_raw = cap.get(cv2.CAP_PROP_FPS)
print(f"[VIDEO] Raw FPS from OpenCV: {fps_raw}")

if fps_raw <= 0 or fps_raw > 60:
    fps = 30.0
else:
    fps = fps_raw

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
        # Only advance frames based on fps and current_speed
        if now - last_time >= frame_time / max(current_speed, 0.0001):
            last_time = now

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # -------- performance tweak: downscale big frames --------
            h, w = frame.shape[:2]
            MAX_DIM = 720  # tweak this: 480 is faster, 720 nicer

            if max(h, w) > MAX_DIM:
                scale = MAX_DIM / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame_proc = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                frame_proc = frame
            # ---------------------------------------------------------

            filtered = apply_filter_stack(frame_proc, active_filters, mode)
            cv2.imshow("Video Filters", filtered)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        stop_flag = True
    elif key == ord('1'):
        mode = "laplacian"
        print("Mode: Laplacian (manual)")
    elif key == ord('2'):
        mode = "morph_gradient"
        print("Mode: Morph Gradient (manual)")
    elif key == ord('3'):
        mode = "colormap"
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
