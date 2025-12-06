import cv2
import numpy as np
import threading
import time
import random
import os

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

# ============================================================
# GLOBAL STATE
# ============================================================
current_bpm = 100.0
current_speed = 1.0
current_flux = 0
stop_flag = False

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(BASE_DIR, "..", "media", "test.mp4")

# ============================================================
# FILTER STACK
# (Define default filters here — no user input anymore)
# ============================================================
active_filters = [
    # Example defaults:
    # "gaussian",
    # "sobel",
    # "stylization",
]

# ============================================================
# OSC CALLBACKS
# ============================================================
def got_bpm(addr, bpm_value):
    global current_bpm, current_speed
    try:
        bpm_value = float(bpm_value)
    except:
        return

    current_bpm = bpm_value
    current_speed = 0.0 if bpm_value <= 0 else 1.0 + 0.09 * (bpm_value - 100)

    print(f"[OSC] BPM = {bpm_value:.2f} → speed = {current_speed:.3f}x")

def got_flux(addr, flux_value):
    global current_flux
    try:
        current_flux = int(flux_value)
    except:
        return
    print(f"[OSC] FLUX = {current_flux}")

# ============================================================
# SAFE OSC SERVER (start & clean shutdown)
# ============================================================
osc_server = None

def start_osc():
    global osc_server

    dispatcher = Dispatcher()
    dispatcher.map("/bpm", got_bpm)
    dispatcher.map("/flux", got_flux)

    osc_server = BlockingOSCUDPServer(("0.0.0.0", 9000), dispatcher)
    print("[OSC] Listening on port 9000...")
    try:
        osc_server.serve_forever()
    except Exception as e:
        print(f"[OSC] Server stopped: {e}")

def stop_osc():
    global osc_server
    if osc_server:
        print("[OSC] Shutting down...")
        try:
            osc_server.shutdown()
        except:
            pass
        try:
            osc_server.server_close()
        except:
            pass
        osc_server = None
        print("[OSC] Port released.")

osc_thread = threading.Thread(target=start_osc, daemon=True)
osc_thread.start()

# ============================================================
# FILTER FUNCTIONS
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
    # Currently not used in stack, but left here if you want it back later
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F))
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

def apply_canny(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_gaussian(frame):
    return cv2.GaussianBlur(frame, (15, 15), 3)

def apply_median(frame):
    return cv2.medianBlur(frame, 9)

def apply_bilateral(frame):
    return cv2.bilateralFilter(frame, 9, 75, 75)

def apply_stylization(frame):
    return cv2.stylization(frame)

def apply_pencil(frame):
    _, sketch = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return sketch

def apply_detail_enhance(frame):
    return cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)

def apply_edge_preserving(frame):
    return cv2.edgePreservingFilter(frame, flags=1, sigma_s=60, sigma_r=0.4)

def apply_morph_gradient(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    return cv2.cvtColor(grad.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def apply_sharpen(frame):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(frame, -1, kernel)

# ============================================================
# COLORMAP LOGIC DRIVEN BY FLUX (WITH BLENDING)
# ============================================================
COLORMAPS = [
    (cv2.COLORMAP_AUTUMN,  "AUTUMN"),
    (cv2.COLORMAP_BONE,    "BONE"),
    (cv2.COLORMAP_JET,     "JET"),
    (cv2.COLORMAP_WINTER,  "WINTER"),
    (cv2.COLORMAP_RAINBOW, "RAINBOW"),
    (cv2.COLORMAP_OCEAN,   "OCEAN"),
    (cv2.COLORMAP_SUMMER,  "SUMMER"),
    (cv2.COLORMAP_SPRING,  "SPRING"),
    (cv2.COLORMAP_COOL,    "COOL"),
    (cv2.COLORMAP_HSV,     "HSV"),
    (cv2.COLORMAP_PINK,    "PINK"),
    (cv2.COLORMAP_HOT,     "HOT"),
    (cv2.COLORMAP_PARULA,  "PARULA"),
    (cv2.COLORMAP_MAGMA,   "MAGMA"),
    (cv2.COLORMAP_INFERNO, "INFERNO"),
    (cv2.COLORMAP_PLASMA,  "PLASMA"),
    (cv2.COLORMAP_VIRIDIS, "VIRIDIS"),
    (cv2.COLORMAP_CIVIDIS, "CIVIDIS"),
    (cv2.COLORMAP_TWILIGHT, "TWILIGHT"),
    (cv2.COLORMAP_TWILIGHT_SHIFTED, "TWILIGHT_SHIFTED"),
    (cv2.COLORMAP_TURBO,   "TURBO"),
    (cv2.COLORMAP_DEEPGREEN, "DEEPGREEN"),
]

# Groups based on flux, like laplacian_count_from_flux but for color
LOW_GROUP = [
    (cv2.COLORMAP_BONE, "BONE"),
]

MID_GROUP = [
    (cv2.COLORMAP_HOT,  "HOT"),
    (cv2.COLORMAP_COOL, "COOL"),
]

HIGH_GROUP = [
    (cv2.COLORMAP_DEEPGREEN, "DEEPGREEN"),
    (cv2.COLORMAP_HOT,       "HOT"),
    (cv2.COLORMAP_COOL,      "COOL"),
    (cv2.COLORMAP_SPRING,    "SPRING"),
    (cv2.COLORMAP_MAGMA,     "MAGMA"),
]

# --- blending state ---
current_cmap_id = cv2.COLORMAP_BONE
prev_cmap_id = cv2.COLORMAP_BONE
cmap_blend = 1.0                # 0..1, 1 = fully current_cmap_id
last_cmap_change = time.time()
CMAP_FADE_TIME = 0.7            # seconds to fade between colors

def choose_colormap_from_flux(flux_value: int):
    """
    Interpret flux tiers:
      - flux < 10  → BONE only
      - 10–19      → HOT / COOL
      - >= 20      → DEEPGREEN / HOT / COOL / SPRING / MAGMA
    """
    if flux_value < 10:
        group = LOW_GROUP
    elif flux_value < 20:
        group = MID_GROUP
    else:
        group = HIGH_GROUP

    cmap, name = random.choice(group)
    return cmap

def apply_colormap(frame):
    global current_cmap_id, prev_cmap_id, cmap_blend, last_cmap_change

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1) Decide what colormap we *want* based on current flux
    target_cmap = choose_colormap_from_flux(current_flux)

    now = time.time()
    dt = now - last_cmap_change
    last_cmap_change = now

    # 2) If target changed, start a new crossfade
    if target_cmap != current_cmap_id and cmap_blend >= 1.0:
        prev_cmap_id = current_cmap_id
        current_cmap_id = target_cmap
        cmap_blend = 0.0  # reset blend, start fading to new color

    # 3) Advance blend toward 1.0 over CMAP_FADE_TIME
    if CMAP_FADE_TIME > 0:
        cmap_blend = min(1.0, cmap_blend + dt / CMAP_FADE_TIME)
    else:
        cmap_blend = 1.0

    # 4) Colorize with both previous + current colormaps and blend
    col_prev = cv2.applyColorMap(gray, prev_cmap_id)
    col_curr = cv2.applyColorMap(gray, current_cmap_id)

    blended = cv2.addWeighted(col_prev, 1.0 - cmap_blend, col_curr, cmap_blend, 0.0)
    return blended

# ============================================================
# MAPPING
# ============================================================
def laplacian_count_from_flux(n):
    # You can still use this later if you bring Laplacian back into the stack
    if n < 10: return 1
    if n < 20: return 3
    return 5

def apply_filter_by_name(frame, name):
    return {
        "sobel":          apply_sobel,
        "scharr":         apply_scharr,
        "laplacian":      apply_laplacian,
        "canny":          apply_canny,
        "gaussian":       apply_gaussian,
        "median":         apply_median,
        "bilateral":      apply_bilateral,
        "stylization":    apply_stylization,
        "pencil":         apply_pencil,
        "detail":         apply_detail_enhance,
        "edge_preserve":  apply_edge_preserving,
        "morph_gradient": apply_morph_gradient,
        "sharpen":        apply_sharpen,
        "colormap":       apply_colormap,
    }.get(name, lambda f: f)(frame)

# ============================================================
# APPLY FILTER STACK
# ============================================================
def apply_filter_stack(frame, filters):
    out = frame

    # Apply base filters (ignore laplacian & colormap here)
    base = [f for f in filters if f not in ("laplacian", "colormap")]
    for f in base:
        out = apply_filter_by_name(out, f)

    # Laplacian currently disabled in stack; uncomment if you want it:
    # for _ in range(laplacian_count_from_flux(current_flux)):
    #     out = apply_laplacian(out)

    # Always colormap last, driven by current_flux and smoothly blended
    out = apply_colormap(out)
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

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
frame_time = 1.0 / fps
last_time = time.time()
frozen_frame = None

print("\n=== VIDEO FILTER ENGINE RUNNING (FLUX → BLENDED COLORMAPS) ===")
print("Press Q to quit.\n")

while not stop_flag:
    now = time.time()

    if current_speed > 0:
        if now - last_time >= frame_time / current_speed:
            last_time = now

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            filtered = apply_filter_stack(frame, active_filters)
            frozen_frame = filtered
            cv2.imshow("Video Filters", filtered)

    else:
        if frozen_frame is not None:
            cv2.imshow("Video Filters", frozen_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        stop_flag = True
        break

# ============================================================
# CLEAN EXIT
# ============================================================
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
stop_osc()
time.sleep(0.2)
print("Exited cleanly.")
