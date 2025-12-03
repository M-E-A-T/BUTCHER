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
video_path = os.path.join(BASE_DIR, "../../", "media", "test.mp4")

# ============================================================
# FILTER STACK (your originals)
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
# SAFE OSC SERVER
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
        try: osc_server.shutdown()
        except: pass
        try: osc_server.server_close()
        except: pass
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
# COLORMAP CYCLER (NO LAPLACIAN ANYWHERE)
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

current_cmap_idx = 0
last_cmap_change = time.time()
CMAP_PERIOD = 5.0

def apply_colormap(frame):
    global current_cmap_idx, last_cmap_change

    now = time.time()
    if now - last_cmap_change >= CMAP_PERIOD:
        current_cmap_idx = (current_cmap_idx + 1) % len(COLORMAPS)
        last_cmap_change = now

    cmap, name = COLORMAPS[current_cmap_idx]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    colored = cv2.applyColorMap(gray, cmap)

    cv2.putText(
        colored, f"COLORMAP: {name}", (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA
    )
    return colored

# ============================================================
# FILTER MAPPING
# ============================================================
def apply_filter_by_name(frame, name):
    return {
        "sobel": apply_sobel,
        "scharr": apply_scharr,
        "canny": apply_canny,
        "gaussian": apply_gaussian,
        "median": apply_median,
        "bilateral": apply_bilateral,
        "stylization": apply_stylization,
        "pencil": apply_pencil,
        "detail": apply_detail_enhance,
        "edge_preserve": apply_edge_preserving,
        "morph_gradient": apply_morph_gradient,
        "sharpen": apply_sharpen,
        "colormap": apply_colormap,
    }.get(name, lambda f: f)(frame)

# ============================================================
# APPLY FILTER STACK  (NO LAPLACIAN)
# ============================================================
def apply_filter_stack(frame, filters):
    out = frame

    # Apply all base filters except laplacian (removed)
    base = [f for f in filters if f != "laplacian"]
    for f in base:
        out = apply_filter_by_name(out, f)

    # Always apply colormap last
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

print("\n=== VIDEO FILTER ENGINE RUNNING (NO LAPLACIAN) ===")
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
