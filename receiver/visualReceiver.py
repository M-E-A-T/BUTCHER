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
current_flux = 0   # expected 0..1000 from your flux script
stop_flag = False

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(BASE_DIR, "..", "media", "test2.mp4")

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
    return cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)

def apply_sharpen(frame):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(frame, -1, kernel)

def apply_colormap(frame):
    COLORMAPS = [
        cv2.COLORMAP_AUTUMN, cv2.COLORMAP_BONE, cv2.COLORMAP_JET,
        cv2.COLORMAP_WINTER, cv2.COLORMAP_RAINBOW, cv2.COLORMAP_OCEAN,
        cv2.COLORMAP_SUMMER, cv2.COLORMAP_SPRING, cv2.COLORMAP_COOL,
        cv2.COLORMAP_HSV, cv2.COLORMAP_PINK, cv2.COLORMAP_HOT,
        cv2.COLORMAP_PARULA, cv2.COLORMAP_MAGMA, cv2.COLORMAP_INFERNO,
        cv2.COLORMAP_PLASMA, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_CIVIDIS,
        cv2.COLORMAP_TWILIGHT, cv2.COLORMAP_TWILIGHT_SHIFTED,
        cv2.COLORMAP_TURBO, cv2.COLORMAP_DEEPGREEN,
    ]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, random.choice(COLORMAPS))

# ============================================================
# MAPPING
# ============================================================
def count_from_flux(n, mode):
    if mode == "laplacian":
        if n < 100: return 1
        if n < 200: return 2
        if n < 300: return 3
        if n < 400: return 3
        if n < 500: return 5
        if n < 600: return 5
        if n < 700: return 7
        if n < 800: return 7
        if n < 900: return 9
        return 9
    if mode == "morph_gradient":
        if n < 100: return 1
        if n < 200: return 2
        if n < 300: return 3
        if n < 400: return 4
        if n < 500: return 5
        if n < 600: return 7
        if n < 700: return 9
        if n < 800: return 11
        if n < 900: return 13
        return 15
    if mode == "sobel":
        if n < 100: return 1
        if n < 200: return 3
        if n < 300: return 3
        if n < 400: return 4
        if n < 500: return 5
        if n < 600: return 6
        if n < 700: return 7
        if n < 800: return 8
        if n < 900: return 9
        return 10
    else:
        return 0

def apply_filter_by_name(frame, name):
    return {
        "sobel": apply_sobel,
        "scharr": apply_scharr,
        "laplacian": apply_laplacian,
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
# APPLY FILTER STACK
# ============================================================
def apply_filter_stack(frame, filters, mode):
    out = frame

    # === NEW: COLORMAP MODE (key '3') ===
    if mode == "colormap":
        # current_flux is 0..1000
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        #COLORMAPS = [
        #     cv2.COLORMAP_AUTUMN, cv2.COLORMAP_BONE, cv2.COLORMAP_JET,
        #     cv2.COLORMAP_WINTER, cv2.COLORMAP_RAINBOW, cv2.COLORMAP_OCEAN,
        #     cv2.COLORMAP_SUMMER, cv2.COLORMAP_SPRING, cv2.COLORMAP_COOL,
        #     cv2.COLORMAP_HSV, cv2.COLORMAP_PINK, cv2.COLORMAP_HOT,
        #     cv2.COLORMAP_PARULA, cv2.COLORMAP_MAGMA, cv2.COLORMAP_INFERNO,
        #     cv2.COLORMAP_PLASMA, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_CIVIDIS,
        #     cv2.COLORMAP_TWILIGHT, cv2.COLORMAP_TWILIGHT_SHIFTED,
        #     cv2.COLORMAP_TURBO, cv2.COLORMAP_DEEPGREEN,
        # ]

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
            out = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
        elif current_flux < 700:
            out = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
        elif current_flux < 800:
            out = cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)
        elif current_flux < 900:
            out = cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)
        else:
            # 400–1000: hot
            out = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

        out = apply_gaussian(out)

        return out

    # === EXISTING LOGIC FOR EDGE MODES ===
    modeDict = ["laplacian", "sobel", "morph_gradient"]
    modeIndex = modeDict.index(mode)

    if current_flux == 0:
        if modeIndex == len(modeDict) - 1:
            print("route 1")
            mode = modeDict[0]
        else:
            mode = modeDict[modeIndex + 1]
            modeIndex += 1

    for _ in range(count_from_flux(current_flux, mode)):
        if mode == "laplacian":
            out = apply_laplacian(out)
        elif mode == "sobel":
            out = apply_sobel(out)
        elif mode == "morph_gradient":
            out = apply_morph_gradient(out)

    # (You can still uncomment this if you ever want colormap after edges)
    # out = apply_colormap(out)

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

print("\n=== VIDEO FILTER ENGINE RUNNING ===")
print("Press Q to quit.")
print("1 = Laplacian mode")
print("2 = Morph gradient mode")
print("3 = Colormap mode (gray / cool / hot by flux)\n")

mode = "laplacian"

while not stop_flag:
    now = time.time()

    if current_speed > 0:
        if now - last_time >= frame_time / current_speed:
            last_time = now

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            filtered = apply_filter_stack(frame, active_filters, mode)
            frozen_frame = filtered
            cv2.imshow("Video Filters", filtered)

    else:
        if frozen_frame is not None:
            cv2.imshow("Video Filters", frozen_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        stop_flag = True
        break
    elif key == ord('1'):
        print("Mode: laplacian")
        mode = "laplacian"
    elif key == ord('2'):
        print("Mode: morph_gradient")
        mode = "morph_gradient"
    elif key == ord('3'):
        print("Mode: colormap")
        mode = "colormap"

# ============================================================
# CLEAN EXIT
# ============================================================
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
stop_osc()
time.sleep(0.2)
print("Exited cleanly.")
