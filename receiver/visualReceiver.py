import cv2
import numpy as np
import threading
import time
import random
import os
import subprocess

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

# ============================================================
# GLOBAL STATE
# ============================================================
current_bpm = 100.0
current_speed = 1.0
current_flux = 0   # expected 0..1000 from flux script
stop_flag = False

screen_w, screen_h = None, None  # set later

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(BASE_DIR, "..", "media", "test.mp4")

# ============================================================
# FILTER STACK
# ============================================================
active_filters = []

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
# START OSC SERVER
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
        try: osc_server.shutdown()
        except: pass
        try: osc_server.server_close()
        except: pass
        osc_server = None
        print("[OSC] Closed port 9000")


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

def apply_morph_gradient(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    return cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)

def apply_gaussian(frame):
    return cv2.GaussianBlur(frame, (3, 3), 3)

# def apply_colormap(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     return cv2.applyColorMap(gray, random.choice([
#         cv2.COLORMAP_HOT, cv2.COLORMAP_JET, cv2.COLORMAP_COOL,
#         cv2.COLORMAP_MAGMA, cv2.COLORMAP_RAINBOW, cv2.COLORMAP_TURBO
#     ]))

# ============================================================
# MAPPING
# ============================================================
def count_from_flux(n, mode):
    if mode == "laplacian":
        if n < 100: return 1
        if n < 200: return 2
        if n < 300: return 3
        if n < 500: return 5
        if n < 700: return 7
        return 9

    if mode == "morph_gradient":
        if n < 100: return 1
        if n < 200: return 2
        if n < 300: return 3
        if n < 500: return 7
        return 13

    if mode == "sobel":
        if n < 100: return 1
        if n < 300: return 3
        if n < 600: return 6
        return 10

    return 0

# ============================================================
# APPLY FILTER STACK
# ============================================================
def apply_filter_stack(frame, filters, mode):
    out = frame

    # === COLORMAP MODE ===
    if mode == "colormap":
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

        return apply_gaussian(out)

    # === EDGE MODES ===
    for _ in range(count_from_flux(current_flux, mode)):
        if mode == "laplacian":
            out = apply_laplacian(out)
        elif mode == "sobel":
            out = apply_sobel(out)
        elif mode == "morph_gradient":
            out = apply_morph_gradient(out)

    return out


# ============================================================
# SAFE FULLSCREEN RESIZE
# ============================================================
def safe_fullscreen_resize(img):
    global screen_w, screen_h

    if screen_w is None or screen_h is None:
        # Try OpenCV
        try:
            _, _, w, h = cv2.getWindowImageRect("Video Filters")
            if w > 0 and h > 0:
                screen_w, screen_h = w, h
            else:
                raise ValueError
        except:
            # macOS fallback using AppleScript
            try:
                cmd = """osascript -e 'tell application "Finder" to get bounds of window of desktop'"""
                out = subprocess.check_output(cmd, shell=True).decode().strip().split(",")
                screen_w = int(out[2]) - int(out[0])
                screen_h = int(out[3]) - int(out[1])
            except:
                # fallback to image size
                screen_h, screen_w = img.shape[:2]

        print(f"[Fullscreen] Using {screen_w}x{screen_h}")

    # final safe resize
    try:
        return cv2.resize(img, (screen_w, screen_h), interpolation=cv2.INTER_LINEAR)
    except:
        return img  # fallback (never crashes)

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

mode = "laplacian"

print("\n=== VIDEO FILTER ENGINE RUNNING ===")
print("Press Q to quit.")
print("1 = Laplacian mode")
print("2 = Morph gradient mode")
print("3 = Colormap mode\n")

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
            stretched = safe_fullscreen_resize(filtered)
            cv2.imshow("Video Filters", stretched)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        stop_flag = True
    elif key == ord('1'):
        mode = "laplacian"
        print("Mode: Laplacian")
    elif key == ord('2'):
        mode = "morph_gradient"
        print("Mode: Morph Gradient")
    elif key == ord('3'):
        mode = "colormap"
        print("Mode: Colormap")

# ============================================================
# CLEAN EXIT
# ============================================================
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
stop_osc()
time.sleep(0.2)
print("Exited cleanly.")
