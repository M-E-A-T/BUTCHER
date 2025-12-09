import cv2
import numpy as np
import threading
import time
import random
import os
import subprocess

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import re



# ============================================================
# GLOBAL STATE
# ============================================================
current_bpm = 100.0
current_speed = 1.0
current_flux = 0   # expected 0..1000 from flux script
stop_flag = False

screen_w, screen_h = None, None  # set later

# current visual mode (laplacian, morph_gradient, colormap)
mode = "laplacian"   # default


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(BASE_DIR, "..", "media")

print(f"[PATH] Looking in: {MEDIA_DIR}")

video_path = None

try:
    files = os.listdir(MEDIA_DIR)
    print(f"[PATH] Found files: {files}")
except Exception as e:
    print(f"[ERROR] Cannot read media directory: {e}")
    # stop_osc() would be undefined here, so just exit
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
    # stop_osc() would be undefined here, so just exit
    exit()

print(f"[VIDEO] Loaded: {video_path}")
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


def got_mode(addr, mode_value):
    """
    /butcher/mode callback
    1 -> laplacian
    2 -> morph_gradient
    3 -> colormap
    """
    global mode
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

    print(f"[OSC] MODE = {m} → {mode}")


# ============================================================
# START OSC SERVER
# ============================================================
osc_server = None

def start_osc():
    global osc_server

    dispatcher = Dispatcher()
    dispatcher.map("/bpm", got_bpm)
    dispatcher.map("/butcher/flux", got_flux)
    dispatcher.map("/butcher/mode", got_mode)   # NEW: mode endpoint

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


# ============================================================
# MAPPING
# ============================================================
def count_from_flux(n, mode_name):
    if mode_name == "laplacian":
        if n < 50: return 0
        if n < 100: return 1
        if n < 200: return 2
        if n < 300: return 3
        if n < 500: return 5
        if n < 700: return 7
        return 9

    if mode_name == "morph_gradient":
        if n < 50: return 0
        if n < 100: return 1
        if n < 200: return 2
        if n < 300: return 3
        if n < 500: return 7
        return 13

    if mode_name == "sobel":
        if n < 50: return 0
        if n < 100: return 1
        if n < 300: return 3
        if n < 600: return 6
        return 10

    return 0

# ============================================================
# APPLY FILTER STACK
# ============================================================
def apply_filter_stack(frame, filters, mode_name):
    out = frame

    # === COLORMAP MODE ===
    if mode_name == "colormap":
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        if current_flux < 50:
            out = gray
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
            # 400–1000: hot
            out = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

        return apply_gaussian(out)

    # === EDGE MODES ===
    for _ in range(count_from_flux(current_flux, mode_name)):
        if mode_name == "laplacian":
            out = apply_laplacian(out)
        elif mode_name == "sobel":
            out = apply_sobel(out)
        elif mode_name == "morph_gradient":
            out = apply_morph_gradient(out)

    return out


# ============================================================
# SAFE FULLSCREEN RESIZE
# ============================================================
def safe_fullscreen_resize(img):
    global screen_w, screen_h

    if screen_w is None or screen_h is None:
        # Try to get real screen size on macOS via AppleScript
        try:
            cmd = r"""osascript -e 'tell application "Finder" to get bounds of window of desktop'"""
            raw = subprocess.check_output(cmd, shell=True).decode().strip()
            # raw looks like: "{0, 0, 1440, 900}"
            raw = raw.replace("{", "").replace("}", "")
            parts = [int(v.strip()) for v in raw.split(",")]
            left, top, right, bottom = parts
            screen_w = right - left
            screen_h = bottom - top
        except Exception as e:
            print(f"[Fullscreen] AppleScript failed: {e}")
            # Fallback to OpenCV window rect
            try:
                _, _, w, h = cv2.getWindowImageRect("Video Filters")
                if w > 0 and h > 0:
                    screen_w, screen_h = w, h
                else:
                    raise ValueError
            except Exception as e2:
                print(f"[Fullscreen] OpenCV rect failed: {e2}")
                # Final fallback: image size
                screen_h, screen_w = img.shape[:2]

        print(f"[Fullscreen] Using {screen_w}x{screen_h}")

    try:
        return cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    except Exception as e:
        print(f"[Fullscreen] Resize failed: {e}")
        return img


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

# ---- FPS HANDLING (ONLY CHANGE) ----
fps_raw = cap.get(cv2.CAP_PROP_FPS)
print(f"[VIDEO] Raw FPS from OpenCV: {fps_raw}")

if fps_raw <= 0 or fps_raw > 60:
    fps = 30.0
else:
    fps = fps_raw

print(f"[VIDEO] Using FPS: {fps}")
frame_time = 1.0 / fps
# ------------------------------------

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
        if now - last_time >= frame_time / current_speed:
            last_time = now

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # -------- performance tweak: downscale big frames --------
            h, w = frame.shape[:2]

            # You can tweak this threshold; 720 is a good starting point
            MAX_DIM = 720  

            if max(h, w) > MAX_DIM:
                scale = MAX_DIM / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame_proc = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                frame_proc = frame
            # ---------------------------------------------------------

            filtered = apply_filter_stack(frame_proc, active_filters, mode)
            stretched = safe_fullscreen_resize(filtered)
            cv2.imshow("Video Filters", stretched)

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
