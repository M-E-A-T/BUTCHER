
import cv2
import mediapipe as mp
import numpy as np
import threading
import time

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

# ============================================================
# GLOBAL STATE FOR OSC / FILTERS
# ============================================================
DEBUG = False
current_bpm = 100.0
current_speed = 1.0
current_flux = 0          # 0..1000 from flux script
mode = "laplacian"        # "laplacian", "morph_gradient", "colormap"
_last_bpm_print = 0.0
_last_flux_print = 0.0
_last_mode_print = 0.0

MORPH_KERNEL = np.ones((5, 5), np.uint8)
active_filters = []
osc_server = None

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

    now = time.time()
    if DEBUG and (now - _last_bpm_print > 0.5):
        print(f"[OSC] BPM = {bpm_value:.2f}, speed logic = {current_speed:.3f}")
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
    /butcher/mode:
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
        print(f"[OSC] MODE = {m} -> {mode}")
        _last_mode_print = now

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
# FILTER IMPLEMENTATION (UNCHANGED)
# ============================================================
def apply_gaussian(frame):
    return cv2.GaussianBlur(frame, (3, 3), 3)

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

def count_from_flux(n, mode_name):
    if mode_name == "laplacian":
        if n < 50: return 0
        if n < 100: return 1
        if n < 200: return 2
        if n < 300: return 3
        if n < 500: return 5
        if n < 700: return 5
        return 5

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

def apply_filter_stack(frame, filters, mode_name):
    global current_flux
    out = frame

    # COLORMAP MODE
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
            out = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)

        return apply_gaussian(out)

    # EDGE MODES
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

# -------------------------------------------------
# MONITOR / WINDOW SETUP (SINGLE WINDOW, SIMPLE)
# -------------------------------------------------
SCREEN_W = 2560
SCREEN_H = 1440

MONITOR_OFFSETS = [
    (0, 0),   # main monitor; change if needed
]

WINDOW_NAMES = ["BUTCHER1"]

for name, (ox, oy) in zip(WINDOW_NAMES, MONITOR_OFFSETS):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(name, ox, oy)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.waitKey(500)

screen_w, screen_h = SCREEN_W, SCREEN_H
print(f"Using logical screen size per monitor: {screen_w}x{screen_h}")

# -------------------------------------------------
# MEDIAPIPE SETUP
# -------------------------------------------------
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.3
)

# 👇 change index here if you want another camera
cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# No tracking, no colors, just blur boxes
frame_counter = 0

# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_counter += 1
    frame = cv2.flip(frame, 1)

    orig_h, orig_w = frame.shape[:2]

    # Face detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    # Blur faces only
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x_min = int(bbox.xmin * orig_w)
            y_min = int(bbox.ymin * orig_h)
            box_w = int(bbox.width * orig_w)
            box_h = int(bbox.height * orig_h)

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(orig_w, x_min + box_w)
            y_max = min(orig_h, y_min + box_h)

            face_roi = frame[y_min:y_max, x_min:x_max]
            if face_roi.size != 0:
                # strong blur; adjust kernel if you want more/less blur
                blurred = cv2.GaussianBlur(face_roi, (51, 51), 30)
                frame[y_min:y_max, x_min:x_max] = blurred

    # Scale to screen (keep aspect, center-crop if needed)
    if screen_w > 0 and screen_h > 0:
        frame_aspect = orig_w / orig_h
        screen_aspect = screen_w / screen_h

        if frame_aspect > screen_aspect:
            new_h = screen_h
            new_w = int(new_h * frame_aspect)
            resized_frame = cv2.resize(frame, (new_w, new_h),
                                       interpolation=cv2.INTER_AREA)
            x_offset = (new_w - screen_w) // 2
            display_frame = resized_frame[:, x_offset:x_offset+screen_w]
        else:
            new_w = screen_w
            new_h = int(new_w / frame_aspect)
            resized_frame = cv2.resize(frame, (new_w, new_h),
                                       interpolation=cv2.INTER_AREA)
            y_offset = (new_h - screen_h) // 2
            display_frame = resized_frame[y_offset:y_offset+screen_h, :]
    else:
        display_frame = frame

    # Apply mode-based filters
    filtered_frame = apply_filter_stack(display_frame, active_filters, mode)

    # Show same frame on the defined windows
    for name in WINDOW_NAMES:
        cv2.imshow(name, filtered_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):   # ESC or 'q' to quit
        break
    elif key == ord('1'):
        mode = "laplacian"
    elif key == ord('2'):
        mode = "morph_gradient"
    elif key == ord('3'):
        mode = "colormap"

cap.release()
cv2.destroyAllWindows()
stop_osc()
time.sleep(0.2)
print("Exited cleanly.")