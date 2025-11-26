import random
import cv2
import numpy as np
import threading
import time
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import os


# -----
# LAPLACIAN -> LAPLACIAN -> LAPLACIAN -> LAPLACIAN -> LAPLACIAN
# MORPH GRANDIENT -> COLOR MAP (DYNAMIC)
# MORPH GRADIENT -> MORPH GRADIENT -> MORPH GRADIENT -> MORPH GRADIENT -> MORPH GRADIENT
# LAPLACIAN -> COLOR MAP
# CANNY -> CANNY -> CANNY -> CANNY -> CANNY
# SOBEL -> SOBEL -> SOBEL -> SOBEL -> SOBEL
# ----

# -----------------------------
# GLOBAL BPM + SPEED
# -----------------------------
current_bpm = 100.0
current_speed = 1.0

# Active filter stack (in order applied)
active_filters = []  # e.g. ["gaussian", "sobel", "colormap"]

# Folder where *this script* lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the path relative to the script location
video_path = os.path.join(BASE_DIR, "..", "..", "media", "test.mp4")


# -----------------------------
# OSC CALLBACK FOR BPM
# -----------------------------
def got_bpm(addr, bpm_value):
    global current_bpm, current_speed
    try:
        bpm_value = float(bpm_value)
    except:
        return

    current_bpm = bpm_value

    if bpm_value <= 0:
        current_speed = 0.0
    else:
        # 100 BPM → 1.0, 200 BPM → 3.0
        current_speed = 1.0 + 0.09 * (bpm_value - 100)

    print(f"BPM RECEIVED: {bpm_value:.2f}   --> Speed = {current_speed:.3f}x")

# -----------------------------
# START OSC RECEIVER THREAD
# -----------------------------
def start_osc():
    dispatcher = Dispatcher()
    dispatcher.map("/bpm", got_bpm)

    server = BlockingOSCUDPServer(("0.0.0.0", 9000), dispatcher)
    print("OSC Receiver running on port 9000... (listening for /bpm)")
    server.serve_forever()

#osc_thread = threading.Thread(target=start_osc, daemon=True)
#osc_thread.start()

# -----------------------------
# FILTER FUNCTIONS
# -----------------------------
def apply_sobel(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(mag.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def apply_scharr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    mag = cv2.magnitude(scharrx, scharry)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(mag.astype(np.uint8), cv2.COLOR_GRAY2BGR)

def apply_laplacian(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = cv2.convertScaleAbs(lap)
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

def apply_canny(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ran = (random.randrange(1, 10)) * 10
    edges = cv2.Canny(gray, ran, ran)
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
    gray_sketch, color_sketch = cv2.pencilSketch(
        frame, sigma_s=60, sigma_r=0.07, shade_factor=0.05
    )
    return color_sketch

def apply_detail_enhance(frame):
    return cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)

def apply_edge_preserving(frame):
    return cv2.edgePreservingFilter(frame, flags=1, sigma_s=60, sigma_r=0.4)

def apply_morph_gradient(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    return cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)

def apply_colormap(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

def apply_sharpen(frame):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)

# -----------------------------
# APPLY SINGLE FILTER BY NAME
# -----------------------------
def apply_filter_by_name(frame, name):
    if name == "sobel":
        return apply_sobel(frame)
    elif name == "scharr":
        return apply_scharr(frame)
    elif name == "laplacian":
        return apply_laplacian(frame)
    elif name == "canny":
        return apply_canny(frame)
    elif name == "gaussian":
        return apply_gaussian(frame)
    elif name == "median":
        return apply_median(frame)
    elif name == "bilateral":
        return apply_bilateral(frame)
    elif name == "stylization":
        return apply_stylization(frame)
    elif name == "pencil":
        return apply_pencil(frame)
    elif name == "detail":
        return apply_detail_enhance(frame)
    elif name == "edge_preserve":
        return apply_edge_preserving(frame)
    elif name == "morph_gradient":
        return apply_morph_gradient(frame)
    elif name == "colormap":
        return apply_colormap(frame)
    elif name == "sharpen":
        return apply_sharpen(frame)
    else:
        return frame

# -----------------------------
# APPLY FULL STACK
# -----------------------------
def apply_filter_stack(frame, filters):
    out = frame
    for f_name in filters:
        out = apply_filter_by_name(out, f_name)

    # Label: show stack
    if filters:
        label = "Stack: " + " -> ".join(filters)
    else:
        label = "Stack: none"

    cv2.putText(out, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out

def print_stack():
    if active_filters:
        print("Current stack:", " -> ".join(active_filters))
    else:
        print("Current stack: [none]")

# -----------------------------
# VIDEO LOOP
# -----------------------------

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open test.mp4")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0
frame_time = 1.0 / fps

print("Playing video... Press P to quit.")
print("Filter keys (stacks in order of presses):")
print("  0: clear stack (no filters)")
print("  1: add Sobel")
print("  2: add Laplacian")
print("  3: add Canny")
print("  4: add Morph gradient")
print("  q: add Color map (JET)")
print("  e: add Detail enhance")
print("  r: add Gaussian blur")

last_time = time.time()
frozen_frame = None

while True:
    now = time.time()

    if current_speed > 0:
        wait_time = frame_time / current_speed

        if now - last_time >= wait_time:
            last_time = now

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Apply full stack
            frame_filtered = apply_filter_stack(frame, active_filters)

            frozen_frame = frame_filtered
            cv2.imshow("Video Filters (Stacking)", frame_filtered)

    else:
        if frozen_frame is not None:
            cv2.imshow("Video Filters (Stacking)", frozen_frame)

    # KEY CONTROLS
    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):          # quit with 'p'
        break
    elif key == ord('0'):        # clear stack
        active_filters.clear()
        print_stack()
    elif key == ord('1'):        # Sobel
        active_filters.append("sobel")
        print_stack()
    elif key == ord('2'):        # Laplacian
        active_filters.append("laplacian")
        print_stack()
    elif key == ord('3'):        # Canny
        active_filters.append("canny")
        print_stack()
    elif key == ord('4'):        # Morph gradient
        active_filters.append("morph_gradient")
    elif key == ord('q'):        # Color map
        active_filters.append("colormap")
        print_stack()
        print_stack()
    elif key == ord('e'):        # Detail enhance
        active_filters.append("detail")
        print_stack()
    elif key == ord('r'):        # Gaussian blur
        active_filters.append("gaussian")
        print_stack()

cap.release()
cv2.destroyAllWindows()
