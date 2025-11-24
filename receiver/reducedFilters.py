import cv2
import numpy as np
import threading
import time
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import random

# -----------------------------
# GLOBAL BPM + SPEED + FLUX
# -----------------------------
current_bpm = 100.0
current_speed = 1.0

current_flux = 0        # store last flux value

# Active filter stack (in order applied)
active_filters = []  # e.g. ["gaussian", "sobel", "sharpen"]

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
# OSC CALLBACK FOR FLUX
# -----------------------------
def got_flux(addr, flux_value):
    global current_flux
    try:
        flux_value = int(flux_value)
    except:
        return

    current_flux = flux_value
    print(f"FLUX RECEIVED: {current_flux}")

# -----------------------------
# START OSC RECEIVER THREAD
# -----------------------------
def start_osc():
    dispatcher = Dispatcher()
    dispatcher.map("/bpm", got_bpm)
    dispatcher.map("/flux", got_flux)

    server = BlockingOSCUDPServer(("0.0.0.0", 9000), dispatcher)
    print("OSC Receiver running on port 9000... (listening for /bpm and /flux)")
    server.serve_forever()

osc_thread = threading.Thread(target=start_osc, daemon=True)
osc_thread.start()

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
    COLORMAPS = [
    cv2.COLORMAP_AUTUMN,
    cv2.COLORMAP_BONE,
    cv2.COLORMAP_JET,
    cv2.COLORMAP_WINTER,
    cv2.COLORMAP_RAINBOW,
    cv2.COLORMAP_OCEAN,
    cv2.COLORMAP_SUMMER,
    cv2.COLORMAP_SPRING,
    cv2.COLORMAP_COOL,
    cv2.COLORMAP_HSV,
    cv2.COLORMAP_PINK,
    cv2.COLORMAP_HOT,
    cv2.COLORMAP_PARULA,
    cv2.COLORMAP_MAGMA,
    cv2.COLORMAP_INFERNO,
    cv2.COLORMAP_PLASMA,
    cv2.COLORMAP_VIRIDIS,
    cv2.COLORMAP_CIVIDIS,
    cv2.COLORMAP_TWILIGHT,
    cv2.COLORMAP_TWILIGHT_SHIFTED,
    cv2.COLORMAP_TURBO,
    cv2.COLORMAP_DEEPGREEN,
]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, COLORMAPS[random.randint(0, len(COLORMAPS) - 1)])

def apply_sharpen(frame):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(frame, -1, kernel)

# -----------------------------
# MAP FLUX → LAPLACIAN STACK COUNT
# -----------------------------
def laplacian_count_from_flux(flux_int: int) -> int:
    if flux_int < 10:
        return 1
    elif flux_int < 20:
        return 3
    else:
        return 5   # high flux = 5 passes

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

    # 1) Apply all filters except 'laplacian' and 'colormap'
    base_filters = [f for f in filters if f not in ("laplacian", "colormap")]
    for f_name in base_filters:
        out = apply_filter_by_name(out, f_name)

    # 2) Apply Laplacian N times based on current flux
    lap_count = laplacian_count_from_flux(current_flux)
    for _ in range(lap_count):
        out = apply_laplacian(out)

    # 3) ALWAYS apply colormap last
    out = apply_colormap(out)

    # 4) Label: show base stack + flux + lap count + note colormap
    if base_filters:
        stack_str = " -> ".join(base_filters)
    else:
        stack_str = "none"

    label = f"Stack: {stack_str} + colormap | flux={current_flux} | laps={lap_count}"

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
cap = cv2.VideoCapture("../media/test.mp4")

if not cap.isOpened():
    print("ERROR: Could not open test.mp4")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0
frame_time = 1.0 / fps

print("Playing video... Press Q to quit.")
print("Filter keys (stacks in order of presses):")
print("  0: clear stack (no filters)")
print("  1: add Sobel")
print("  2: add Scharr")
print("  3: (Laplacian is flux-controlled now)")
print("  4: add Canny")
print("  5: add Gaussian blur")
print("  6: add Median blur")
print("  7: add Bilateral filter")
print("  8: add Stylization")
print("  9: add Pencil sketch")
print("  d: add Detail enhance")
print("  e: add Edge-preserving")
print("  g: add Morph gradient")
print("  c: (colormap is always ON, last)")
print("  h: add Sharpen")

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

            frame_filtered = apply_filter_stack(frame, active_filters)

            frozen_frame = frame_filtered
            cv2.imshow("Video Filters (Stacking)", frame_filtered)

    else:
        if frozen_frame is not None:
            cv2.imshow("Video Filters (Stacking)", frozen_frame)

    # KEY CONTROLS
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('0'):
        active_filters.clear()
        print_stack()
    elif key == ord('1'):
        active_filters.append("sobel")
        print_stack()
    elif key == ord('2'):
        active_filters.append("scharr")
        print_stack()
    # '3' no longer manually adds laplacian – it's flux-driven
    elif key == ord('4'):
        active_filters.append("canny")
        print_stack()
    elif key == ord('5'):
        active_filters.append("gaussian")
        print_stack()
    elif key == ord('6'):
        active_filters.append("median")
        print_stack()
    elif key == ord('7'):
        active_filters.append("bilateral")
        print_stack()
    elif key == ord('8'):
        active_filters.append("stylization")
        print_stack()
    elif key == ord('9'):
        active_filters.append("pencil")
        print_stack()
    elif key == ord('d'):
        active_filters.append("detail")
        print_stack()
    elif key == ord('e'):
        active_filters.append("edge_preserve")
        print_stack()
    elif key == ord('g'):
        active_filters.append("morph_gradient")
        print_stack()
    elif key == ord('c'):
        # colormap is always on, so just log it
        print("Colormap is always ON and applied last.")
    elif key == ord('h'):
        active_filters.append("sharpen")
        print_stack()
    
cap.release()
cv2.destroyAllWindows()
