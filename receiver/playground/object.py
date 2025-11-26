import random
import cv2
import numpy as np
import threading
import time
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import os

# -----------------------------
# GLOBAL BPM + SPEED
# -----------------------------
current_bpm = 100.0
current_speed = 1.0

# Active filter stack (in order applied)
active_filters = []  # e.g. ["gaussian", "sobel", "colormap"]

# Object detection toggle
object_detection_enabled = False

# Folder where *this script* lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the path relative to the script location
video_path = os.path.join(BASE_DIR, "..", "..", "media", "test.mp4")

# -----------------------------
# YOLO (GENERAL OBJECT DETECTION)
# -----------------------------
YOLO_DIR = os.path.join(BASE_DIR, "yolo")
YOLO_CONFIG = os.path.join(YOLO_DIR, "yolov3.cfg")
YOLO_WEIGHTS = os.path.join(YOLO_DIR, "yolov3.weights")
YOLO_NAMES = os.path.join(YOLO_DIR, "coco.names")

# Load class names
with open(YOLO_NAMES, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load network
net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG, YOLO_WEIGHTS)
# (Optional: use CPU explicitly; comment these if you have CUDA and want to use it)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

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

# osc_thread = threading.Thread(target=start_osc, daemon=True)
# osc_thread.start()

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
# GENERAL OBJECT DETECTION (YOLO)
# -----------------------------
def apply_object_detection(frame_display, frame_for_detection):
    """
    Run YOLO on frame_for_detection, draw boxes on frame_display.
    This lets you detect on 'clean' frames but draw over filtered ones.
    """
    h, w = frame_for_detection.shape[:2]

    # Create input blob
    blob = cv2.dnn.blobFromImage(
        frame_for_detection, 1/255.0, (416, 416),
        swapRB=True, crop=False
    )
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    conf_threshold = 0.5
    nms_threshold = 0.4

    # Parse YOLO outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)
                x = int(center_x - bw / 2)
                y = int(center_y - bh / 2)

                boxes.append([x, y, bw, bh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    detected_count = 0
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            label = classes[class_ids[i]] if class_ids[i] < len(classes) else str(class_ids[i])
            conf = confidences[i]

            # Draw box + label on the display frame
            cv2.rectangle(frame_display, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(frame_display, text, (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            detected_count += 1

    cv2.putText(
        frame_display,
        f"Detections: {detected_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return frame_display

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
print("  o: toggle YOLO object detection (general objects)")

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

            # Keep a clean copy for detection
            frame_clean = frame.copy()

            # Apply filter stack
            frame_filtered = apply_filter_stack(frame, active_filters)

            # Apply object detection overlay if enabled
            if object_detection_enabled:
                frame_filtered = apply_object_detection(frame_filtered, frame_clean)

            frozen_frame = frame_filtered
            cv2.imshow("Video Filters (Stacking + YOLO Detection)", frame_filtered)

    else:
        if frozen_frame is not None:
            cv2.imshow("Video Filters (Stacking + YOLO Detection)", frozen_frame)

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
        print_stack()
    elif key == ord('q'):        # Color map
        active_filters.append("colormap")
        print_stack()
    elif key == ord('e'):        # Detail enhance
        active_filters.append("detail")
        print_stack()
    elif key == ord('r'):        # Gaussian blur
        active_filters.append("gaussian")
        print_stack()
    elif key == ord('o'):        # Toggle object detection
        object_detection_enabled = not object_detection_enabled
        print(f"YOLO object detection: {'ON' if object_detection_enabled else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
