import cv2
import mediapipe as mp
import time
import numpy as np
import mido
from datetime import datetime   # <-- for timestamp


def add_rec_time_overlay(frame, frame_counter):
    """
    Draw CCTV-style timestamp and flashing REC text.
    Uses frame_counter so blink speed tracks your FPS.
    """
    overlay = frame.copy()
    height, width = overlay.shape[:2]

    # Timestamp (top-left)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        overlay,
        timestamp,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Flashing REC (top-right) – toggles about every 0.5s at ~30fps
    if (frame_counter // 15) % 2 == 0:
        cv2.putText(
            overlay,
            "REC",
            (width - 100, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(overlay, (width - 30, 30), 8, (0, 0, 255), -1)

    return overlay


class HandDetector:
    def __init__(self, capture_width=640, capture_height=480,
                 display_width=1920, display_height=1080,
                 max_hands=1, min_detection_confidence=0.3,
                 min_tracking_confidence=0.3,
                 logo_path="../media/logo_clear.png"):
        # -----------------------
        # MIDI SETUP (your code)
        # -----------------------
        try:
            available_ports = mido.get_output_names()
            print(available_ports)
            if available_ports:
                print(f"Available MIDI ports: {available_ports}")
                # Connect to the first available port
                self.midi_out = mido.open_output(available_ports[0])
                print(f"MIDI connected to: {available_ports[0]}")
            else:
                # Create a virtual port if no physical ports are available
                self.midi_out = mido.open_output('HandTracking', virtual=True)
                print("No MIDI ports available. Created virtual port: HandTracking")
            
            print(f"MIDI initialized successfully")
        except Exception as e:
            print(f"MIDI initialization error: {e}")
            self.midi_out = None

        self.hand_present_last_frame = False

        # -----------------------
        # CAMERA
        # -----------------------
        self.cap = cv2.VideoCapture(1) # need be 0
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.capture_width = capture_width
        self.capture_height = capture_height
        
        self.display_width = display_width
        self.display_height = display_height
        
        # -----------------------
        # MEDIAPIPE HANDS
        # -----------------------
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=0,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Hand landmark indices
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_FINGER_TIP = 8
        self.MIDDLE_FINGER_TIP = 12
        self.RING_FINGER_TIP = 16
        self.PINKY_TIP = 20
        
        # FPS calculation
        self.prev_time = 0
        
        cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
        cv2.moveWindow("Hand_Tracking",4000, 0)
        cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        

        # -----------------------
        # PRECOMPUTED SCANLINES
        # -----------------------
        self.scanline_overlay = np.zeros(
            (self.display_height, self.display_width, 3), dtype=np.uint8
        )
        for y in range(0, self.display_height, 20):
            cv2.line(
                self.scanline_overlay,
                (0, y),
                (self.display_width, y),
                (200, 200, 200),
                1
            )

        # -----------------------
        # LOGO LOAD / PREP
        # -----------------------
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if logo is None:
            print(f"Logo not found at {logo_path}")
            self.logo_rgb = None
            self.logo_alpha = None
            self.logo_w = 0
            self.logo_h = 0
        else:
            target_w = 250
            aspect = logo.shape[1] / logo.shape[0]
            target_h = int(target_w / aspect)
            logo = cv2.resize(logo, (target_w, target_h))

            if logo.shape[2] == 4:
                self.logo_alpha = logo[:, :, 3].astype(np.float32) / 255.0
                self.logo_rgb = logo[:, :, :3]
            else:
                self.logo_alpha = np.ones((target_h, target_w), dtype=np.float32)
                self.logo_rgb = logo

            self.logo_h = target_h
            self.logo_w = target_w
    
    def process_frame(self, frame):
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Enhance contrast for better detection
        rgb_frame = cv2.convertScaleAbs(rgb_frame, alpha=1.5, beta=10)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        return results
           
    def find_hand(self, frame, results):
        """Process results and draw landmarks"""
        hand_position = None
        display_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            h, w = display_frame.shape[:2]
            
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(display_frame, (x, y), 3, (255, 255, 255), -1)
            
            connections = [
                (self.WRIST, self.THUMB_TIP),
                (self.WRIST, self.INDEX_FINGER_TIP),
                (self.WRIST, self.MIDDLE_FINGER_TIP),
                (self.WRIST, self.RING_FINGER_TIP),
                (self.WRIST, self.PINKY_TIP)
            ]
            
            for start_idx, end_idx in connections:
                start_point = (
                    int(hand_landmarks.landmark[start_idx].x * w)
                , int(hand_landmarks.landmark[start_idx].y * h)
                )
                end_point = (
                    int(hand_landmarks.landmark[end_idx].x * w)
                , int(hand_landmarks.landmark[end_idx].y * h)
                )
                cv2.line(display_frame, start_point, end_point, (255, 255, 255), 2)
            
            wrist = hand_landmarks.landmark[self.WRIST]
            wrist_x = int(wrist.x * w)
            wrist_y = int(wrist.y * h)
            hand_position = (wrist_x, wrist_y)
        
        display_frame = cv2.resize(display_frame, (self.display_width, self.display_height), 
                                  interpolation=cv2.INTER_LINEAR)
        
        return display_frame, hand_position

    def run(self):
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return
        
        print("Press 'q' to quit")
        
        frame_count = 0
        
        while True:
            frame_count += 1
            
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            frame = cv2.flip(frame, 1)
            
            results = self.process_frame(frame)
            display_frame, hand_position = self.find_hand(frame, results)

            # -----------------------
            # BLACK & WHITE
            # -----------------------
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
            gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # -----------------------
            # SCANLINE OVERLAY
            # -----------------------
            frame_with_overlay = cv2.addWeighted(
                gray3, 1.0,
                self.scanline_overlay, 0.3,
                0
            )

            # -----------------------
            # LOGO OVERLAY
            # -----------------------
            if self.logo_rgb is not None:
                y0 = self.display_height - self.logo_h - 20
                x0 = self.display_width - self.logo_w - 20
                roi = frame_with_overlay[y0:y0+self.logo_h, x0:x0+self.logo_w]

                alpha = self.logo_alpha[..., None]  # (h,w,1)
                roi[:] = (alpha * self.logo_rgb + (1 - alpha) * roi).astype(np.uint8)

            # -----------------------
            # TIMESTAMP + REC OVERLAY
            # -----------------------
            frame_with_overlay = add_rec_time_overlay(frame_with_overlay, frame_count)

            # -----------------------
            # FPS (for logging)
            # -----------------------
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time) if self.prev_time else 0
            self.prev_time = current_time
            
            # -----------------------
            # MIDI (your original logic)
            # -----------------------
            if hand_position and self.midi_out:
                x, y = hand_position
                
                midi_x = int((x / self.capture_width) * 127)
                midi_y = int((1 - (y / self.capture_height)) * 127)
                
                midi_x = max(0, min(127, midi_x))
                midi_y = max(0, min(127, midi_y))
                
                try:
                    self.midi_out.send(mido.Message('control_change', control=1, value=midi_x))
                    self.midi_out.send(mido.Message('control_change', control=2, value=midi_y))
                    
                    if not self.hand_present_last_frame:
                        self.midi_out.send(mido.Message('note_on', note=60, velocity=100))

                        
                        print("Target Engaged - Hand detected")
                        self.hand_present_last_frame = True
                    
                    if frame_count % 10 == 0:
                        print(f"Position Updated - X: {midi_x}, Y: {midi_y}, FPS: {fps:.1f}")
                except Exception as e:
                    print(f"MIDI error: {e}")
            elif self.midi_out:
                if self.hand_present_last_frame:
                    try:
                        self.midi_out.send(mido.Message('note_off', note=60, velocity=0))
                        print("Target Disengaged - Hand lost")
                        print("Goodbye, see you again soon...")
                        print()
                        print()
                        print()
                        
                        print("SURVEIL was developed by M.E.A.T., an Atlanta based collective that explores")
                        print("the intersection of music, community, and technology.")
                        self.hand_present_last_frame = False
                    except Exception as e:
                        print(f"MIDI error: {e}")

            # -----------------------
            # DISPLAY
            # -----------------------
            cv2.imshow("Hand Tracking", frame_with_overlay)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    detector = HandDetector(
        capture_width=640,
        capture_height=480,
        display_width=2560,
        display_height=1440,
        max_hands=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        logo_path="../media_safe/logo_clear.png",
    )
    detector.run()
