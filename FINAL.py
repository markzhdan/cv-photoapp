import numpy as np
import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import Frame, Button, Label, Canvas, Scrollbar
from PIL import Image, ImageTk
import threading
import time
import os

# Initialize MediaPipe hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Mediapipe face and drawing utilizites
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


# Function to recognize basic gestures
def recognize_gesture(hand_landmarks):
    # Finger states: True if finger is extended
    thumb_up = False
    index_up = False
    middle_up = False
    ring_up = False
    pinky_up = False

    # Extract key landmarks
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Determine finger states
    thumb_up = thumb_tip.x < thumb_ip.x  # Thumb extended if tip is left of IP joint
    index_up = index_tip.y < index_dip.y  # Index extended if tip is above DIP joint
    middle_up = middle_tip.y < middle_dip.y
    ring_up = ring_tip.y < ring_dip.y
    pinky_up = pinky_tip.y < pinky_dip.y

    if index_up and middle_up and not ring_up and not pinky_up and not thumb_up:
        return "Peace Sign"
    else:
        return "Unknown Gesture"


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Photo App")
        self.root.geometry("800x500")
        self.face_count = 0
        self.face_count_var = tk.StringVar()
        self.peace_sign_count = 0
        self.peace_sign_count_var = tk.StringVar()

        # Match tracking variables
        self.match_start_time = None  # When the match started
        self.match_duration_required = 1  # Time in seconds to confirm match

        # Flash effect
        self.flash_opacity = 0
        self.flash_decay_rate = 10

        # Video capture and processing
        self.cap = cv2.VideoCapture(0)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.size = (640, 360)

        # Flags and data
        self.running = True
        self.current_frame = None  # Store the current frame for capturing
        self.photo_paths = []  # List to store captured photos
        self.displaying_image = False  # Tracks whether to display the video feed

        # Create UI elements
        self.create_widgets()

        # Start video processing in a separate thread
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.daemon = True
        self.video_thread.start()

    def create_widgets(self):
        # Colors
        background = "#0e101a"
        background_light = "#14142d"
        button_color = "#1f1f33"
        button_fg = "black"

        # Header
        header_frame = Frame(self.root, height=50, bg=background)
        header_frame.pack(side="top", fill="x")

        # Header content
        Label(
            header_frame,
            text="Auto Photo App",
            font=("Helvetica", 18, "bold"),
            bg=background,
            fg="white",
        ).pack(side="left", padx=20, pady=5)

        # Icons for face and peace sign counts
        self.face_icon_label = Label(
            header_frame,
            text="👤",  # Face icon
            font=("Arial", 16),
            bg=background,
            fg="white",
        )
        self.face_icon_label.pack(side="left", padx=10)

        self.face_count_var.set(f"{self.face_count}")
        self.face_count_label = Label(
            header_frame,
            textvariable=self.face_count_var,
            font=("Arial", 16),
            bg=background,
            fg="white",
        )
        self.face_count_label.pack(side="left")

        self.peace_icon_label = Label(
            header_frame,
            text="✌️",  # Peace sign icon
            font=("Arial", 16),
            bg=background,
            fg="white",
        )
        self.peace_icon_label.pack(side="left", padx=10)

        self.peace_sign_count_var.set(f"{self.peace_sign_count}")
        self.peace_sign_count_label = Label(
            header_frame,
            textvariable=self.peace_sign_count_var,
            font=("Arial", 16),
            bg=background,
            fg="white",
        )
        self.peace_sign_count_label.pack(side="left")

        # Main Content
        main_frame = Frame(self.root)
        main_frame.pack(side="top", fill="both", expand=True)

        # Left Column (Buttons + Scrollable Area)
        left_frame = Frame(main_frame, width=200, bg=background)
        left_frame.pack(side="left", fill="y")

        # Camera Button
        Button(
            left_frame,
            text="Camera",
            command=self.switch_to_camera,
            width=20,
            bg=background_light,
            fg=button_fg,
            font=("Helvetica", 12),
        ).pack(pady=10, padx=10)

        # Scrollable Area for Photos
        self.canvas = Canvas(left_frame, bg=background)
        self.scrollable_frame = Frame(self.canvas, bg=background)
        self.scrollbar = Scrollbar(
            left_frame, orient="vertical", command=self.canvas.yview, bg=background
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Enable mouse wheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mouse_wheel)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        # Right Column (Video Feed)
        video_frame = Frame(main_frame, bg=background)
        video_frame.pack(side="left", fill="both", expand=True)

        self.video_panel = Label(video_frame, bg=background)
        self.video_panel.pack(fill="both", expand=True)

        # Take Picture Button
        Button(
            video_frame,
            text="Take Picture",
            command=self.take_picture,
            width=15,
            bg=button_color,
            fg=button_fg,
            font=("Helvetica", 12),
        ).pack(pady=10)

    def _on_mouse_wheel(self, event):
        self.canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    def switch_to_camera(self):
        if self.displaying_image:
            self.displaying_image = False
            print("Switched back to camera view.")

    def take_picture(self):
        if self.current_frame is not None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            if not os.path.exists("photos"):
                os.mkdir("photos")

            filename = f"photos/photo_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            print(f"Picture saved as {filename}")

            # Add the photo path to the list
            self.photo_paths.append(filename)

            # Display the thumbnail in the scrollable area
            self.add_thumbnail(filename)

    def add_thumbnail(self, image_path):
        # Resize image to thumbnail size
        img = Image.open(image_path)
        img.thumbnail((150, 150))

        img_tk = ImageTk.PhotoImage(img)

        # Add to frame
        thumbnail_label = Label(
            self.scrollable_frame,
            image=img_tk,
            relief="flat",
        )
        thumbnail_label.image = img_tk  # Keep a reference to avoid garbage collection
        thumbnail_label.pack(pady=5, padx=5)

        # Bind click event to display the full image
        thumbnail_label.bind(
            "<Button-1>", lambda e: self.display_full_image(image_path)
        )

    def display_full_image(self, image_path):
        self.displaying_image = True
        img = Image.open(image_path)
        img = img.resize((self.size[0], self.size[1]), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        self.video_panel.imgtk = img_tk
        self.video_panel.configure(image=img_tk)
        print(f"Displaying full image: {image_path}")

    def quit_app(self):
        self.running = False
        self.cap.release()
        self.root.destroy()
        cv2.destroyAllWindows()

    def process_video(self):
        hands = mp_hands.Hands(
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        )
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.8)

        while self.running:
            if not self.displaying_image:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Flip the frame horizontally
                frame = cv2.flip(frame, 1)

                # Convert the frame to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Perform face detection
                face_results = face_detection.process(rgb_frame)

                # Update face count
                new_face_count = (
                    len(face_results.detections) if face_results.detections else 0
                )
                if new_face_count != self.face_count:
                    self.face_count = new_face_count
                    self.face_count_var.set(f"{self.face_count}")

                # Perform hand detection and update peace sign count
                hand_results = hands.process(rgb_frame)
                new_peace_sign_count = 0

                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        try:
                            gesture = recognize_gesture(hand_landmarks)
                            if gesture == "Peace Sign":
                                new_peace_sign_count += 1
                        except AttributeError as e:
                            print("Error recognizing gesture:", e)

                if new_peace_sign_count != self.peace_sign_count:
                    self.peace_sign_count = new_peace_sign_count
                    self.peace_sign_count_var.set(f"{self.peace_sign_count}")

                # Auto take photo
                if self.peace_sign_count == self.face_count and self.face_count > 0:
                    if self.match_start_time is None:
                        # Start match timer
                        self.match_start_time = time.time()
                    elif (
                        time.time() - self.match_start_time
                        >= self.match_duration_required
                    ):
                        # Face and peace sign counts match for required duration
                        self.take_picture()
                        self.match_start_time = None
                        # Trigger flash
                        self.flash_opacity = 255
                else:
                    self.match_start_time = None

                # Flash
                if self.flash_opacity > 0:
                    overlay = np.ones_like(frame, dtype=np.uint8) * 255  # White overlay
                    overlay = cv2.addWeighted(
                        overlay,
                        self.flash_opacity / 255,
                        frame,
                        1 - self.flash_opacity / 255,
                        0,
                    )
                    frame = overlay
                    self.flash_opacity -= (
                        self.flash_decay_rate
                    )  # Reduce opacity for fade-out

                # Store the current frame for capturing
                self.current_frame = frame.copy()

                # Resize and convert frame to tkinter-compatible format
                frame = cv2.resize(frame, self.size)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)

                # Update video panel
                self.video_panel.imgtk = imgtk
                self.video_panel.configure(image=imgtk)


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
