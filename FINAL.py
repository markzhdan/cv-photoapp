import numpy as np
import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import Frame, Button, Label, Canvas, Scrollbar
from PIL import Image, ImageTk
import threading
import time
import os
import math

from filters import (
    pencil_sketch,
    super_neon_glow_with_gradient,
    pixelated_minecraft_mosaic,
)

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
    elif thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "Thumbs Up"
    elif (
        not thumb_up and not index_up and not middle_up and not ring_up and not pinky_up
    ):
        return "Fist"
    elif thumb_up and index_up and middle_up and ring_up and pinky_up:
        return "Open Hand"
    elif index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
        return "Pointing Gesture"
    else:
        return "Unknown Gesture"


def get_centroid(box):
    x, y, w, h = box
    return x + w // 2, y + h // 2


def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def associate_hands(hand_centroids, face_centroids, threshold):
    hand_face_associations = []
    for hand in hand_centroids:
        closest_face = None
        min_distance = float("inf")
        for face in face_centroids:
            distance = euclidean_distance(hand, face)
            if distance < min_distance and distance < threshold:
                min_distance = distance
                closest_face = face
        if closest_face:
            hand_face_associations.append((hand, closest_face))
    return hand_face_associations


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Photo App")
        self.root.geometry("1600x1000")
        self.face_count = 0
        self.face_count_var = tk.StringVar()
        self.peace_sign_count = 0
        self.peace_sign_count_var = tk.StringVar()
        self.face_count_var.set("0")
        self.peace_sign_count_var.set("0")

        self.draw_detections = True

        self.raw_frame = None

        self.filters = self.apply_filters(self.raw_frame)

        self.button_frame = None

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

        self.size = (1280, 720)

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
        header_frame = Frame(self.root, height=50, bg=background, bd=0, relief="flat")
        header_frame.pack(side="top", fill="x")

        # Quit Button (top right)
        Button(
            header_frame,
            text="❌",
            command=self.quit_app,
            bg="#ff4d4d",
            fg="white",
            font=("Helvetica", 14, "bold"),
            bd=0,
            relief="flat",
        ).pack(side="right", padx=20, pady=10)

        Label(
            header_frame,
            text="Auto Photo App",
            font=("Helvetica", 18, "bold"),
            bg=background,
            relief="flat",
        ).pack(side="left", padx=20, pady=5)

        # Face and Peace Sign Counts
        count_frame = Frame(header_frame, bg=background)
        count_frame.pack(side="top", padx=20, pady=10)

        # Icons and counts
        self.face_icon_label = Label(
            count_frame,
            text="👤",  # Face icon
            font=("Arial", 16),
            bg=background,
            fg="white",
        )
        self.face_icon_label.pack(side="left")

        self.face_count_label = Label(
            count_frame,
            text="0",
            textvariable=self.face_count_var,
            font=("Arial", 16),
            bg=background,
            fg="white",
        )
        self.face_count_label.pack(side="left", padx=2)

        # Spacer
        spacer = Label(
            count_frame,
            text="",
            bg=background,
        )
        spacer.pack(side="left", padx=10)

        self.peace_icon_label = Label(
            count_frame,
            text="✌️",
            font=("Arial", 16),
            bg=background,
            fg="white",
        )
        self.peace_icon_label.pack(side="left", padx=2)

        self.peace_sign_count_label = Label(
            count_frame,
            text="0",
            textvariable=self.peace_sign_count_var,
            font=("Arial", 16),
            bg=background,
            fg="white",
        )
        self.peace_sign_count_label.pack(side="left", padx=2)

        # Main Content
        main_frame = Frame(self.root, bg=background, bd=0, relief="flat")
        main_frame.pack(side="top", fill="both", expand=True)

        # Left Column (Buttons + Scrollable Area)
        left_frame = Frame(main_frame, width=200, bg=background, bd=0, relief="flat")
        left_frame.pack(side="left", fill="y")

        # Camera Button
        Button(
            left_frame,
            text="Camera",
            command=self.switch_to_camera,
            width=15,
            bg=background_light,
            fg=button_fg,
            font=("Helvetica", 12),
            bd=0,  # No border
            relief="flat",  # Flat style
        ).pack(pady=10, padx=10)

        # Scrollable Area for Photos
        self.canvas = Canvas(
            left_frame, bg=background, bd=0, relief="flat", highlightthickness=0
        )
        self.scrollable_frame = Frame(self.canvas, bg=background, bd=0, relief="flat")
        self.scrollbar = Scrollbar(
            left_frame,
            orient="vertical",
            command=self.canvas.yview,
            bg=background,
            bd=0,
            relief="flat",
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
        video_frame = Frame(main_frame, bg=background, bd=0, relief="flat")
        video_frame.pack(side="left", fill="both", expand=True)

        self.video_panel = Label(video_frame, bg=background, bd=0, relief="flat")
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
            bd=0,  # No border
            relief="flat",  # Flat style
        ).pack(pady=10)

    def _on_mouse_wheel(self, event):
        self.canvas.yview_scroll(-1 * int(event.delta / 120), "units")

    def switch_to_camera(self):
        if self.displaying_image:
            self.displaying_image = False
            print("Switched back to camera view.")

            # Remove the filter buttons
            if self.button_frame:
                self.button_frame.destroy()
                self.button_frame = None

    def apply_filters_to_image(self, raw_frame, timestamp, filter_filenames):
        for i, filter_func in enumerate(self.filters):
            filtered_image = filter_func(raw_frame)
            filter_filename = f"photos/photo_{timestamp}_filter_{i+1}.jpg"
            cv2.imwrite(filter_filename, filtered_image)
            filter_filenames.append(filter_filename)
            print(f"Filtered photo saved as {filter_filename}")

        # Update photo paths with filtered images
        print("All filters applied for the photo.")

    def take_picture(self):
        if self.current_frame is not None:
            # Trigger flash
            self.flash_opacity = 255

            # Use the raw frame for saving
            raw_frame = self.raw_frame  # Unannotated frame stored in process_video
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            if not os.path.exists("photos"):
                os.mkdir("photos")

            raw_filename = f"photos/photo_{timestamp}_raw.jpg"
            cv2.imwrite(raw_filename, raw_frame)
            print(f"Raw photo saved as {raw_filename}")

            # Add the photo path to the list (filters will be added later)
            filter_filenames = []
            self.photo_paths.append((raw_filename, filter_filenames))

            # Display the thumbnail in the scrollable area
            self.add_thumbnail((raw_filename, filter_filenames))

            # Start a thread to apply filters
            threading.Thread(
                target=self.apply_filters_to_image,
                args=(raw_frame, timestamp, filter_filenames),
                daemon=True,
            ).start()

    def apply_filters(self, frame):
        def minecraft_mosaic(frame):
            csv_path = "block_data.csv"
            return pixelated_minecraft_mosaic(frame, block_size=32, csv_path=csv_path)

        return [
            pencil_sketch,
            super_neon_glow_with_gradient,
            minecraft_mosaic,
        ]

    def add_thumbnail(self, image_paths):
        raw_path, filtered_paths = (
            image_paths  # Ensure both raw and filtered paths are passed
        )

        # Resize the raw image for the thumbnail
        img = Image.open(raw_path)
        img.thumbnail((200, 200))

        img_tk = ImageTk.PhotoImage(img)

        # Add to frame
        thumbnail_label = Label(
            self.scrollable_frame,
            image=img_tk,
            relief="flat",
        )
        thumbnail_label.image = img_tk  # Keep a reference to avoid garbage collection
        thumbnail_label.pack(pady=5, padx=45)

        # Bind click event to display the full image
        thumbnail_label.bind(
            "<Button-1>", lambda e: self.display_full_image(image_paths)
        )

    def display_full_image(self, image_paths):
        raw_path, filtered_paths = image_paths

        self.displaying_image = True

        def show_image(image_path):
            img = Image.open(image_path)
            img = img.resize((self.size[0], self.size[1]), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            self.video_panel.imgtk = img_tk
            self.video_panel.configure(image=img_tk)

        # Display the raw image by default
        show_image(raw_path)

        filter_names = ["✏️", "✨", "🪵"]

        # Remove any existing button frame
        if self.button_frame:
            self.button_frame.destroy()

        # Create a new button frame for filter buttons
        self.button_frame = Frame(self.video_panel, bg="#0e101a")
        self.button_frame.place(relx=0.5, rely=0.05, anchor="n")

        # Raw Image Button
        Button(
            self.button_frame,
            text="🌄",
            command=lambda: show_image(raw_path),
            bg="#1f1f33",
            fg="black",
            font=("Helvetica", 25),
        ).pack(side="left", padx=5)

        # Filter Buttons
        for filter_name, filter_path in zip(filter_names, filtered_paths):
            Button(
                self.button_frame,
                text=filter_name,
                command=lambda path=filter_path: show_image(path),
                bg="#1f1f33",
                fg="black",
                font=("Helvetica", 25),
            ).pack(side="left", padx=5)

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
        prox_threshold = 700

        while self.running:
            if not self.displaying_image:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Flip the frame horizontally
                frame = cv2.flip(frame, 1)

                # Store the raw frame
                self.raw_frame = frame.copy()

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

                face_centroids = []
                hand_centroids = []

                # Draw face
                if face_results.detections and self.draw_detections:
                    for detection in face_results.detections:
                        mp_drawing.draw_detection(frame, detection)
                        bbox = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x = int(bbox.xmin * iw)
                        y = int(bbox.ymin * ih)
                        w = int(bbox.width * iw)
                        h = int(bbox.height * ih)
                        face_centroids.append(get_centroid((x, y, w, h)))

                # Perform hand detection and update peace sign count
                hand_results = hands.process(rgb_frame)
                new_peace_sign_count = 0

                if hand_results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(
                        hand_results.multi_hand_landmarks
                    ):
                        # Compute hand centroid
                        x = int(
                            hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                            * frame.shape[1]
                        )
                        y = int(
                            hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                            * frame.shape[0]
                        )
                        hand_centroids.append((x, y))

                        hand_face_associations = associate_hands(
                            hand_centroids, face_centroids, prox_threshold
                        )

                        # Draw associations
                        for hand, face in hand_face_associations:
                            cv2.line(frame, hand, face, (50, 255, 0), 2)
                            cv2.putText(
                                frame,
                                "Associated",
                                (hand[0] - 50, hand[1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0),
                                2,
                            )

                        # Draw landmarks
                        if self.draw_detections:
                            mp_drawing.draw_landmarks(
                                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                            )

                        # Get wrist coordinates for positioning text
                        wrist_x = int(
                            hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                            * frame.shape[1]
                        )
                        wrist_y = int(
                            hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                            * frame.shape[0]
                        )

                        # Determine hand type
                        hand_type = (
                            "Left"
                            if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                            < 0.5
                            else "Right"
                        )
                        cv2.putText(
                            frame,
                            f"{hand_type} Hand",
                            (wrist_x - 50, wrist_y - 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )

                        # Recognize gesture
                        try:
                            gesture = recognize_gesture(hand_landmarks)
                            cv2.putText(
                                frame,
                                f"Gesture: {gesture}",
                                (wrist_x - 50, wrist_y - 80),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )
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
