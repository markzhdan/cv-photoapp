import numpy as np
import cv2
import tkinter as tk
from tkinter import Frame, Button, Label, Canvas, Scrollbar
from PIL import Image, ImageTk
import threading
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
import time


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Auto Photo App")
        self.root.geometry("800x500")

        # Initialize MediaPipe Hands
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
            max_num_hands=2,
        )
        self.mpDraw = mp.solutions.drawing_utils

        # Video capture setup
        self.cap = cv2.VideoCapture(0)
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

        Label(
            header_frame,
            text="Auto Photo App",
            font=("Helvetica", 18, "bold"),
            bg=background,
            fg="white",
        ).pack(side="left", padx=20)

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
            filename = f"photo_{int(time.time())}.jpg"
            cv2.imwrite(filename, cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))
            print(f"Picture saved as {filename}")

            self.photo_paths.append(filename)
            self.add_thumbnail(filename)

    def add_thumbnail(self, image_path):
        img = Image.open(image_path)
        img.thumbnail((150, 150))

        img_tk = ImageTk.PhotoImage(img)

        thumbnail_label = Label(
            self.scrollable_frame,
            image=img_tk,
            relief="flat",
        )
        thumbnail_label.image = img_tk
        thumbnail_label.pack(pady=5, padx=5)

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

    def process_video(self):
        while self.running:
            if not self.displaying_image:
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Flip the frame horizontally for a mirrored view
                frame = cv2.flip(frame, 1)

                # Convert BGR to RGB for MediaPipe
                imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame with MediaPipe
                results = self.hands.process(imgRGB)

                # Draw hand landmarks and annotate right/left hand
                if results.multi_hand_landmarks:
                    for idx, hand_handedness in enumerate(results.multi_handedness):
                        label = MessageToDict(hand_handedness)["classification"][0][
                            "label"
                        ]
                        hand_landmarks = results.multi_hand_landmarks[idx]

                        # Draw landmarks
                        self.mpDraw.draw_landmarks(
                            frame, hand_landmarks, self.mpHands.HAND_CONNECTIONS
                        )

                        # Annotate hand label
                        if label == "Left":
                            cv2.putText(
                                frame,
                                "Left Hand",
                                (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )
                        elif label == "Right":
                            cv2.putText(
                                frame,
                                "Right Hand",
                                (460, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )

                # Convert the frame to RGB for display in tkinter
                frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frameRGB)
                imgtk = ImageTk.PhotoImage(image=img)

                self.video_panel.imgtk = imgtk
                self.video_panel.configure(image=imgtk)

    def quit_app(self):
        self.running = False
        self.cap.release()
        self.root.destroy()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
