import cv2
from picamera2 import Picamera2
import numpy as np
import time
import os
from datetime import datetime

class CameraApp:
    def __init__(self):
        # Initialize camera
        self.picam2 = Picamera2()
        self.picam2.preview_configuration.main.size = (1280, 720)
        self.picam2.preview_configuration.main.format = "RGB888"
        self.picam2.configure("preview")
        self.picam2.start()

        # Initialize recording variables
        self.is_recording = False
        self.output_video = None
        self.start_time = None

        # Create output directories if they don't exist
        self.image_dir = "captured_images"
        self.video_dir = "captured_videos"
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)

        # Window name
        self.window_name = "Camera App"
        cv2.namedWindow(self.window_name)

    def generate_filename(self, file_type):
        """Generate unique filename based on timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if file_type == "image":
            return os.path.join(self.image_dir, f"image_{timestamp}.jpg")
        else:  # video
            return os.path.join(self.video_dir, f"video_{timestamp}.mp4")

    def draw_ui(self, frame):
        """Draw UI elements on the frame"""
        # Add recording indicator (red circle)
        if self.is_recording:
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            # Add recording duration
            elapsed_time = int(time.time() - self.start_time)
            duration = f"REC {elapsed_time//60:02d}:{elapsed_time%60:02d}"
            cv2.putText(frame, duration, (50, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Add instructions
        instructions = [
            "Press 'r' to start/stop recording",
            "Press 'c' to capture image",
            "Press 'q' to quit"
        ]
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, frame.shape[0] - 20 - (i * 30)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def start_recording(self):
        """Start video recording"""
        filename = self.generate_filename("video")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video = cv2.VideoWriter(
            filename, fourcc, 30.0, (1280, 720))
        self.is_recording = True
        self.start_time = time.time()
        print(f"Started recording: {filename}")

    def stop_recording(self):
        """Stop video recording"""
        if self.output_video is not None:
            self.output_video.release()
            self.is_recording = False
            self.output_video = None
            print("Recording stopped")

    def capture_image(self):
        """Capture a single image"""
        filename = self.generate_filename("image")
        frame = self.picam2.capture_array()
        cv2.imwrite(filename, frame)
        print(f"Image captured: {filename}")

    def run(self):
        """Main application loop"""
        try:
            while True:
                # Capture frame
                frame = self.picam2.capture_array()
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                
                # Create a copy for display (to avoid recording UI elements)
                display_frame = frame.copy()
                
                # Draw UI on display frame
                display_frame = self.draw_ui(display_frame)
                
                # Show frame
                cv2.imshow(self.window_name, display_frame)
                
                # Record frame (without UI elements)
                if self.is_recording and self.output_video is not None:
                    self.output_video.write(frame)
                
                # Handle key presses
                key = cv2.waitKey(25) & 0xFF
                
                if key == ord('q'):
                    if self.is_recording:
                        self.stop_recording()
                    break
                elif key == ord('r'):
                    if not self.is_recording:
                        self.start_recording()
                    else:
                        self.stop_recording()
                elif key == ord('c'):
                    self.capture_image()

        finally:
            # Cleanup
            if self.is_recording:
                self.stop_recording()
            self.picam2.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = CameraApp()
    app.run()