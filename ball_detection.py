import cv2
import numpy as np
import os
from ultralytics import YOLO

class VideoProcessor:
    def __init__(self, video_path, resize_dim=(1280, 720)):
        """
        This class preprocess the vieo to prepare for the following procedures
        :param video_path: the path to the video processed
        :param resize_dim: the size of frame resized
        """

        self.video_path = video_path
        self.resize_dim = resize_dim
        self.frames = [] # list to save each frame processed
    
    def load_video(self):
        """
        This method loads the video and process each frame of it
        """

        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"The video file was not found: {self.video_path}")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("The video could not open")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break #The end of the video

            processed_frame = self.preprocess_frame(frame)
            self.frames.append(processed_frame)

        cap.release()
        print(f"{len(self.frames)} frames were loaded")

    def preprocess_frame(self, frame):
        """
        This method resizes and normalizes the given frame
        :param frame: the frame input
        :return: processed frame
        """

        frame_resized = cv2.resize(frame, self.resize_dim)
        frame_normalized = frame_resized / 255.0 # normalized to 0-1
        return frame_normalized
    
    def get_frames(self):
        """
        This method obtains the frames loaded
        :return: the list of the frames
        """
        
        return self.frames


class BallDetector:
    def __init__(self, model_path = 'yolov8x.pt'):
        """
        This class detects the basketball on the frame
        :param model_path: the path of model of YOLO
        """

        self.model = YOLO(model_path)
    
    def detect_ball(self, frame):
        """
        This method detect the basketball using YOLO model
        :param frame: the frame used
        :return: the coordination of the ball
        """

        results = self.model(frame, conf = 0.2)
        for result in results:
            for box in result.boxes:
                if box.cls == 32: # Class ID 32 indicates sports ball
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    return int(x), int(y), int(w), int(h)
        return None
    
    def visualize_detections(self, frame, bbox):
        """
        This method draws the result of the detection on the frame
        :param frame: the frame used and drawn
        :param bbox: the coordination of the ball(x, y, w, h)
        :return: the frame with visualization of the ball
        """

        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
        return frame
    
# Operation Confirmation
import tkinter as tk
from tkinter import filedialog

if __name__ == "__main__":
    # root = tk.Tk()
    # root.withdraw()

    # video_path = filedialog.askopenfilename(
    # title="動画ファイルを選択してください",
    # filetypes=[("動画ファイル", "*.mp4 *.avi *.mov"), ("すべてのファイル", "*.*")]
    # )

    video_path = "3-Pointer/GSW_7_last6s/last6s_J_made_1.mp4"

    processor = VideoProcessor(video_path)
    processor.load_video()
    frames = processor.get_frames()
    print(f"The number of frames processed: {len(frames)}")

    detector = BallDetector()
    for i, frame in enumerate(frames):
        bbox = detector.detect_ball(frame)
        print("Detected bbox:", bbox)
        visualized_frame = detector.visualize_detections(frame, bbox)
        cv2.imshow("Ball Detection", visualized_frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()