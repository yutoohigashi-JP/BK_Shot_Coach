import cv2
import numpy as np
import os

class VideoProcessor:
    def __init__(self, video_path, resize_dim=(640, 480))
        """
        This class preprocess the vieo to prepare for the following procedures
        :param video_path: the path to the video processed
        :param resize_dim: the size of frame resized
        """

        self.video_path = video_path
        self.resize_dim = resize_dim
        self.frame = []
    
    def load_video(self):
        """"
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

# 
