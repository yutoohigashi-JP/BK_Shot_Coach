import cv2
import os

class Clip:
    def __init__(self, video_dir, video_name, output_dir):
        """
        This class is created to clip the video given into frames
        """
        # Select the video to be clipped to frame
        self.video_dir = video_dir
        self.video_name = video_name
        self.output_dir = output_dir


    def clip_video(self):
        # Designate the directory to which the frames are exported
        input_video = os.path.join(self.video_dir, self.video_name)
        output_folder = self.output_dir

        cap = cv2.VideoCapture(input_video)

        frame_total = 30
        interval = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)/frame_total)

        frame_count = 0
        frame_saved = 0
        frame_index = int(input("Designate the index to begin with"))
        while cap.isOpened() and frame_saved <= frame_total:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % interval == 0:
                cv2.imwrite(f"{output_folder}/ball_frame_{frame_index}.jpg", frame)
                frame_index += 1
                frame_saved += 1
            frame_count += 1

        cap.release()

if __name__ == "__main__":
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()

    video_dir = filedialog.askdirectory(title="Select the directory that has videos to be clipped")
    output_dir = filedialog.askdirectory(title="Select the directory to which the frames are exported")

    files = os.listdir("yolo_relearn/materials_last6s")
    for file in files:
        clip_instance = Clip(video_dir, file, output_dir)
        clip_instance.clip_video()
