import os
import tkinter as tk
from tkinter import filedialog
from moviepy import VideoFileClip

"""
This module was intended to cut the videos in the selected directory into last 6s
"""

root = tk.Tk()
root.withdraw()
# Select the directory which has videos to be clipped
input_dir = filedialog.askdirectory(title = "Please select the directory to import the videos")
# The directory to which the clipped videos are exported is named automatically, by adding "_last6s" at the end
output_dir = input_dir + "_last6s"

# Clear all of the files in the output directory if the directory exists
# Otherwise a new directory is created

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
#     for item in os.listdir(output_dir):
#         item_path = os.path.join(output_dir, item)
#         os.remove(item_path)
    

video_extensions = (".mp4")

# Clip all the videos in the input directory into the last 6s
# Export all the clipped videos to the output directory
for file_name in os.listdir(input_dir):
    if file_name.endswith(video_extensions):
        output_path = os.path.join(output_dir, "last6s_"+file_name)
        if not os.path.exists(output_path):
            video_path = os.path.join(input_dir, file_name)
            with VideoFileClip(video_path) as video:
                duration = video.duration # The whole time of the video(s)
                start_time = max(duration - 7, 0)
                clip = video.subclipped(start_time, duration - 1)
                output_path = os.path.join(output_dir, f"last6s_{file_name}")
                clip.write_videofile(output_path, codec="libx264", audio_codec="aac")