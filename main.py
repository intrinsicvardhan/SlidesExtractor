import os
import cv2
# from googleapiclient.discovery import build
# from subprocess import Popen, PIPE
import numpy as np
from urllib.parse import urlparse, parse_qs
# from iso8601 import parse_date
from pytube import YouTube

# Import is_blank_frame from extract_frames module
from extract_frames import is_blank_frame
from extract_slides_ocr import detect_slide_transition

API_KEY = ''
VIDEO_ID = 'fM5angbo2JI'  # Remove the additional parameters from the URL
VIDEO_URL = f'https://www.youtube.com/watch?v={VIDEO_ID}'
INTERVAL = 20

if not os.path.exists('slides'):
    os.makedirs('slides')


def extract_slides(video_path, interval):
    slide_count = 0
    prev_frame = None
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the frame is not empty and normalize it
        if frame is not None:
            frame = frame / 255.0  # Normalize the frame to the range [0, 1]

        # Check if the frame is not blank
        if not is_blank_frame(frame) and (prev_frame is None or detect_slide_transition(prev_frame, frame)):
            cv2.imwrite(os.path.join('slides', f'frame_{slide_count}.jpg'), (frame * 255).astype(np.uint8))
            print(f'Saved frame_{slide_count}.jpg')
            slide_count += 1

        prev_frame = frame

    cap.release()

try:
    yt = YouTube(VIDEO_URL)
    stream = yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first()

    video_file_path = stream.download(output_path='.', filename='video.mp4')

    extract_slides(video_file_path, INTERVAL)

except Exception as e:
    print("An error occurred:", str(e))