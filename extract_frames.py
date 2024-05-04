import cv2

import numpy as np 
#Load the frames of the yt video directory

# def did_frame_transition(prev_frame, curr_frame, threshold=0.8):
#     '''
#     Check if the percentage of content on the frame has changed significantly
#     '''
#     if prev_frame is None or curr_frame is None:
#         return False

#     # Normalize frames to the range [0, 1]
#     prev_gray = prev_frame / 255.0
#     curr_gray = curr_frame / 255.0

#     # Check if frame sizes match
#     if prev_gray.shape != curr_gray.shape:
#         raise ValueError("Frame sizes do not match.")

#     # Calculate SSIM
#     (score, _) = cv2.ssim(prev_gray, curr_gray, full=True)

#     # Check for invalid SSIM values
#     if not (0 <= score <= 1):
#         raise ValueError("Invalid SSIM value.")

#     return score < threshold

def is_blank_frame(frame):
    """
    Check if a frame is blank (i.e., contains only zeros or a single color)

    Args:
        frame (numpy array): The frame to check

    Returns:
        bool: True if the frame is blank, False otherwise
    """
    # Calculate the standard deviation of the frame's pixel values
    std_dev = np.std(frame)

    # If the standard deviation is very low (e.g., < 10), the frame is likely blank
    return std_dev < 10



def did_frame_transition(prev_frame, curr_frame, threshold=10):
    '''
    Check if the textual content on the frame has changed significantly
    '''
    if prev_frame is None or curr_frame is None:
        return False

    # Normalize frames to the range [0, 1]
    prev_gray = prev_frame / 255.0
    curr_gray = curr_frame / 255.0

    # Check if frame sizes match
    if prev_gray.shape != curr_gray.shape:
        raise ValueError("Frame sizes do not match.")

    # Calculate Mean Absolute Difference (MAD)
    mad = np.mean(np.abs(prev_gray - curr_gray))

    return mad > threshold




def extract_frames(frames_dir, output_dir):
    slide_count = 0
    prev_frame = None
    while True: 
        frame = cv2.imread(frames_dir + f'frame_{slide_count}.jpg')
        if frame is None:
            break
        if not is_blank_frame(prev_frame, frame):
            cv2.imwrite(output_dir + f'frame_{slide_count}.jpg', frame)
            print(f'')
        prev_frame = frame
        slide_count += 1