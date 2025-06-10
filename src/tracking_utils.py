"""
video_tracking_utils.py
------------------
Utility functions for processing video and tracking a colored object by HSV bounds.
"""

import cv2
import numpy as np

def track_colored_object(frame, lower_hsv, upper_hsv, min_contour_area):
    """
    Detects and tracks the largest object within the specified HSV range.

    Args:
        frame (np.ndarray): Input video frame (BGR format).
        lower_hsv (np.ndarray): Lower HSV threshold for object color.
        upper_hsv (np.ndarray): Upper HSV threshold for object color.
        min_contour_area (int): Minimum area to accept a contour as a valid object.

    Returns:
        tuple: (processed_frame, (cX, cY), object_found)
            - processed_frame (np.ndarray): Frame with drawn contour and cross if object is found.
            - (cX, cY) (tuple[int, int] or None): Coordinates of the object's centroid.
            - object_found (bool): Whether a valid object was detected.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return frame, None, False

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) <= min_contour_area:
        return frame, None, False

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return frame, None, False

    cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    cv2.drawContours(frame, [largest], -1, (0, 0, 0), 2)
    cv2.line(frame, (cX - 20, cY), (cX + 20, cY), (0, 255, 0), 2)
    cv2.line(frame, (cX, cY - 20), (cX, cY + 20), (0, 255, 0), 2)

    return frame, (cX, cY), True

def process_video(video_path, output_path, lower_hsv, upper_hsv, min_area, rotate):
    """
    Processes a video, tracks a colored object, saves the annotated output video,
    and returns time-series data of object positions.

    Args:
        video_path (str): Path to the input video.
        output_path (str): Path to save the processed video.
        lower_hsv (np.ndarray): Lower HSV threshold for object color.
        upper_hsv (np.ndarray): Upper HSV threshold for object color.
        min_area (int): Minimum area for valid object contour.
        rotate (bool): Whether to rotate frames 90° clockwise before processing.

    Returns:
        tuple: (times, x_positions, y_positions)
            - times (list[float]): Time (in seconds) of each detected object.
            - x_positions (list[int]): X coordinates of the object.
            - y_positions (list[int]): Y coordinates of the object.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_dim = (height, width) if rotate else (width, height)
    new_w = 640
    new_h = int(out_dim[1] * (new_w / out_dim[0]))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (new_w, new_h))

    times, xs, ys = [], [], []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        t_sec = frame_count / fps
        processed, pos, valid = track_colored_object(frame, lower_hsv, upper_hsv, min_area)
        resized = cv2.resize(processed, (new_w, new_h))
        out.write(resized)

        if valid:
            times.append(t_sec)
            xs.append(pos[0])
            ys.append(pos[1])

        frame_count += 1

    cap.release()
    out.release()
    return times, xs, ys
