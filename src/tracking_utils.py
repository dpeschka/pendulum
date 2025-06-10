"""
tracking_utils.py
------------------
Provides utility functions for tracking colored objects in video frames
using HSV color thresholds, and an optional interactive widget for HSV tuning.
"""

import cv2
import numpy as np

# === Core functionality ===

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

# === Optional interactive widget interface ===
def launch_hsv_tuning_widget(video_path="INPUT.MOV", max_frames_to_load=100):
    """
    Launches an interactive widget interface to test and adjust HSV bounds
    and minimum area threshold for object detection. Displays the selected
    frame with annotations and a hue gradient bar for visual reference.
    """
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display

    def generate_hue_bar(width=360, height=30):
        """Creates a horizontal hue gradient bar from HSV to RGB."""
        hue = np.linspace(0, 180, width, dtype=np.uint8)
        hsv = np.zeros((height, width, 3), dtype=np.uint8)
        hsv[..., 0] = hue[None, :]
        hsv[..., 1] = 255
        hsv[..., 2] = 255
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb

    cap = cv2.VideoCapture(video_path)
    frames_bgr = []
    while len(frames_bgr) < max_frames_to_load:
        ret, frame = cap.read()
        if not ret:
            break
        frames_bgr.append(frame)
    cap.release()

    if not frames_bgr:
        print(f"Could not read frames from {video_path}")
        return

    # Widget controls
    lower_h = widgets.IntSlider(value=140, min=0, max=180, description='Lower H')
    lower_s = widgets.IntSlider(value=50, min=0, max=255, description='Lower S')
    lower_v = widgets.IntSlider(value=50, min=0, max=255, description='Lower V')
    upper_h = widgets.IntSlider(value=180, min=0, max=180, description='Upper H')
    upper_s = widgets.IntSlider(value=255, min=0, max=255, description='Upper S')
    upper_v = widgets.IntSlider(value=255, min=0, max=255, description='Upper V')
    min_area_slider = widgets.IntSlider(value=1500, min=0, max=5000, step=100, description='Min Area:')
    frame_idx_slider = widgets.IntSlider(value=0, min=0, max=len(frames_bgr)-1, description='Frame:')

    lb_label = widgets.Label()
    ub_label = widgets.Label()
    output = widgets.Output()

    hue_bar_img = generate_hue_bar()

    def update_display(lower_h, lower_s, lower_v,
                       upper_h, upper_s, upper_v,
                       min_contour_area, frame_index):
        lower = np.minimum([lower_h, lower_s, lower_v], [upper_h, upper_s, upper_v])
        upper = np.maximum([lower_h, lower_s, lower_v], [upper_h, upper_s, upper_v])
        lb_label.value = f"Lower HSV: {lower}"
        ub_label.value = f"Upper HSV: {upper}"

        frame = frames_bgr[frame_index].copy()
        processed, _, _ = track_colored_object(frame, np.array(lower), np.array(upper), min_contour_area)
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        with output:
            output.clear_output(wait=True)
            fig, axs = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [4, 1]})

            # Image with annotations
            axs[0].imshow(rgb)
            axs[0].set_title(f"Frame {frame_index}")
            axs[0].axis('off')

            # Hue bar with ticks
            axs[1].imshow(hue_bar_img, aspect='auto')
            axs[1].set_title("Hue values (OpenCV scale 0–180)")
            axs[1].set_yticks([])

            ticks = np.linspace(0, hue_bar_img.shape[1] - 1, 10, dtype=int)
            tick_labels = np.linspace(0, 180, 10, dtype=int)
            axs[1].set_xticks(ticks)
            axs[1].set_xticklabels(tick_labels)

            # Optional: show selected hue range
            axs[1].axvline(lower[0] * (hue_bar_img.shape[1] / 180), color='black', linestyle='--')
            axs[1].axvline(upper[0] * (hue_bar_img.shape[1] / 180), color='black', linestyle='--')
            plt.xlim([0,180])
            plt.tight_layout()
            plt.show()

    ui = widgets.VBox([
        lb_label,
        ub_label,
        widgets.HBox([lower_h, lower_s, lower_v]),
        widgets.HBox([upper_h, upper_s, upper_v]),
        min_area_slider,
        frame_idx_slider
    ])

    out = widgets.interactive_output(update_display, {
        "lower_h": lower_h, "lower_s": lower_s, "lower_v": lower_v,
        "upper_h": upper_h, "upper_s": upper_s, "upper_v": upper_v,
        "min_contour_area": min_area_slider,
        "frame_index": frame_idx_slider
    })

    display(ui, output, out)
    update_display(lower_h.value, lower_s.value, lower_v.value,
                   upper_h.value, upper_s.value, upper_v.value,
                   min_area_slider.value, frame_idx_slider.value)
