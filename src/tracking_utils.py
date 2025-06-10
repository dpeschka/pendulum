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
    frame with annotations and a hue color wheel next to it.
    """
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display

    def generate_hue_wheel(radius=100):
        """Creates an RGB hue wheel image as a NumPy array."""
        y, x = np.ogrid[-radius:radius, -radius:radius]
        mask = x**2 + y**2 <= radius**2
        angle = (np.arctan2(-y, x) + np.pi) * 180 / np.pi
        hsv = np.zeros((2*radius, 2*radius, 3), dtype=np.uint8)
        hsv[..., 0] = angle.astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = 255
        hsv[~mask] = 0
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        rgb[~mask] = 255  # white background outside circle
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

    hue_wheel_img = generate_hue_wheel(radius=100)

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

        avg_hsv = np.array([(lower[0]+upper[0])//2, (lower[1]+upper[1])//2, (lower[2]+upper[2])//2], dtype=np.uint8)
        avg_rgb = cv2.cvtColor(avg_hsv.reshape((1,1,3)), cv2.COLOR_HSV2RGB).reshape(3) / 255.0

        with output:
            output.clear_output(wait=True)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [3, 1]})
            ax1.imshow(rgb)
            ax1.set_title(f"Frame {frame_index}")
            ax1.axis('off')

            ax2.imshow(hue_wheel_img)
            ax2.set_title("Hue wheel")
            ax2.axis('off')

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
