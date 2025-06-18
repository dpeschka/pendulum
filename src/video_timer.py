import cv2

def add_timer(video_path, output_path, drawing_params={'timer_color': (255, 255, 255)}):
    """
    Add a timer overlay to a video showing elapsed time in seconds.
    
    Args:
        video_path (str): Path to input video file
        output_path (str): Path for output video file
        drawing_params (dict): Parameters for drawing, expects 'timer_color' as (r,g,b) tuple
    """
    # Open input video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    timer_color = drawing_params.get('timer_color', (255, 255, 255))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate time in seconds
        time_seconds = frame_count / fps
        timer_text = f"{time_seconds:.2f}s"
        
        # Add timer text to lower left corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_size = cv2.getTextSize(timer_text, font, font_scale, thickness)[0]
        
        # Position in lower left corner with some padding
        x = 10
        y = height - 10
        
        # Add text to frame
        cv2.putText(frame, timer_text, (x, y), font, font_scale, timer_color, thickness)
        
        # Write frame to output video
        out.write(frame)
        frame_count += 1
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
