import cv2
import numpy as np
import sys
import os
import glob
import matplotlib.pyplot as plt

# Determine video source
video_path = ''

if len(sys.argv) > 1:
    video_path = sys.argv[1]
else:
    # Look for the specific video file first
    specific_video = "Meeting with Muhammad H. Rais-20260212_100250-Meeting Recording.mp4"
    if os.path.exists(specific_video):
        video_path = specific_video
    else:
        # Find most recent mp4 that isn't the output
        mp4_files = glob.glob('*.mp4')
        # Filter out known output file
        input_candidates = [f for f in mp4_files if f != 'output_tracking.mp4']
        
        if input_candidates:
            # Sort by modification time, newest first
            input_candidates.sort(key=os.path.getmtime, reverse=True)
            video_path = input_candidates[0]

if not video_path:
    print("Error: No .mp4 video files found in the directory.")
    exit()

print(f"Processing video: {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Output video writer
output_path = 'output_tracking.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Data output file
data_file = open('tracked_data.csv', 'w')
data_file.write("Timestamp,X,Y\n")

# Path visualization image
path_canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Calculate center of the frame
center_x = width / 2.0
center_y = height / 2.0

# Tracking Method Selection
# We will use HSV Color Thresholding for the Orange Nozzle
# This is often more robust than SIFT for featureless colored objects in motion

# Define range of orange color in HSV
# Adjust these values if the orange is too light or dark
lower_orange = np.array([5, 120, 120])
upper_orange = np.array([25, 255, 255])

frame_idx = 0
points = []
points_rel = [] # For graphing

# Initialization: Find the starting position of the nozzle
# Priority 1: Use the screenshot template
screenshot_path = "Screenshot 2026-02-12 at 9.56.49 AM.png"
last_known_center = None

if os.path.exists(screenshot_path):
    print(f"Loading template: {screenshot_path}")
    template = cv2.imread(screenshot_path)
    if template is not None:
        # Read the first frame to find the match
        ret, first_frame = cap.read()
        if ret:
            # Check if template is larger than frame (resize if needed)
            if template.shape[0] > first_frame.shape[0] or template.shape[1] > first_frame.shape[1]:
                # If template is huge, assume it's a full screenshot and try to find the nozzle in it? 
                # Or maybe the video is smaller. Let's just resize template to be smaller if needed
                # But typically the screenshot is a crop. If it's a full screenshot, this might fail.
                # Assuming it's a crop or reasonable size.
                pass
                
            try:
                # Use Template Matching
                res = cv2.matchTemplate(first_frame, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                
                # If match is good enough
                if max_val > 0.7: 
                    top_left = max_loc
                    h, w = template.shape[:2]
                    # Center of the matched template
                    cx = top_left[0] + w // 2
                    cy = top_left[1] + h # Bottom of template? User wants bottom most portion.
                    # Actually, let's stick to center for tracking logic, but record bottom for plotting
                    last_known_center = (cx, cy)
                    print(f"Template matched! Starting at: {last_known_center}")
                else:
                    print(f"Template match score too low: {max_val}")
            except Exception as e:
                print(f"Template matching failed: {e}")
            
            # Reset video to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
# Priority 2: If template failed, we will initialize with the largest orange contour in the first loop iteration
if last_known_center is None:
    print("Using largest orange contour for initialization.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only orange colors
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Clean up the mask
    # Morphological operations to remove small noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    timestamp = frame_idx / fps
    
    # If we found any contours
    if contours:
        # Filter contours by area first to remove small noise
        valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
        
        target_contour = None
        
        if valid_contours:
            if last_known_center is not None:
                # Find the contour whose center is closest to the last known center
                min_dist = float('inf')
                best_contour = None
                
                for cnt in valid_contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Current center of this contour
                    cx = x + w // 2
                    cy = y + h # Bottom-most point as tracking reference
                    
                    # Calculate distance
                    dist = np.sqrt((cx - last_known_center[0])**2 + (cy - last_known_center[1])**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_contour = cnt
                
                # Update target
                # We can also add a max_dist threshold here if needed to avoid jumping too far
                target_contour = best_contour
            else:
                # Initialization phase if template matching failed
                # Pick the largest orange object
                target_contour = max(valid_contours, key=cv2.contourArea)

        if target_contour is not None:
            x, y, w, h = cv2.boundingRect(target_contour)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Calculate bottom-center of the detected object
            # Use bottom edge (y+h) because that's where the nozzle tip is
            cX_float = x + w / 2.0
            cY_float = float(y + h) 
            
            # Update last known center for next frame tracking
            last_known_center = (cX_float, cY_float)
            
            cX = int(cX_float)
            cY = int(cY_float)

            # Calculate relative coordinates (center as 0,0, Y up positive)
            rel_x = cX_float - center_x
            rel_y = center_y - cY_float
            
            # Output: timestamp x-y coordinates (rounded to nearest whole number)
            print(f"{timestamp:.4f} {int(round(rel_x))},{int(round(rel_y))}")
            data_file.write(f"{timestamp:.4f},{int(round(rel_x))},{int(round(rel_y))}\n")
            
            # Add point to list
            points.append((cX, cY))
            points_rel.append((rel_x, rel_y))

    # Draw tracking line
    if len(points) > 1:
        for i in range(1, len(points)):
            # Draw on video frame
            cv2.line(frame, points[i - 1], points[i], (0, 0, 255), 2)
            # Draw on path canvas
            cv2.line(path_canvas, points[i - 1], points[i], (0, 255, 0), 2)
    
    # Draw current position
    if points:
        cv2.circle(frame, points[-1], 5, (0, 255, 0), -1)

    # Show tracking visualization side by side
    combined_frame = np.hstack((frame, path_canvas))
    
    # Create info panel
    info_panel_height = 50
    info_panel = np.zeros((info_panel_height, combined_frame.shape[1], 3), dtype=np.uint8)
    
    # Prepare text
    if points_rel:
        current_x = int(round(points_rel[-1][0]))
        current_y = int(round(points_rel[-1][1]))
        text = f"Time: {timestamp:.2f}s | X: {current_x} | Y: {current_y}"
        cv2.putText(info_panel, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        cv2.putText(info_panel, "Searching for nozzle...", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Stack vertically
    final_output = np.vstack((combined_frame, info_panel))
    
    cv2.imshow('Tracking Visualization', final_output)
    
    # Slow down video playback: wait 100ms instead of 1ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    out.write(frame)
    
    frame_idx += 1

cv2.destroyAllWindows()
cap.release()
out.release()
data_file.close()
cv2.imwrite('path_visualization.png', path_canvas)

print(f"Tracking video saved to {output_path}")
print("Tracking data saved to tracked_data.csv")
print("Path visualization saved to path_visualization.png")

# Generate Matplotlib Graph
if points_rel:
    x_coords = [p[0] for p in points_rel]
    y_coords = [p[1] for p in points_rel]

    # Shift coordinates so the bottom-left point of the path is at (0,0)
    min_x = min(x_coords)
    min_y = min(y_coords)
    x_shifted = [x - min_x for x in x_coords]
    y_shifted = [y - min_y for y in y_coords]

    # Scale the path to fit within a 200x200 grid while preserving aspect ratio
    # Find the largest dimension of the path
    max_dim = max(max(x_shifted), max(y_shifted))
    scale_factor = 200 / max_dim if max_dim > 0 else 1

    x_scaled = [x * scale_factor for x in x_shifted]
    y_scaled = [y * scale_factor for y in y_shifted]

    plt.figure(figsize=(10, 10))
    plt.scatter(x_scaled, y_scaled, c='blue', marker='o', s=15, label='Tracked Nozzle Tip')
    plt.plot(x_scaled, y_scaled, c='red', alpha=0.5, linewidth=1)

    plt.title('Nozzle Tip Tracking Path (Calibrated to 0-200)')
    plt.xlabel('X Coordinate (0-200)')
    plt.ylabel('Y Coordinate (0-200)')

    # Set fixed axes from 0 to 200
    plt.xlim(0, 200)
    plt.ylim(0, 200)

    # Add grid lines for better readability
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray')

    # Ensure aspect ratio is equal within the new fixed axes
    plt.gca().set_aspect('equal', adjustable='box')

    plt.legend()

    plt.savefig('tracking_graph.png')
    print("Tracking graph saved to tracking_graph.png")
else:
    print("No points tracked, skipping graph generation.")
