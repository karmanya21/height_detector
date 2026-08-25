import cv2
import numpy as np
import csv
import time
from datetime import datetime
import pickle  # <-- ADDED
import os      # <-- ADDED

# ================= CONFIGURATION =================

STREAM_SOURCE = "http://172.17.133.169:4747/video"          #change address to your local host server  

LOG_FILE = "liquid_level_log_multi.csv"  # File to store height readings
CALIB_FILE = "calibration_data.pkl"    # File to store calibration settings
EDGE_LOW = 50              # Lower threshold for edge detection (Canny)
EDGE_HIGH = 150            # Upper threshold for edge detection (Canny)
SMOOTHING_KERNEL = 5       # Gaussian blur kernel size (for noise removal)
LOG_INTERVAL = 1          # Time (in seconds) between each CSV log entry
MAX_ROIS = 3               # Maximum number of ROIs allowed
# =================================================

# --- Global variables (now lists for multiple ROIs) ---
num_rois = 0               # Will store the user-defined number of ROIs
all_roi_coords = []        # Stores (x1, y1), (x2, y2) for each ROI
all_calib_points = []      # Stores (0 cm point, 5 cm point) for each ROI
all_cm_per_pixel = []      # Conversion factor for each ROI
last_log_time = 0          # For controlling CSV logging interval

# --- Mouse callback handler state ---
current_setup_step = 'ROI'  # 'ROI' or 'CALIBRATION'
current_roi_index = 0       # The index of the ROI currently being set up
temp_points = []            # Temporarily stores 1 or 2 points during setup

def initialize_camera(source):  
    """Open camera and verify it’s working."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise Exception(f"Camera or stream {source} could not be opened.")
    print(f"Camera {source} initialized.")
    return cap


def detect_liquid_height(frame, roi_coords):
    """Detect the meniscus (liquid surface) inside the selected ROI."""
    (x1, y1), (x2, y2) = roi_coords
    x1, x2 = sorted([x1, x2])  # ensure proper ordering
    y1, y2 = sorted([y1, y2])

    # Crop only the region where liquid is expected
    roi = frame[y1:y2, x1:x2]

    # Handle empty ROI crop
    if roi.size == 0:
        return None, None, None

    # Convert to grayscale and blur to reduce noise
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (SMOOTHING_KERNEL, SMOOTHING_KERNEL), 0)

    # Detect edges (meniscus usually forms a strong horizontal edge)
    edges = cv2.Canny(blurred, EDGE_LOW, EDGE_HIGH)

    # Sum edge intensities across each row
    horizontal_sum = np.sum(edges, axis=1)
    if np.max(horizontal_sum) == 0:
        # No strong edges found
        return None, roi, edges

    # Row with maximum edge intensity = meniscus position
    level_y = np.argmax(horizontal_sum)
    absolute_y = y1 + level_y  # convert back to full-frame coordinates
    return absolute_y, roi, edges


def log_data(heights_cm):
    """Write heights (in cm) to CSV file every LOG_INTERVAL seconds."""
    global last_log_time
    current_time = time.time()

    # Only log if enough time has passed
    if current_time - last_log_time >= LOG_INTERVAL:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format the height data for the CSV row
        log_row = [timestamp] + [f"{h:.2f}" for h in heights_cm]

        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(log_row)

        last_log_time = current_time

def setup_mouse_callback(event, x, y, flags, param):
    """Generic mouse callback for both ROI and Calibration selection."""
    global temp_points, current_setup_step, all_roi_coords, all_calib_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_setup_step == 'ROI':
            # Collect 2 points for the ROI bounding box
            if len(temp_points) < 2:
                temp_points.append((x, y))
                print(f"ROI point selected: {x}, {y}")

        elif current_setup_step == 'CALIBRATION':
            # Collect 2 points for calibration marks
            if len(temp_points) < 2:
                temp_points.append((x, y))
                print(f"Calibration point selected: {x}, {y}")


def get_user_input_rois():
    """Asks the user for the number of ROIs."""
    while True:
        try:
            n = int(input(f"Enter the number of ROIs (1-{MAX_ROIS}): "))
            if 1 <= n <= MAX_ROIS:
                return n
            else:
                print(f"Please enter a number between 1 and {MAX_ROIS}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

def get_user_input_cm_dist(roi_index):
    """Asks the user for the real-world distance between calibration marks."""
    while True:
        try:
            dist = float(input(f"Enter the real-world distance (in cm) between the two marks for ROI {roi_index+1}: "))
            if dist > 0:
                return dist
            else:
                 print("Distance must be a positive number.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    # <-- global list -->
    global num_rois, current_setup_step, current_roi_index, temp_points
    global all_roi_coords, all_calib_points, all_cm_per_pixel, last_log_time

    load_success = False
    if os.path.exists(CALIB_FILE):
        while True:
            choice = input(f"Found saved data ({CALIB_FILE}). Use it? (y/n): ").lower().strip()
            if choice == 'y':
                try:
                    with open(CALIB_FILE, 'rb') as f:
                        saved_data = pickle.load(f)
                    
                    # Load data into global variables
                    num_rois = saved_data['num_rois']
                    all_roi_coords = saved_data['all_roi_coords']
                    all_calib_points = saved_data['all_calib_points']
                    all_cm_per_pixel = saved_data['all_cm_per_pixel']
                    
                    print(f"Successfully loaded {num_rois} ROIs from file.")
                    load_success = True
                    break
                except Exception as e:
                    print(f"Error loading file: {e}. Starting new setup.")
                    break
            elif choice == 'n':
                print("Starting new setup.")
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    
    # ---------- STEP 1: Setup camera (needs to be initialized for both paths) ----------
    cap = initialize_camera(STREAM_SOURCE) 


    # ---------- MODIFIED: Only run setup if data was NOT loaded ----------
    if not load_success:
        num_rois = get_user_input_rois()
        
        cv2.namedWindow("Setup")
        cv2.setMouseCallback("Setup", setup_mouse_callback)

        # --- Setup loop for all ROIs (ROI selection and Calibration) ---
        for current_roi_index in range(num_rois):
            # ----------------- Sub-Step 1: ROI Selection -----------------
            print(f"\n>>> ROI {current_roi_index+1} / {num_rois}: Draw the region of interest (ROI) around the tube.")
            print(">>> Left-click two opposite corners. Press 'c' when done.")
            current_setup_step = 'ROI'
            temp_points = []
            roi_confirmed = False

            while not roi_confirmed:
                ret, frame = cap.read()
                if not ret:
                    print("Frame capture failed.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                display = frame.copy()

                # Draw all previously confirmed ROIs
                for i, roi_coords in enumerate(all_roi_coords):
                    (x1, y1), (x2, y2) = roi_coords
                    color = (0, 255, 0) if i == current_roi_index else (255, 255, 0) # Highlight current
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display, f"ROI {i+1}", (min(x1, x2), min(y1, y2) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Show live clicks for the current ROI setup
                if len(temp_points) == 1:
                    cv2.circle(display, temp_points[0], 5, (0, 255, 0), -1)
                elif len(temp_points) == 2:
                    cv2.rectangle(display, temp_points[0], temp_points[1], (0, 255, 0), 2)

                cv2.imshow("Setup", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'): return # Global quit

                if key == ord('r'): # Reset current ROI points
                    temp_points.clear()

                if key == ord('c') and len(temp_points) == 2:
                    all_roi_coords.append((temp_points[0], temp_points[1]))
                    roi_confirmed = True
                    print(f"ROI {current_roi_index+1} locked.")

            # ----------------- Sub-Step 2: Calibration -----------------
            print(f"\n>>> ROI {current_roi_index+1} / {num_rois}: Click two calibration marks on the scale.")
            print(">>> First = lower mark (0 cm), Second = upper mark (e.g., 5 cm).")
            current_setup_step = 'CALIBRATION'
            temp_points = []
            calib_confirmed = False

            while not calib_confirmed:
                ret, frame = cap.read()
                if not ret: break

                display = frame.copy()
                (x1_roi, y1_roi), (x2_roi, y2_roi) = all_roi_coords[current_roi_index]
                cv2.rectangle(display, (x1_roi, y1_roi), (x2_roi, y2_roi), (0, 255, 0), 2)

                # Show live calibration clicks
                if len(temp_points) == 1:
                    cv2.circle(display, temp_points[0], 5, (255, 0, 0), -1)
                elif len(temp_points) == 2:
                    cv2.line(display, temp_points[0], temp_points[1], (255, 0, 0), 2)

                cv2.imshow("Setup", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'): return # Global quit

                if len(temp_points) == 2:
                    all_calib_points.append(temp_points.copy())
                    calib_confirmed = True

            # ----------------- Sub-Step 3: Compute pixel-to-cm ratio -----------------
            zero_cm_point, upper_cm_point = all_calib_points[current_roi_index]
            pixel_dist = abs(upper_cm_point[1] - zero_cm_point[1])

            cm_mark_distance = get_user_input_cm_dist(current_roi_index)

            # Conversion factor
            cm_per_pixel = cm_mark_distance / pixel_dist
            all_cm_per_pixel.append(cm_per_pixel)
            print(f"Calibration for ROI {current_roi_index+1} complete: {cm_per_pixel:.5f} cm/pixel")

        if len(all_roi_coords) != num_rois or len(all_calib_points) != num_rois:
            print("Setup not completed for all ROIs. Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            return

        # ---------- ADDED: Save data after successful setup ----------
        try:
            data_to_save = {
                'num_rois': num_rois,
                'all_roi_coords': all_roi_coords,
                'all_calib_points': all_calib_points,
                'all_cm_per_pixel': all_cm_per_pixel
            }
            with open(CALIB_FILE, 'wb') as f:
                pickle.dump(data_to_save, f)
            print(f"Calibration data saved to {CALIB_FILE}")
        except Exception as e:
            print(f"Error saving calibration data: {e}")
        
        # Close setup window
        cv2.destroyWindow("Setup")

    # ---------- STEP 4: Live detection (This section runs for both loaded and new data) ----------
    print("\n>>> Starting live liquid level detection. Press 'q' to quit.\n")

    # Create CSV file and write header
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Timestamp"] + [f"ROI {i+1} Height (cm)" for i in range(num_rois)]
        writer.writerow(header)

    last_log_time = time.time()

    # --- Live feed loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed.")
            break

        current_heights_cm = []
        display_frame = frame.copy()

        # Iterate over all ROIs
        for i in range(num_rois):
            roi_coords = all_roi_coords[i]
            cm_per_pixel = all_cm_per_pixel[i]
            zero_cm_y = all_calib_points[i][0][1] # 0 cm reference (bottom mark)
            (x1, y1), (x2, y2) = roi_coords
            x1, x2 = sorted([x1, x2]) # <-- sorting just in case
            y1, y2 = sorted([y1, y2])

            height_px, roi_view, edges_view = detect_liquid_height(frame, roi_coords)

            # Draw ROI on screen
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if height_px is not None:
                # Convert pixel height → cm using calibration
                height_cm = (zero_cm_y - height_px) * cm_per_pixel
                height_cm = max(0.0, height_cm) # Clamp to zero

                current_heights_cm.append(height_cm)

                # Draw red line at detected level
                cv2.line(display_frame, (x1, height_px), (x2, height_px), (0, 0, 255), 2)
                cv2.putText(display_frame, f"ROI {i+1}: {height_cm:.2f} cm",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)
            else:
                # If no level detected, log None or a placeholder
                current_heights_cm.append(None)
                cv2.putText(display_frame, f"ROI {i+1}: No Level",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

        # Log all data simultaneously
        if all(h is not None for h in current_heights_cm):
             log_data(current_heights_cm)
        
        cv2.imshow("Live Feed", display_frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()