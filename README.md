# Liquid Height Detection using Computer Vision

## The Problem
Monitoring liquid levels in multiple containers manually is tedious, time-consuming, and prone to human error. In laboratory or industrial settings, continuous tracking of liquid height requires constant supervision, making automated data logging difficult without expensive, specialized physical sensors.

## The Impact
This project solves the problem by using accessible computer vision technology to automate liquid level monitoring. By simply using a smartphone camera as a video stream and a computer, users can track the meniscus of liquids across multiple containers simultaneously. This automated approach not only provides real-time visual feedback but also continuously logs the data (in centimeters) directly into a CSV file with timestamps. It significantly reduces manual effort, increases measurement consistency, and provides ready-to-analyze data for experiments or process monitoring.

## Features
* **Multi-Container Tracking:** Define multiple Regions of Interest (ROIs) to monitor several tubes or containers at once.
* **Real-World Calibration:** Convert pixel measurements to physical units (centimeters) by clicking on known reference marks.
* **Automated Data Logging:** Continuously saves timestamps and height readings to a CSV file at customizable intervals.
* **Session Saving:** Automatically saves calibration data to a `.pkl` file, allowing you to skip the setup phase on subsequent runs.

## How to Use It

### 1. Setup (Before You Run)
* **Get the Video Stream:** Install the IP Webcam app (by Thyoni Tech for Android) or DroidCam (by DEV47APPS for iPhone) on your phone. Connect your phone to the same Wi-Fi as your computer, open the app, and tap "Start server" to get your IP address.
* **Prepare Your Computer:** Ensure Python is installed. Install the required libraries by running `pip install opencv-python numpy` in your terminal.

### 2. Code Configuration
* Open the Python script file (`height_detector_381 (1).py`) in a code editor.
* **Set Stream Source:** Locate line 11 and change the `STREAM_SOURCE` variable to match the IP address from your app (make sure to keep the `/video` at the end).
* *(Optional)* Change the `LOG_FILE` name or adjust the `LOG_INTERVAL` (the time in seconds between each data log).

### 3. Running the Detector
* Open your terminal, navigate to the script's folder, and run the script.
* **Check for Saved Calibration:** The script will check if you have run it before. The terminal will ask `Found saved data... Use it? (y/n)`. Type `y` to skip to live monitoring, or `n` to start a fresh setup.
* **Calibration (If new setup):**
  * Enter the number of ROIs (containers) you want to measure.
  * **Step A (Select ROI):** In the setup window, click one corner of your liquid container, then click the opposite corner to draw a box around it. Press `c` to confirm.
  * **Step B (Calibrate Marks):** Click your lower reference mark (e.g., "0 cm"), then click your upper reference mark (e.g., "5 cm").
  * **Step C (Enter Distance):** In the terminal, enter the real-world distance between the two clicked points.
  * *Note: Your settings will be automatically saved for next time.*

### 4. Monitoring & Output
Once calibrated, a "Live Feed" window will open displaying:
* **Green Boxes:** Your selected ROIs.
* **Red Lines:** The detected liquid level.
* **Text:** The calculated height in cm.

The script automatically logs the data to a `.csv` file in the same folder. Press `q` in the video window at any time to stop the script.
