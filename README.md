# Simulater-3D-Print

# 3D Printer Nozzle Tracking System

#Overview

This project implements a computer vision–based tracking system for monitoring the motion of a 3D printer nozzle using recorded video input.
The software detects the nozzle tip, tracks its movement frame-by-frame, converts coordinates into a centered reference system, and exports motion data for analysis and visualization.
This serves as a foundational framework for real-time motion validation and future integration with live camera feeds and G-code comparison.

#Features
-Frame-by-frame nozzle tracking using HSV color thresholding
-Template matching–based initialization (optional)
-Bottom-tip positional detection for physical accuracy
-Centered coordinate transformation (origin at frame center)
-CSV export of timestamped motion data
-Path visualization image generation
-Scaled 2D motion graph (normalized to 0–200 grid)
-Output video with tracking overlays

#How It Works
The program selects a video file automatically or via command-line argument.
Each frame is converted to HSV color space.
Orange regions (nozzle) are isolated using thresholding.
Contours are filtered and tracked based on proximity to the previous frame.
The bottom-center of the detected nozzle is recorded as the extrusion point.

#Position data is:
Logged to a CSV file
Drawn onto a cumulative path canvas
Scaled and plotted for motion analysis
