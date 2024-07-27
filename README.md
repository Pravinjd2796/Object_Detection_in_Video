# YOLOv8 Video Analysis Streamlit App

This repository contains a Streamlit application for analyzing videos using YOLOv8 (You Only Look Once version 8). The app processes uploaded videos to detect objects in each frame and provides a summary of detected classes.

## Features

- Upload and process video files (.mp4, .avi, .mov).
- Display the video within the Streamlit interface.
- Detect and annotate objects in video frames using YOLOv8.
- Extract and display detected object classes for each frame.
- Provide a downloadable JSON file with detailed detection information.

## Installation

To get started, you'll need to install the necessary Python packages. You can do this using `pip`:

```bash
pip install streamlit pyngrok opencv-python-headless ultralytics transformers
