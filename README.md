# Vehicle Speed Detection System
---
## Overview

This project implements a real-time vehicle speed detection system using computer vision techniques.  
It processes video feeds to estimate the speed of moving vehicles, which can be instrumental in traffic monitoring and law enforcement applications.

## 🚗 Features

- **Real-Time Speed Estimation**: Calculates the speed of vehicles as they move through the camera's field of view.
- **Object Detection**: Identifies vehicles in the video frames using advanced detection algorithms.
- **Tracking**: Maintains the identity of each vehicle across frames to ensure accurate speed measurement.
- **Visualization**: Displays annotated video with speed information overlaid on the detected vehicles.

---

## ⚙️ Technologies Used

- **Python**: Programming language for implementing the system.
- **OpenCV**: Library for computer vision tasks such as video processing and object detection.
- **YOLOv8**: Pre-trained model for vehicle detection.
- **DeepSORT**: Algorithm for tracking objects across video frames.

---

## 🛠️ Setup and Usage

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- OpenCV
- PyTorch
- NumPy

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/meetbhayani/vehicle-speed-detection.git
   cd vehicle-speed-detection
   ```
   
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the python script:
   ```bash
   python speed_est.py

- The output will be an annotated video saved as annotated_output.mp4, displaying the estimated speed of each detected vehicle.

# 📁 Project Structure
- **speed_est.py :** Main script for speed estimation.
- **tracker.py :** Contains the tracking logic using DeepSORT.
- **yolov8s.pt :** Pre-trained YOLOv8 model weights for vehicle detection.
- **highway.mp4 :** Sample video for testing the system.
- **annotated_output.mp4 :** Output video with speed annotations.

# 🤖 How It Works
1. **Vehicle Detection:** The system uses the YOLOv8 model to detect vehicles in each frame of the video.
2. **Object Tracking:** DeepSORT is employed to track each detected vehicle across frames, maintaining consistent identities.
3. **Speed Calculation:** The system calculates the speed of each vehicle based on the distance traveled between frames and the time elapsed.
4. **Visualization:** The estimated speed is overlaid on the video, and the annotated video is saved as output


