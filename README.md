Copy# Real-Time Object Detection with Custom YOLO Model

This project demonstrates real-time object detection using a custom-trained YOLO (You Only Look Once) model, featuring Streamlit integration and advanced detection capabilities.

## Features

- Real-time object detection using webcam
- Custom-trained YOLO model detection
- Streamlit interactive interface
- Supports both image and live video detection
- Displays object class and confidence scores

## Prerequisites

- Python 3.8+
- Webcam
- Custom-trained YOLO weights

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Real-Time-Object-Detection.git
   cd Real-Time-Object-Detection

Create virtual environment (recommended):
bashCopypython3 -m venv venv
source venv/bin/activate

Install dependencies:
bashCopypip install -r requirements.txt


Usage

Run Streamlit application:
bashCopystreamlit run main.py

In the Streamlit interface:

Choose between image upload or real-time detection
For real-time detection, click "Start Real-Time Detection"
Press 'q' to stop detection



Project Structure
CopyReal-Time-Object-Detection/
│
├── data/
│   └── yolo-Weights/
│       └── best.pt
├── src/
│   ├── object_detection.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
Customization

Replace best.pt in data/yolo-Weights/ with your custom-trained weights
Modify classes.txt to match your detection classes

Dependencies

OpenCV
Ultralytics
Streamlit
NumPy

Contributing
Contributions are welcome! Please submit pull requests or open issues.
License
MIT License
