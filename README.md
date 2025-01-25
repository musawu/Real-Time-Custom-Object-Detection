Real-Time Object Detection with Custom YOLO Model

This project demonstrates real-time object detection using a custom-trained YOLO (You Only Look Once) model, featuring Streamlit integration and advanced detection capabilities.

Features

Real-time object detection via webcam

Custom-trained YOLO model for detection

Interactive Streamlit interface

Supports both image upload and live video detection

Displays detected object classes with confidence scores

Prerequisites

Python 3.8+

Webcam for live detection

Custom-trained YOLO weights (best.pt)

Installation

Clone the repository:

git clone https://github.com/yourusername/Real-Time-Object-Detection.git
cd Real-Time-Object-Detection

Create a virtual environment (recommended):

python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt

Usage

Run the Streamlit application:

streamlit run main.py

In the Streamlit interface:

Choose between image upload or real-time detection.

For real-time detection, click Start Real-Time Detection.

Press q to stop detection.

Project Structure

Real-Time-Object-Detection/
│
├── data/
│   └── yolo-Weights/
│       └── best.pt  # Custom-trained YOLO weights
├── src/
│   ├── object_detection.py  # Core object detection logic
│   └── utils.py             # Utility functions
├── main.py                  # Streamlit application entry point
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

Customization

Replace the best.pt file in data/yolo-Weights/ with your custom-trained YOLO weights.

Update classes.txt (if applicable) to match the object classes you want to detect.

Dependencies

OpenCV

Ultralytics

Streamlit

NumPy

Contributing

Contributions are welcome! If you'd like to contribute, feel free to:

Submit pull requests

Open issues for bugs or feature requests

License

This project is licensed under the MIT License.
