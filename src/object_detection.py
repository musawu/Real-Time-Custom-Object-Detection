import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from src.utils import read_class_names

def load_model_and_classes():
    yolo_weights_path = "/Users/syntichemusawu/Downloads/Real-Time-Object-Detection-main/data/yolo-Weights/yolov8n.pt"
    class_names_file = "data/classes.txt"
    
    model = YOLO(yolo_weights_path)
    class_names = read_class_names(class_names_file)
    
    return model, class_names

def process_frame(frame, model, class_names):
    results = model(frame, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Display confidence and class name
            confidence = np.round(box.conf[0], decimals=2)
            cls = int(box.cls[0])
            class_name = class_names[cls]

            text = f"{class_name}: {confidence:.2f}"
            org = (x1, y1 - 10)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame

def main():
    st.title("Object Detection with Custom Model")

    # Load model and classes
    model, class_names = load_model_and_classes()

    # Sidebar for controls
    st.sidebar.header("Detection Controls")
    detection_type = st.sidebar.radio("Select Detection Mode", 
                                      ["Upload Image", "Real-Time Detection"])

    if detection_type == "Upload Image":
        uploaded_file = st.sidebar.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            # Read the image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Perform detection
            detected_img = process_frame(img, model, class_names)
            
            # Display the image
            st.image(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB), 
                     caption='Detected Image', use_column_width=True)

    else:  # Real-Time Detection
        run_detection = st.sidebar.button("Start Real-Time Detection")
        stop_detection = st.sidebar.button("Stop Detection")

        # Video capture placeholder
        frame_placeholder = st.empty()

        if run_detection:
            cap = cv2.VideoCapture(0)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break

                # Process the frame
                detected_frame = process_frame(frame, model, class_names)
                
                # Display the frame
                frame_placeholder.image(cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB), 
                                        channels="RGB")

                # Add a stop button
                if stop_detection:
                    break

            # Release the camera
            cap.release()

if __name__ == "__main__":
    main()