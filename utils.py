from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import os
from pathlib import Path
import numpy as np
import csv

def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.
    :param conf (float): Confidence threshold for object detection.
    :param model (YOLOv8): An instance of the `YOLOv8` class containing the YOLOv8 model.
    :param st_frame (Streamlit object): A Streamlit object to display the detected video.
    :param image (numpy array): A numpy array representing the video frame.
    :return: image with detections, list of detected objects (time, class_id)
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)
    
    # Extract detected classes
    detected_objects = []
    for box in res[0].boxes:
        detected_objects.append((box.cls, box.conf))
    
    return res_plotted, detected_objects

@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def save_detection_results(image_path, boxes, save_dir='detection'):
    os.makedirs(save_dir, exist_ok=True)
    
    base_name = Path(image_path).stem
    txt_path = os.path.join(save_dir, f"{base_name}.txt")
    
    with open(txt_path, 'w') as f:
        for box in boxes:
            class_id = float(box.cls)
            x_center, y_center, width, height = box.xywhn[0]
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def save_csv_results(csv_file_path, detections):
    with open(csv_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Time', 'Class'])
        writer.writerows(detections)

def infer_uploaded_image(conf, model):
    """
    Execute inference for uploaded image
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_img = st.sidebar.file_uploader(label="Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(image=uploaded_image, caption="Uploaded Image", use_column_width=True)

    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image, conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted, caption="Detected Image", use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                xywhn = box.xywhn
                                classes = box.cls
                                st.write(classes, xywhn)

                        # Save results in YOLO format
                        save_detection_results(source_img.name, boxes)

                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)

def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    source_video = st.sidebar.file_uploader(label="Choose a video...")

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(tfile.name)
                    st  
                    st_frame = st.empty()

                    # Определяем параметры для записи видео
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    output_path = os.path.join('detection', 'output_video.mp4')
                    out = cv2.VideoWriter(output_path, fourcc, 30.0, (720, int(720 * (9 / 16))))

                    # Список для сохранения таймингов и классов
                    detection_results = []

                    frame_number = 0
                    while vid_cap.isOpened():
                        success, image = vid_cap.read()
                        if success:
                            detected_frame, detected_objects = _display_detected_frames(conf, model, st_frame, image)

                            # Записываем кадр в видеофайл
                            out.write(detected_frame)

                            # Сохраняем результаты детекции с таймингами (время в секундах, класс)
                            for obj in detected_objects:
                                detection_results.append((round(frame_number / 30.0, 1), obj[0]))  # 30.0 - кадров в секунду (FPS)

                            frame_number += 1
                        else:
                            break

                    vid_cap.release()
                    out.release()

                    # Сохраняем результаты в CSV файл
                    csv_path = os.path.join('detection', 'detection_results.csv')
                    save_csv_results(csv_path, detection_results)
                    st.success(f"Video saved to {output_path}")
                    st.success(f"CSV saved to {csv_path}")

                except Exception as e:
                    st.error(f"Error loading video: {e}")

def infer_uploaded_webcam(conf, model):
    """
    Execute inference for webcam.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the `YOLOv8` class containing the YOLOv8 model.
    :return: None
    """
    try:
        flag = st.button(label="Stop running")
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()
        
        # Определяем параметры для записи видео
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join('detection', 'webcam_output_video.mp4')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (720, int(720 * (9 / 16))))
        
        while not flag:
            success, image = vid_cap.read()
            if success:
                detected_frame = _display_detected_frames(conf, model, st_frame, image)

                # Записываем кадр в видеофайл
                out.write(detected_frame)
            else:
                break

        vid_cap.release()
        out.release()
        st.success(f"Webcam video saved to {output_path}")

    except Exception as e:
        st.error(f"Error loading video: {str(e)}")

