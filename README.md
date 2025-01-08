# Human-activity-and-energy-emission-calculation


## Pre trained model - [yolov and blip processor](https://github.com/harish-AK/Human-activity-and-energy-emission-calculation/blob/main/HUman_detection_and_activity_recognition.ipynb)
Here's a detailed summary of the project:

## Project Overview

This project is a Human Activity Recognition (HAR) system that uses computer vision and deep learning techniques to detect and recognize human activities in a video. The system extracts frames from a video, detects people in each frame using YOLOv4, and then uses the Blip model for image captioning to generate a caption for each detected person. The caption is then used to infer the action being performed by the person.

## Techniques Used

Computer Vision: OpenCV is used for image and video processing, including frame extraction, object detection, and image resizing.
Object Detection: YOLOv4 is used for detecting people in each frame. YOLOv4 is a real-time object detection system that uses a single neural network to predict bounding boxes and class probabilities directly from full images.
Image Captioning: The Blip model is used for generating captions for each detected person. The Blip model is a transformer-based model that uses a combination of computer vision and natural language processing techniques to generate captions for images.
Deep Learning: The Blip model is a deep learning model that uses a transformer architecture to generate captions for images.

                                      +---------------+
                                      |  Video File  |
                                      +---------------+
                                             |
                                             |
                                             v
                                      +---------------+
                                      |  Frame Extraction  |
                                      |  (OpenCV)          |
                                      +---------------+
                                             |
                                             |
                                             v
                                      +---------------+
                                      |  Object Detection  |
                                      |  (YOLOv4)          |
                                      +---------------+
                                             |
                                             |
                                             v
                                      +---------------+
                                      |  Image Captioning  |
                                      |  (Blip Model)      |
                                      +---------------+
                                             |
                                             |
                                             v
                                      +---------------+
                                      |  Action Recognition  |
                                      |  (Caption Analysis) |
                                      +---------------+
                                             |
                                             |
                                             v
                                      +---------------+
                                      |  Data Storage      |
                                      |  (Frame, Person ID,  |
                                      |   Action)          |
                                      +---------------+


---


## Custom dataset - [conv lstm](https://github.com/harish-AK/Human-activity-and-energy-emission-calculation/blob/main/HAR_model_ConvLstm_model.ipynb)
I developed a deep learning model using TensorFlow and Keras to classify videos into 7 classes. The model is a Convolutional LSTM (ConvLSTM) network, which is a type of recurrent neural network (RNN) that combines convolutional neural networks (CNNs) with LSTM layers to process sequential data with spatial hierarchies, such as videos.

---

## Techniques Used:

Video Frame Extraction: You've written a function frames_extraction to extract frames from video files using OpenCV.
Data Preprocessing: You've resized the frames to a fixed height and width, normalized the pixel values to be between 0 and 1, and stored the preprocessed frames in a list.
Dataset Creation: You've created a custom dataset by iterating through a directory containing video files organized by class, extracting frames from each video, and storing the frames, labels, and video file paths in separate lists.
ConvLSTM Model Architecture: You've designed a ConvLSTM model with multiple layers, including ConvLSTM2D, MaxPooling3D, TimeDistributed, and Dense layers. The model takes a sequence of frames as input and outputs a class label.
Model Compilation: You've compiled the model with a suitable optimizer and loss function.
## Workflow Diagram:
                                      +---------------+
                                      |  Video Files  |
                                      +---------------+
                                             |
                                             |
                                             v
                                      +---------------+
                                      |  Frame Extraction  |
                                      |  (OpenCV)          |
                                      +---------------+
                                             |
                                             |
                                             v
                                      +---------------+
                                      |  Data Preprocessing  |
                                      |  (Resize, Normalize)  |
                                      +---------------+
                                             |
                                             |
                                             v
                                      +---------------+
                                      |  Dataset Creation    |
                                      |  (Custom Dataset)     |
                                      +---------------+
                                             |
                                             |
                                             v
                                      +---------------+
                                      |  ConvLSTM Model      |
                                      |  (TensorFlow, Keras)  |
                                      +---------------+
                                             |
                                             |
                                             v
                                      +---------------+
                                      |  Model Compilation   |
                                      |  (Optimizer, Loss)    |
                                      +---------------+
