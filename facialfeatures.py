#!/usr/bin/env python3

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import pandas as pd
import tempfile

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

# Replace with the path to your downloaded model
model_path = "/home/mujtahid/MediapipeFacialDataset/face_landmarker_v2_with_blendshapes.task"

# Create FaceLandmarker object with appropriate options
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


# Initialize MediaPipe FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Initialize Video Capture
video_path = '/home/mujtahid/MediapipeFacialDataset/FacialDataset.mp4'
cap = cv2.VideoCapture(video_path)

# Create a DataFrame with features and labels
columns = ["X", "Y", "Z", "Smiling", "Mouth_Open", "Eyes_Open"]
#dataset = pd.DataFrame(columns=columns)
features_df = pd.DataFrame(columns=columns)



# Initialize feature flags
smiling = False
mouth_open = False
eyes_open = False

i = 0


while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to detect facial landmarks
    results = face_mesh.process(rgb_image)

    # Create mp.Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    detection_result = detector.detect(mp_image)



    # Extract relevant facial landmarks (assuming only one face is detected)
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        blendshapes = detection_result.face_blendshapes[0]
        # Example thresholds for features (you can adjust these)
        mouth_threshold = 0.01
        eye_threshold = 0.02

        # Calculate distances for features
        mouth_distance = np.abs(face_landmarks.landmark[13].y - face_landmarks.landmark[14].y)
        left_eye_distance = np.abs(face_landmarks.landmark[159].y - face_landmarks.landmark[145].y)
        right_eye_distance = np.abs(face_landmarks.landmark[386].y - face_landmarks.landmark[374].y)

        # Set feature flags based on thresholds
        if mouth_distance > mouth_threshold:
            mouth_open = True
        if left_eye_distance > eye_threshold and right_eye_distance > eye_threshold:
            eyes_open = True

        if "Smile" in blendshapes:
            if blendshapes["Smile"] > 0.01:  # Example threshold (adjust as needed)
                smiling = True

    
    #features_df = pd.DataFrame(columns=columns)
    features_df.loc[i] = [face_landmarks.landmark[13].x, face_landmarks.landmark[13].y, face_landmarks.landmark[13].z,
                              smiling, mouth_open, eyes_open]
        
    i = i+1
    print(i)


    # Save the landmarks to a CSV file
    csv_file_path = "facial_landmarks.csv"
    features_df.to_csv(csv_file_path, index=False)

    #print(f"Facial landmarks saved to {csv_file_path}")



    


    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    ann_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Facial Landmarks", ann_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    


# Release resources
face_mesh.close()

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()

    
    

    

