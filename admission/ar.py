import cv2
import numpy as np
import mediapipe as mp
import os
from datetime import datetime

# Initialize MediaPipe's Image Segmenter
mp_drawing = mp.solutions.drawing_utils
mp_segmentation = mp.solutions.selfie_segmentation

# Load the video background
background = cv2.VideoCapture("bg1.mp4") 

# Start capturing video from the camera
camera = cv2.VideoCapture(0)  

# Get the screen resolution
screen_width = 1920  # Update with your screen resolution
screen_height = 1080  # Update with your screen resolution

# Set the camera resolution to match the screen resolution
camera.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# Set the background video resolution to match the camera resolution
background.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
background.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

#directory to store captured images
capture_folder = 'D:/Projects/ar/Capture'
os.makedirs(capture_folder, exist_ok=True)

with mp_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
    while camera.isOpened():
        success, image = camera.read()
        if not success:
            print("Error: Failed to capture frame from camera.")
            break

        image = cv2.flip(image, 1)

        # Segment the foreground from the camera feed
        results = segmenter.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        segmentation_mask = results.segmentation_mask

        # morphological operations for smoothing
        kernel = np.ones((5, 5), np.uint8)
        segmentation_mask = cv2.erode(segmentation_mask, kernel, iterations=1)
        segmentation_mask = cv2.dilate(segmentation_mask, kernel, iterations=1)

        # Threshold segmentation mask
        condition = segmentation_mask > 0.5

        # mask image for visualization
        segmented_image = np.zeros(image.shape, dtype=np.uint8)
        segmented_image[:] = (0, 255, 0)  # Green color for segmentation mask
        segmented_image = np.where(condition[:, :, None], image, segmented_image)

        # video background to match the camera feed size
        success_bg, bg_frame = background.read()
        if not success_bg:
            background.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, bg_frame = background.read()
        bg_frame = cv2.resize(bg_frame, (image.shape[1], image.shape[0]))

        # Combine the segmented image and the video background
        combined_image = np.where(condition[:, :, None], image, bg_frame)

        # Make the OpenCV window fullscreen
        cv2.namedWindow('Virtual Background', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Virtual Background', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Display the result
        cv2.imshow('Virtual Background', combined_image)

        # Press 'c' to capture image
        if cv2.waitKey(1) & 0xFF == ord('c'):
            # Generate file path for captured image with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            capture_path = os.path.join(capture_folder, f'ClickedAtChristUniversity_{timestamp}.png')
            # Save the combined image
            cv2.imwrite(capture_path, combined_image)
            print(f"Image captured and saved as {capture_path}")
            # Show message
            cv2.putText(combined_image, 'Image saved successfully!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Virtual Background', combined_image)
            cv2.waitKey(2000)  # Show the message for 2 seconds

        # Press 'q' to exit
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

camera.release()
background.release()
cv2.destroyAllWindows()
