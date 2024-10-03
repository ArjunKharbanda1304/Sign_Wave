import cv2
import time
from ultralytics import YOLO

# Load the pre-trained YOLO model
model = YOLO(r"C:\Users\Arjun Kharbanda\Documents\SIH-2\best-final.pt")  # Replace with the path to your model

# Define the hand sign labels that the model can detect
import cv2
import time
from ultralytics import YOLO

# Load the pre-trained YOLO model
model = YOLO(r"C:\Users\Arjun Kharbanda\Documents\SIH-2\best-final.pt")  # Replace with the path to your model

# Define the hand sign labels that the model can detect
labels = ["Hello", "Help", "ILoveYou", "need", "Please", "how are"]

# Initialize variables for detection timing
detected_start_time = None
min_detection_duration = 3  # Minimum duration for detection in seconds
detected_label = None

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Ensure the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run YOLO prediction on the captured frame
    results = model.predict(frame, conf=0.3)

    # Loop through the results and draw bounding boxes for detected hand signs
    for result in results:
        img = result.orig_img  # Get the original image
        current_time = time.time()
        detected_this_frame = False

        # Loop through the detections in the current frame
        for det in result.boxes.data:
            x1, y1, x2, y2 = map(int, det[:4])  # Extract coordinates (x1, y1, x2, y2)
            confidence = det[4].item()  # Extract confidence score
            class_id = int(det[5].item())  # Extract class ID

            # Check if class_id is within the range of labels
            if class_id < len(labels):
                label = labels[class_id]  # Get the label from the predefined list

                # Check if the detected object is one of the specified hand signs
                if label in labels:
                    detected_this_frame = True
                    if detected_start_time is None:
                        detected_start_time = current_time
                        detected_label = label

                    # Draw the bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness 2

                    # Put the label and confidence on the bounding box
                    cv2.putText(img, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Green text with thickness 2
            else:
                print(f"Warning: Detected class_id {class_id} is out of range for the labels list.")

        # Check if a hand sign has been detected continuously for the required duration
        if detected_this_frame:
            if detected_start_time and (current_time - detected_start_time) >= min_detection_duration:
                print(f"Detected {detected_label} continuously for {min_detection_duration} seconds")
                detected_start_time = None  # Reset detection start time after storing data
        else:
            detected_start_time = None  # Reset detection start time if no detection in this frame

        # Display the image with bounding boxes
        cv2.imshow("Hand Sign Detection", img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


# Initialize variables for detection timing
detected_start_time = None
min_detection_duration = 3  # Minimum duration for detection in seconds
detected_label = None

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Ensure the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run YOLO prediction on the captured frame
    results = model.predict(frame, conf=0.3)

    # Loop through the results and draw bounding boxes for detected hand signs
    for result in results:
        img = result.orig_img  # Get the original image
        current_time = time.time()
        detected_this_frame = False

        # Loop through the detections in the current frame
        for det in result.boxes.data:
            x1, y1, x2, y2 = map(int, det[:4])  # Extract coordinates (x1, y1, x2, y2)
            confidence = det[4].item()  # Extract confidence score
            class_id = int(det[5].item())  # Extract class ID

            # Check if class_id is within the range of labels
            if class_id < len(labels):
                label = labels[class_id]  # Get the label from the predefined list

                # Check if the detected object is one of the specified hand signs
                if label in labels:
                    detected_this_frame = True
                    if detected_start_time is None:
                        detected_start_time = current_time
                        detected_label = label

                    # Draw the bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness 2

                    # Put the label and confidence on the bounding box
                    cv2.putText(img, f"{label} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Green text with thickness 2
            else:
                print(f"Warning: Detected class_id {class_id} is out of range for the labels list.")

        # Check if a hand sign has been detected continuously for the required duration
        if detected_this_frame:
            if detected_start_time and (current_time - detected_start_time) >= min_detection_duration:
                print(f"Detected {detected_label} continuously for {min_detection_duration} seconds")
                detected_start_time = None  # Reset detection start time after storing data
        else:
            detected_start_time = None  # Reset detection start time if no detection in this frame

        # Display the image with bounding boxes
        cv2.imshow("Hand Sign Detection", img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
