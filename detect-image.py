import cv2
import numpy as np
from ultralytics import YOLO  # Import the YOLO class from ultralytics package

# Load your trained YOLOv8 model
model = YOLO(r'C:\Users\Arjun Kharbanda\Documents\SIH-2\best-2.pt')

# Function to load and preprocess an image
def preprocess_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, image_rgb

# Function to run the model and get predictions
def detect_sign_language(image_rgb):
    # Run inference using the YOLOv8 model
    results = model(image_rgb)
    return results

# Function to draw bounding boxes and labels on the image
def draw_bounding_boxes(image, results):
    # Access the first result object
    result = results[0]

    # Iterate over the detected boxes
    for box in result.boxes.xyxy:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box[:4])
        # Extract confidence and class index
        confidence = box[4]
        class_idx = int(box[5])
        
        # Get class name using the class index
        label = result.names[class_idx]

        # Draw rectangle for bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put label text on the bounding box
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

# Main function to process an image
def process_image(image_path):
    # Load and preprocess the image
    image, image_rgb = preprocess_image(image_path)

    # Detect sign language in the image
    results = detect_sign_language(image_rgb)

    # Draw bounding boxes and labels
    output_image = draw_bounding_boxes(image, results)

    # Save the output image
    output_path = r'C:\Users\Arjun Kharbanda\Documents\SIH-2\output_image.jpg'
    cv2.imwrite(output_path, output_image)
    print(f"Output image saved to {output_path}")

