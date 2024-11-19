# PROJECT-IMAGE PROCESSING
## Aim
To write a python program using OpenCV to do the following image manipulations.
i) Extract ROI from  an image.
ii) Perform handwritting detection in an image.
iii) Perform object detection with label in an image.
## Software Required:
Anaconda - Python 3.7
## Algorithm:
## I)Perform ROI from an image
### Step1:
Import necessary packages 
### Step2:
Read the image and convert the image into RGB
### Step3:
Display the image
### Step4:
Set the pixels to display the ROI 
### Step5:
Perform bit wise conjunction of the two arrays  using bitwise_and 
### Step6:
Display the segmented ROI from an image.
## Program:
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read and convert the image to RGB format
image = cv2.imread('dog.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Step 2: Display the original image
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')
```
![image](https://github.com/user-attachments/assets/dcab61b4-2334-419c-bee8-f3fa35d11436)


```python
# Step 3: Create an ROI mask
roi_mask = np.zeros_like(image_rgb)
roi_mask[100:250, 100:250, :] = 255  # Define the ROI region(Dog-face)

# Step 4: Segment the ROI using bitwise AND operation
segmented_roi = cv2.bitwise_and(image_rgb, roi_mask)

# Step 5: Display the segmented ROI
plt.imshow(segmented_roi)
plt.title('Segmented ROI')
plt.axis('off')

# Show both images side by side
plt.show()
```
![image](https://github.com/user-attachments/assets/c56d1a77-6c39-4ef6-aae3-11376b72e814)



## II)Perform handwritting detection in an image
### Step1:
Import necessary packages 
### Step2:
Define a function to read the image,Convert the image to grayscale,Apply Gaussian blur to reduce noise and improve edge detection,Use Canny edge detector to find edges in the image,Find contours in the edged image,Filter contours based on area to keep only potential text regions,Draw bounding boxes around potential text regions.
### Step3:
Display the results.
## Program:
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_handwriting(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 100
    text_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    img_copy = img.copy()
    for contour in text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('Handwriting Detection')
    plt.axis('off')
    plt.show()
    
image_path = 'handwriting.jpg'
detect_handwriting(image_path)
```
![image](https://github.com/user-attachments/assets/77d5e972-0504-4721-ad8d-843c63b00de2)



## III)Perform object detection with label in an image
### Step1:
Import necessary packages 
### Step2:
Set and add the config_file,weights to ur folder.
### Step3:
Use a pretrained Dnn model (MobileNet-SSD v3)
### Step4:
Create a classLabel and print the same
### Step5:
Display the image using imshow()
### Step6:
Set the model and Threshold to 0.5
### Step7:
Flatten the index,confidence.
### Step8:
Display the result.
## Program:
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the model files
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

# Create the detection model
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# Load class labels
classLabels = []
file_name = 'Labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

# Print the class labels and their count
print(classLabels)
print(f'Number of Classes: {len(classLabels)}')
```


```python

# Read the input image
img = cv2.imread('thiyagu.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Display original image
```
![my](https://github.com/user-attachments/assets/76a75a22-840b-428d-8d71-a135dd07f212)


```python
# Prepare the model for detection
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)  # Scale the input
model.setInputMean((127.5, 127.5, 127.5))  # Normalize the input
model.setInputSwapRB(True)  # Swap Red and Blue channels

# Perform object detection
ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)

# Print the detected classes and their confidence levels
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

# Check if any objects were detected
if len(ClassIndex) > 0:
    for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
        # Draw bounding box and label on the image
        cv2.rectangle(img, boxes, (0, 0, 255), 2)
        label = f"{classLabels[ClassInd-1]}: {conf:.2f}"  # Format the label with confidence
        cv2.putText(img, label, (boxes[0] + 10, boxes[1] + 40), font, fontScale=font_scale, color=(255, 0, 0), thickness=3)

        # Print the predicted label and confidence
        print(f'Detected: {classLabels[ClassInd-1]}, Confidence: {conf:.2f}')

# Display the image with detections
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Detected Objects')
plt.show()
```




