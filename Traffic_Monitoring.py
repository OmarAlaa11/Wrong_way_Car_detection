# Libraries
import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import os
from datetime import datetime

# Define Model
model = YOLO('yolov8s.pt')

#Define coco Data classes
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().split("\n")
# print (class_list)

# Define areas
area1 = [(569, 91), (281, 114), (266, 79), (531, 64)]
area2 = [(654, 125), (316, 171), (335, 210), (696, 148)]

# Create a folder to save wrong-way car images
save_dir = "wrong_way_cars"
os.makedirs(save_dir, exist_ok=True)

# Define car_status dictionary
car_status={}

# Define set to add wrong way cars in it and count it
wrong_way_cars=set()

cap = cv2.VideoCapture('Wrong_Way.mp4')
while True:    
    ret, frame = cap.read()
    if not ret:
        break
    # resize the frame
    frame = cv2.resize(frame, (1020, 500))

    '''
    the persist=True parameter typically means that the model will retain or store 
    certain information between frames when performing object tracking. 
    This helps maintain continuity of detected objects across multiple frames.
    '''
    wrong_way_counter=len(wrong_way_cars)

    results=model.track(frame,persist=True)

    # Extract All detections
    for result in results:
        boxes=result.boxes.xyxy.cpu().numpy()
        classes=result.boxes.cls.cpu().numpy()
        ids=result.boxes.id.cpu().numpy()
        # Extract each detection
        for i,box in enumerate(boxes):
            x1,y1,x2,y2=map(int,box)
            class_obj=int(classes[i])
            obj_id=int(ids[i])

            # Filter on class labels
            if class_list[class_obj]=='car':
                cx = (x1+x2)//2
                cy= (y1+y2)//2

                # Check if car is in area1 or area2
                in_area1 = cv2.pointPolygonTest(np.array(area1, np.int32), (cx, cy), False) >= 0
                in_area2 = cv2.pointPolygonTest(np.array(area2, np.int32), (cx, cy), False) >= 0

                # Initialize car state if not present
                if obj_id not in car_status:
                    car_status[obj_id] = {'in_area1': False, 'in_area2': False, 'wrong_way': False, 'saved': False}

                # Update car's status based on its current position
                if in_area1:
                    car_status[obj_id]['in_area1'] = True
                if in_area2:
                    car_status[obj_id]['in_area2'] = True

                # Check if the car went from area1 to area2 (wrong way)
                if car_status[obj_id]['in_area1'] and in_area2 and not car_status[obj_id]['wrong_way']:
                    car_status[obj_id]['wrong_way'] = True

                    # Add to wrong-way cars set
                    wrong_way_cars.add(obj_id)

                    # Save the wrong-way car image only once
                    if not car_status[obj_id]['saved']:
                        # Use current time as the filename
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        car_image_path = os.path.join(save_dir, f'car_{obj_id}_{timestamp}.png')

                        # Crop the car from the frame and save the image
                        car_image = frame[y1:y2, x1:x2]
                        cv2.imwrite(car_image_path, car_image)

                        # Mark the car as saved
                        car_status[obj_id]['saved'] = True



                # Draw bounding box and wrong way text if applicable
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cvzone.putTextRect(frame, f'ID: {obj_id}', (x1, y1), 1, 1, colorR=(255, 0, 0)) 

                if car_status[obj_id]['wrong_way']:
                    cvzone.putTextRect(frame, f'Wrong Way with ID: {obj_id}', (x1, y1 - 20), 1, 1, colorR=(0, 0, 255))

    # Visualize areas
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (200, 200, 200), 2)
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (200, 200, 200), 2)

    # Display the number of wrong-way cars on the frame
    cvzone.putTextRect(frame, f'Wrong Way Cars: {wrong_way_counter}', (10, 30), 1, 2, colorR=(0, 255, 0))


    cv2.imshow("Traffic Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    if cv2.getWindowProperty("Traffic Monitoring", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()




