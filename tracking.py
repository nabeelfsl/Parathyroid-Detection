

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# Initialize variables for result analysis
iou_history = []
tracking_failures = 0
frame_times = []
area_history = []

def detect_green_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_detection = cv2.bitwise_and(frame, frame, mask=mask)
    return green_detection, mask

def find_largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h)
    return None

def smooth_bounding_box(history, new_box, max_history=5):
    history.append(new_box)
    if len(history) > max_history:
        history.pop(0)
    avg_box = np.mean(history, axis=0).astype(int)
    return avg_box, history

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# file path
video_path = r"C:\Users\nblfs\Downloads\ag432.mp4"
cap = cv2.VideoCapture(video_path)

# video writer
output_path = 'output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

green_start_time = None
bounding_box = None
box_set = False
tracker_initialized = False
tracker = None
bounding_box_history = []

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Detect green color in the frame
    green_detection, mask = detect_green_color(frame)

    if not box_set:
        if np.any(mask):
            if green_start_time is None:
                green_start_time = time.time()
                print("Green detected, starting timer...")
            elif time.time() - green_start_time >= 2:
                bounding_box = find_largest_contour(mask)
                if bounding_box:
                    x, y, w, h = bounding_box
                    frame_area = frame.shape[0] * frame.shape[1]
                    box_area = w * h
                    if box_area > frame_area * 0.02:
                        box_set = True
                        print(f"Bounding box set at: {bounding_box} with area {box_area}")
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, (x, y, w, h))
                        tracker_initialized = True
                        print("Tracker initialized")
                    else:
                        print(f"Ignored bounding box with small area: {box_area}")
        else:
            green_start_time = None

    if tracker_initialized:
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = map(int, bbox)
            bounding_box, bounding_box_history = smooth_bounding_box(bounding_box_history, (x, y, w, h))
            x, y, w, h = bounding_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            iou = calculate_iou(bounding_box, (x, y, w, h))
            iou_history.append(iou)
            area_history.append(w * h)
            print(f"Frame {len(iou_history)}: IOU={iou}, Area={w*h}")
        else:
            tracking_failures += 1
            print("Tracking failure")

    # Record processing time per frame
    processing_time = time.time() - start_time
    frame_times.append(processing_time)
    print(f"Frame {len(frame_times)}: Processing time={processing_time} seconds")

    # Output video
    out.write(frame)

    # Resize frames for display
    resized_frame = cv2.resize(frame, (640, 480))
    resized_green_detection = cv2.resize(green_detection, (640, 480))
    resized_mask = cv2.resize(mask, (640, 480))

    # Display frames
    cv2.imshow("Processed Frame", resized_frame)
    cv2.imshow("Green Detection", resized_green_detection)
    cv2.imshow("Green Mask", resized_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Plot IoU over time
plt.figure()
plt.plot(iou_history)
plt.title("IoU over Time")
plt.xlabel("Frame")
plt.ylabel("IoU")
plt.show()

# Plot Bounding Box Area over time
plt.figure()
plt.plot(area_history)
plt.title("Bounding Box Area over Time")
plt.xlabel("Frame")
plt.ylabel("Area")
plt.show()

# Plot Processing Time per Frame
plt.figure()
plt.plot(frame_times)
plt.title("Processing Time per Frame")
plt.xlabel("Frame")
plt.ylabel("Time (s)")
plt.show()

print(f"Total Tracking Failures: {tracking_failures}")
