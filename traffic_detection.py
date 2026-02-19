import cv2
import numpy as np
import json
import os
import time
from ultralytics import YOLO

VIDEO_1 = "road_1.mp4"
VIDEO_2 = "road_2.mp4"
MODEL_PATH = "yolov8n.pt"
ROI_FILE = "roi_config.json"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

VEHICLE_CLASSES = ["car", "motorcycle", "bus", "truck"]

BASE_GREEN_TIME = 10
TIME_PER_VEHICLE = 1.5
MIN_GREEN = 10
MAX_GREEN = 60
MAX_RED_TIME = 90

model = YOLO(MODEL_PATH)

polygon_points = []
drawing_done = False

def mouse_callback(event, x, y, flags, param):
    global polygon_points, drawing_done
    if drawing_done:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if polygon_points:
            polygon_points.pop()

def calibrate_polygon(video_path, window_name):
    global polygon_points, drawing_done
    polygon_points = []
    drawing_done = False

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Cannot read video: {video_path}")

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        temp = frame.copy()

        for p in polygon_points:
            cv2.circle(temp, p, 5, (0, 0, 255), -1)

        if len(polygon_points) > 1:
            cv2.polylines(
                temp,
                [np.array(polygon_points, np.int32)],
                False,
                (0, 255, 0),
                2
            )

        cv2.putText(
            temp,
            "Left: add | Right: undo | ENTER: save | R: reset",
            (10, FRAME_HEIGHT - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.imshow(window_name, temp)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and len(polygon_points) >= 3:
            drawing_done = True
            break
        elif key == ord('r'):
            polygon_points = []

    cv2.destroyWindow(window_name)
    return polygon_points

def save_roi(data):
    with open(ROI_FILE, "w") as f:
        json.dump(data, f)

def load_roi():
    with open(ROI_FILE, "r") as f:
        return json.load(f)

def inside_polygon(point, polygon):
    return cv2.pointPolygonTest(
        np.array(polygon, np.int32),
        point,
        False
    ) >= 0

def process_frame(frame, roi_polygon):
    count = 0
    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if inside_polygon((cx, cy), roi_polygon):
                count += 1
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.polylines(
        frame,
        [np.array(roi_polygon, np.int32)],
        True,
        (255, 0, 0),
        2
    )

    return frame, count

def compute_green_time(vehicle_count):
    t = BASE_GREEN_TIME + vehicle_count * TIME_PER_VEHICLE
    return int(max(MIN_GREEN, min(MAX_GREEN, t)))

def main():
    if not os.path.exists(ROI_FILE):
        print("[INFO] Starting calibration...")
        roi_data = {
            "road_1": calibrate_polygon(VIDEO_1, "Calibrate Road 1"),
            "road_2": calibrate_polygon(VIDEO_2, "Calibrate Road 2")
        }
        save_roi(roi_data)
        print("[INFO] Calibration saved.")

    roi_data = load_roi()

    cap1 = cv2.VideoCapture(VIDEO_1)
    cap2 = cv2.VideoCapture(VIDEO_2)

    current_green = "road_1"
    last_switch_time = time.time()
    current_green_time = MIN_GREEN
    last_green_road_1 = time.time()
    last_green_road_2 = time.time()

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        frame1 = cv2.resize(frame1, (FRAME_WIDTH, FRAME_HEIGHT))
        frame2 = cv2.resize(frame2, (FRAME_WIDTH, FRAME_HEIGHT))

        frame1, count1 = process_frame(frame1, roi_data["road_1"])
        frame2, count2 = process_frame(frame2, roi_data["road_2"])

        now = time.time()

        if now - last_switch_time >= current_green_time:
            if now - last_green_road_1 > MAX_RED_TIME:
                current_green = "road_1"
            elif now - last_green_road_2 > MAX_RED_TIME:
                current_green = "road_2"
            else:
                current_green = "road_1" if count1 >= count2 else "road_2"

            vehicles = count1 if current_green == "road_1" else count2
            current_green_time = compute_green_time(vehicles)
            last_switch_time = now

            if current_green == "road_1":
                last_green_road_1 = now
            else:
                last_green_road_2 = now

        signal_state = {
            "road_1": "GREEN" if current_green == "road_1" else "RED",
            "road_2": "GREEN" if current_green == "road_2" else "RED"
        }

        for frame, road, count in [
            (frame1, "road_1", count1),
            (frame2, "road_2", count2)
        ]:
            color = (0, 255, 0) if signal_state[road] == "GREEN" else (0, 0, 255)
            cv2.putText(frame, f"Vehicles: {count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Signal: {signal_state[road]}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        combined = np.hstack((frame1, frame2))
        cv2.imshow("Adaptive Traffic Signal System", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()