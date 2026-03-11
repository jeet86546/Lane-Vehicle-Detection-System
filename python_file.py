from ultralytics import YOLO
import cv2
import numpy as np

VIDEO_PATH = r"C:\Users\patel\Downloads\curved_lane.mp4"

det_model = YOLO("yolov8n.pt")
seg_model = YOLO("yolov8n-seg.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Video not opened")
    exit()

print("✅ Video opened")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No frame")
        break

    # -------- Segmentation --------
    seg_res = seg_model(frame, imgsz=640, conf=0.25)[0]

    if seg_res.masks is not None:
        print("✅ Masks detected")
        for mask in seg_res.masks.data:
            m = (mask.cpu().numpy() * 255).astype(np.uint8)
            m = cv2.resize(m, (frame.shape[1], frame.shape[0]))
            frame[m > 0] = (0, 255, 0)
    else:
        print("⚠️ No masks")

    # -------- Detection --------
    det_res = det_model(frame, imgsz=640, conf=0.25)[0]
    for box in det_res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = det_model.names[int(box.cls[0])]
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(frame, cls, (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    cv2.imshow("DEBUG OUTPUT", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
