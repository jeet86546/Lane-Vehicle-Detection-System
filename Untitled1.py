{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "5362952f-fdb1-473d-9522-19238d8d763f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Models loaded successfully\n",
      "Frame processed\n"
     ]
    }
   ],
   "source": [
    "from ultralytics import YOLO\n",
    "import cv2\n",
    "import numpy as np\n",
    "\n",
    "det_model = YOLO(\"yolov8n.pt\")       # vehicles\n",
    "seg_model = YOLO(\"yolov8n-seg.pt\")   # lanes / road\n",
    "\n",
    "cap = cv2.VideoCapture(\"curved_lane.mp4\")\n",
    "print(\"Models loaded successfully\")\n",
    "while True:\n",
    "    ret, frame = cap.read()\n",
    "    print(\"Frame processed\")\n",
    "\n",
    "    if not ret:\n",
    "        break\n",
    "\n",
    "    # -------- Lane segmentation --------\n",
    "    seg_res = seg_model(frame, imgsz=640, conf=0.4)[0]\n",
    "    if seg_res.masks is not None:\n",
    "        for mask in seg_res.masks.data:\n",
    "            m = (mask.cpu().numpy() * 255).astype(np.uint8)\n",
    "            m = cv2.resize(m, (frame.shape[1], frame.shape[0]))\n",
    "            frame[m > 0] = (0, 200, 0)   # green lanes\n",
    "\n",
    "    # -------- Vehicle detection --------\n",
    "    det_res = det_model(frame, imgsz=640, conf=0.4)[0]\n",
    "    for box in det_res.boxes:\n",
    "        x1, y1, x2, y2 = map(int, box.xyxy[0])\n",
    "        cls = det_model.names[int(box.cls[0])]\n",
    "        if cls in [\"car\", \"bus\", \"truck\", \"motorbike\"]:\n",
    "            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)\n",
    "            cv2.putText(frame, cls, (x1, y1-5),\n",
    "                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)\n",
    "\n",
    "    cv2.imshow(\"Lane + Vehicle System\", frame)\n",
    "    if cv2.waitKey(1) & 0xFF == ord('q'):\n",
    "        break\n",
    "\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bee30ef4-8f1f-494f-af06-51a6b37179ff",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eb5d1266-9054-46c1-be35-4592fea13a11",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
