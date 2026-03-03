# Underwater Object Detection — Remote Server

This repository serves as the companion backend for [ODStream](https://github.com/poran-dip/odstream), the main control interface for real-time object detection.

This backend is designed to be deployed on a Single Board Computer (SBC) such as the Jetson Nano. It handles **Intel RealSense camera streaming** and optionally performs **YOLOv5-based object detection**, making the results accessible to the main controller.

---

## Hardware Requirements

- Jetson Nano (or compatible SBC)
- Intel RealSense camera (connected via USB)
- Python 3.6+

---

## Setup

To set up the server on a Jetson Nano (or other compatible SBC), follow these steps:

```bash
git clone https://github.com/poran-dip/odstream-remote.git
cd odstream-remote
python3 -m venv venv
source venv/bin/activate
chmod +x setup.sh
bash setup.sh
python3 app.py
```

Ensure your Intel RealSense camera is connected via USB.

---

## API Endpoints

- `POST /api/start_camera` - Start live camera streaming
- `POST /api/stop_camera` - Stop camera streaming
- `POST /api/start_recording` - Start YOLO object detection
- `POST /api/stop_recording` - Stop YOLO detection (streaming continues)
- `GET /api/camera_status` - Check camera and recording status
- `GET /api/video_feed` - Live video stream (MJPEG format)
- `GET /api/health` - Health check

---

## Integration with ODStream

Once the server is running, obtain its network URL (e.g., `http://jetson-nano.local:5000`) and enter it in the UI of the ODStream project.

**Workflow:**
1. Start camera for live streaming
2. Start recording to enable YOLO detection
3. Stop recording to disable detection (stream continues)
4. Stop camera to end session

---

## Notes

- Uses **Intel RealSense SDK** for camera capture
- The default YOLO model is `yolov5n.pt`
- Live streaming is always available when camera is active
- YOLO detection only runs when recording is enabled
- Fully compatible with the Flask + React frontend provided by [ODStream](https://github.com/poran-dip/odstream)
