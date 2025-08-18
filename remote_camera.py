"""
The actual remote camera manager that defines all methods
"""

from pathlib import Path
import sys
import threading
import time
import warnings
import cv2
import pyrealsense2 as rs
import numpy as np
import torch
from ultralytics import YOLO
from config import Config

class RemoteCameraManager:
    """Manages RealSense camera capture and YOLO object detection/recording"""
    def __init__(self):
        Config.validate()

        self.model = None
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.camera_lock = threading.Lock()
        self.is_active = False
        self.is_recording = False

    def load_model(self):
        """Load YOLO model based on config version (v5 or v8)"""
        try:
            print(
                f"Loading YOLO model ({Config.YOLO_MODEL_VERSION}) from {Config.YOLO_MODEL_PATH}"
            )

            if Config.OFFICIAL_NEW_MODEL or Config.YOLO_MODEL_VERSION == "v8":
                # YOLOv8 (Ultralytics official)
                self.model = YOLO(Config.YOLO_MODEL_PATH)

            elif Config.YOLO_MODEL_VERSION == "v5":
                # YOLOv5 custom repo
                yolov5_path = Path(__file__).parent / "yolov5"
                if str(yolov5_path) not in sys.path:
                    sys.path.append(str(yolov5_path))

                # Suppress annoying future warnings (e.g., amp.autocast)
                warnings.filterwarnings("ignore", category=FutureWarning)

                self.model = torch.hub.load(
                    str(yolov5_path),
                    "custom",
                    path=str(Path(Config.YOLO_MODEL_PATH)),
                    source="local",
                    force_reload=True,
                )
                self.model.eval()

            else:
                raise ValueError(
                    f"Unsupported YOLO version: {Config.YOLO_MODEL_VERSION}"
                )

            print("Model loaded successfully!")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def start_camera(self):
        """Start RealSense camera capture"""
        try:
            print("Starting RealSense camera...")

            with self.camera_lock:
                if self.is_active:
                    return False, "Camera already active"

                # Configure RealSense streams
                self.config.enable_stream(rs.stream.color,
                                        Config.CAMERA_WIDTH,
                                        Config.CAMERA_HEIGHT,
                                        rs.format.bgr8,
                                        Config.CAMERA_FPS)

                # Start the pipeline
                self.pipeline.start(self.config)

                # Test frame capture with timeout
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                    color_frame = frames.get_color_frame()

                    if not color_frame:
                        self.pipeline.stop()
                        return False, "RealSense opened but failed to capture frame"

                    self.is_active = True
                    print("RealSense camera started successfully")
                    return True, "RealSense camera started successfully"

                except RuntimeError as e:
                    # RealSense timeout
                    self.pipeline.stop()
                    return False, f"Camera timeout during startup: {e}"

        except Exception as e:
            print(f"Error starting RealSense camera: {e}")
            return False, str(e)

    def stop_camera(self):
        """Stop RealSense camera capture"""
        try:
            with self.camera_lock:
                if not self.is_active:
                    return False, "Camera not active"

                self.pipeline.stop()
                self.is_active = False
                self.is_recording = False  # Stop recording when camera stops
                print("RealSense camera stopped")
                return True, "RealSense camera stopped successfully"

        except Exception as e:
            print(f"Error stopping RealSense camera: {e}")
            return False, str(e)

    def _annotate_frame(self, frame):
        """
        Annotate frame using YOLO model results.
        Handles YOLOv5 render() and YOLOv8 plot() differences safely.
        """
        try:
            results = None

            if Config.OFFICIAL_NEW_MODEL or Config.YOLO_MODEL_VERSION == "v8":
                results = self.model(frame, verbose=False)
                # handle list return type
                if (
                    isinstance(results, list)
                    and len(results) > 0
                    and hasattr(results[0], "plot")
                ):
                    return results[0].plot(), results
                if hasattr(results, "plot"):
                    return results.plot(), results

            elif Config.YOLO_MODEL_VERSION == "v5":
                results = self.model(frame)
                if hasattr(results, "render"):
                    return results.render()[0], results

            return frame, results  # fallback

        except Exception as e:
            print(f"Error annotating frame: {e}")
            return frame, results

    def start_recording(self):
        """Start YOLO recording"""
        try:
            if not self.is_active:
                return False, "Camera not active"

            if self.is_recording:
                return False, "Already recording"

            # Load model if not loaded
            if self.model is None:
                if not self.load_model():
                    return False, "Failed to load YOLO model"

            self.is_recording = True
            print("YOLO recording started")
            return True, "Recording started"

        except Exception as e:
            print(f"Error starting recording: {e}")
            return False, str(e)

    def stop_recording(self):
        """Stop YOLO recording"""
        try:
            if not self.is_recording:
                return False, "Not currently recording"

            self.is_recording = False
            print("YOLO recording stopped")
            return True, "Recording stopped"

        except Exception as e:
            print(f"Error stopping recording: {e}")
            return False, str(e)

    def is_camera_active(self):
        """Check if RealSense camera is active"""
        return self.is_active

    def is_recording_active(self):
        """Check if YOLO recording is active"""
        return self.is_recording

    def _process_yolov5_results(self, frame, results):
        """Process YOLOv5 results and draw bounding boxes"""
        try:
            # YOLOv5 returns detections in format [x1, y1, x2, y2, conf, class]
            detections = results.xyxy[0].cpu().numpy()  # Get first image results

            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                if conf > 0.5:  # Confidence threshold
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # Add label
                    label = f'{int(cls)}: {conf:.2f}'
                    cv2.putText(frame, label, (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return frame
        except Exception as e:
            print(f"Error processing YOLOv5 results: {e}")
            return frame

    def generate_frames(self):
        """Generate frames with YOLO when recording"""
        print("Starting frame generation...")
        failed_frames = 0
        max_failed_frames = 30

        while True:
            if not self.is_camera_active():
                print("Camera stopped, ending stream...")
                break

            try:
                with self.camera_lock:
                    if not self.is_active:
                        break

                    try:
                        frames = self.pipeline.wait_for_frames()
                        color_frame = frames.get_color_frame()
                        if not color_frame:
                            failed_frames += 1
                            if failed_frames >= max_failed_frames:
                                print("Too many failed frames, ending stream...")
                                break
                            time.sleep(0.1)
                            continue

                        frame = np.asanyarray(color_frame.get_data())

                    except Exception as e:
                        failed_frames += 1
                        print(f"Failed to get RealSense frame: {e}")
                        if failed_frames >= max_failed_frames:
                            print("Too many failed frames, ending stream...")
                            break
                        time.sleep(0.1)
                        continue

                # Reset failed frames counter on successful frame
                failed_frames = 0

                # Apply YOLO detection only when recording
                if self.is_recording and self.model is not None:
                    processed_frame, _ = self._annotate_frame(frame)
                else:
                    processed_frame = frame

                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', processed_frame,
                                        [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY])

                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    print("Failed to encode frame")
                    break

            except Exception as e:
                print(f"Error processing frame: {e}")
                break

        print("Frame generation ended")
