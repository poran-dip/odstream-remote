"""
Config file for remote camera manager
"""

from pathlib import Path


class Config:
    """Configure your YOLO model, path, and type (official or custom)"""
    OFFICIAL_NEW_MODEL = True  # v5nu, v8n, etc. do not use for v5n or custom models
    YOLO_MODEL_VERSION = "v5"
    YOLO_MODEL_PATH = "models/yolov5nu.pt"

    # Camera settings
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30
    JPEG_QUALITY = 80

    @classmethod
    def validate(cls):
        """Validate the configurations"""
        if not Config.YOLO_MODEL_PATH.endswith(".pt"):
            raise ValueError(f"YOLO model path must be a .pt file, got {Config.YOLO_MODEL_PATH}")
        if not Path(Config.YOLO_MODEL_PATH).exists():
            print(f"Warning: YOLO model file does not exist at {Config.YOLO_MODEL_PATH}")
        if Config.YOLO_MODEL_VERSION not in {"v5", "v8"}:
            raise ValueError(f"YOLO version must be 'v5' or 'v8', got {Config.YOLO_MODEL_VERSION}")
        print("Config validated successfully.")

def main():
    """Validate and display on running config file"""
    Config.validate()
    print(f"Config: \n\
          \tModel = {Config.YOLO_MODEL_PATH} \n\
          \tType = {'Official' if Config.OFFICIAL_NEW_MODEL else 'Custom'}")

if __name__ == "__main__":
    main()
