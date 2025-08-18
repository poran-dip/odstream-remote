"""
The main app module for UWOD-Remote
"""

from flask import Flask
from camera_routes import remote_bp

app = Flask(__name__)

# Register the Jetson camera blueprint
app.register_blueprint(remote_bp)

if __name__ == '__main__':
    try:
        print("Starting Jetson Camera Backend...")
        print("Endpoints available:")
        print("  POST /api/start_camera - Start camera")
        print("  POST /api/stop_camera - Stop camera")
        print("  POST /api/start_recording - Start YOLO recording")
        print("  POST /api/stop_recording - Stop YOLO recording")
        print("  GET  /api/camera_status - Check camera and recording status")
        print("  GET  /api/video_feed - Video stream (live always, YOLO when recording)")
        print("  GET  /api/health - Health check")
        print("  POST /api/shutdown - Shutdown server")

        # Run on all interfaces so it's accessible from other devices
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error starting server: {e}")
