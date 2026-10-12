from fastapi import FastAPI, WebSocket, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import asyncio
from gesture_detector import GestureDetector
from gesture_map import list_all_gestures, get_gesture_description, GESTURE_VOCABULARY
import json
from typing import Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gesture Recognition API",
    description="Real-time gesture detection and translation to text for communication",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize gesture detector
gesture_detector = GestureDetector()

# Global state for streaming
class StreamState:
    def __init__(self):
        self.cap = None
        self.is_streaming = False
        self.latest_detection = None
        self.detection_history = []
        self.max_history = 100
    
    def add_detection(self, detection: Dict):
        self.latest_detection = detection
        self.detection_history.append(detection)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)

stream_state = StreamState()

# ============ API Endpoints ============

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple health check page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gesture Recognition API</title>
        <style>
            body { font-family: Arial; margin: 40px; background: #f0f0f0; }
            .container { background: white; padding: 20px; border-radius: 8px; }
            h1 { color: #333; }
            .endpoint { margin: 10px 0; padding: 10px; background: #e8f4f8; border-left: 4px solid #0066cc; }
            code { background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 Gesture Recognition API</h1>
            <p>Real-time gesture detection and translation to text</p>
            
            <h2>Endpoints:</h2>
            <div class="endpoint"><code>GET /health</code> - Health check</div>
            <div class="endpoint"><code>GET /gestures</code> - List all supported gestures</div>
            <div class="endpoint"><code>GET /gesture/{gesture_name}</code> - Get gesture details</div>
            <div class="endpoint"><code>POST /detect</code> - Detect gesture from image</div>
            <div class="endpoint"><code>WebSocket /ws</code> - Real-time webcam stream</div>
            <div class="endpoint"><code>GET /stream</code> - Get latest detection</div>
            <div class="endpoint"><code>GET /history</code> - Get detection history</div>
            
            <h2>Documentation:</h2>
            <p><a href="/docs">Interactive API Docs (Swagger UI)</a></p>
            <p><a href="/redoc">ReDoc Documentation</a></p>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "gesture-recognition-api",
        "version": "1.0.0"
    }

@app.get("/gestures")
async def get_gestures():
    """Get list of all supported gestures with descriptions."""
    gestures = []
    for gesture_name, gesture_info in GESTURE_VOCABULARY.items():
        gestures.append({
            "name": gesture_name,
            "text": gesture_info.get("text"),
            "command": gesture_info.get("command"),
            "description": gesture_info.get("description"),
            "confidence_threshold": gesture_info.get("confidence_threshold")
        })
    return {
        "total": len(gestures),
        "gestures": gestures
    }

@app.get("/gesture/{gesture_name}")
async def get_gesture_info(gesture_name: str):
    """Get detailed information about a specific gesture."""
    if gesture_name not in GESTURE_VOCABULARY:
        raise HTTPException(status_code=404, detail=f"Gesture '{gesture_name}' not found")
    
    gesture_info = GESTURE_VOCABULARY[gesture_name]
    return {
        "name": gesture_name,
        "text": gesture_info.get("text"),
        "command": gesture_info.get("command"),
        "description": gesture_info.get("description"),
        "confidence_threshold": gesture_info.get("confidence_threshold")
    }

@app.post("/detect")
async def detect_gesture(file: UploadFile = File(...)):
    """
    Detect gesture from an uploaded image.
    
    Returns:
        - gesture: Name of detected gesture
        - text: Text translation of gesture
        - command: Command associated with gesture
        - confidence: Confidence score (0-1)
    """
    try:
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Detect gesture
        result = gesture_detector.detect_gestures(frame)
        
        return {
            "gesture": result["gesture"],
            "text": result["text"],
            "command": result["command"],
            "confidence": float(result["confidence"]),
            "detected_hands": result["detected_hands"],
            "pose_detected": result["pose_detected"]
        }
    except Exception as e:
        logger.error(f"Error in gesture detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/stream")
async def get_latest_detection():
    """
    Get the latest gesture detection result from the stream.
    """
    if stream_state.latest_detection is None:
        return {"status": "No detections yet", "detection": None}
    
    detection = stream_state.latest_detection
    return {
        "gesture": detection["gesture"],
        "text": detection["text"],
        "command": detection["command"],
        "confidence": float(detection["confidence"]),
        "detected_hands": detection["detected_hands"],
        "pose_detected": detection["pose_detected"]
    }

@app.get("/history")
async def get_detection_history(limit: int = 20):
    """
    Get recent gesture detection history.
    """
    history = stream_state.detection_history[-limit:]
    result = []
    
    for detection in history:
        result.append({
            "gesture": detection["gesture"],
            "text": detection["text"],
            "command": detection["command"],
            "confidence": float(detection["confidence"])
        })
    
    return {
        "total_recorded": len(stream_state.detection_history),
        "returned": len(result),
        "history": result
    }

# ============ WebSocket for Real-time Streaming ============

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time gesture detection.
    
    Expected client message format: {"action": "start" | "stop"}
    Server sends: {"gesture": ..., "text": ..., "confidence": ..., "frame_base64": ...}
    """
    await websocket.accept()
    cap = None
    
    try:
        while True:
            # Receive client message
            data = await websocket.receive_text()
            message = json.loads(data)
            action = message.get("action")
            
            if action == "start":
                # Initialize camera
                if cap is None:
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        await websocket.send_json({"error": "Cannot open camera"})
                        continue
                
                stream_state.is_streaming = True
                await websocket.send_json({"status": "streaming_started"})
                
                # Streaming loop
                while stream_state.is_streaming:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Resize for faster processing
                    frame = cv2.resize(frame, (640, 480))
                    
                    # Detect gesture
                    detection = gesture_detector.detect_gestures(frame)
                    stream_state.add_detection(detection)
                    
                    # Draw landmarks
                    frame_with_landmarks = gesture_detector.draw_landmarks(frame, detection)
                    
                    # Encode frame
                    _, buffer = cv2.imencode('.jpg', frame_with_landmarks)
                    frame_base64 = buffer.tobytes()
                    
                    # Send detection result
                    await websocket.send_json({
                        "gesture": detection["gesture"],
                        "text": detection["text"],
                        "command": detection["command"],
                        "confidence": float(detection["confidence"]),
                        "detected_hands": detection["detected_hands"],
                        "pose_detected": detection["pose_detected"]
                    })
                    
                    # Small delay to prevent overwhelming the client
                    await asyncio.sleep(0.03)  # ~30 FPS
            
            elif action == "stop":
                stream_state.is_streaming = False
                if cap is not None:
                    cap.release()
                    cap = None
                await websocket.send_json({"status": "streaming_stopped"})
    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        stream_state.is_streaming = False
        if cap is not None:
            cap.release()
    
    finally:
        if cap is not None:
            cap.release()
        await websocket.close()

# ============ Startup/Shutdown ============

@app.on_event("startup")
async def startup_event():
    logger.info("Gesture Recognition API started")
    logger.info(f"Loaded {len(GESTURE_VOCABULARY)} gesture definitions")

@app.on_event("shutdown")
async def shutdown_event():
    if stream_state.cap is not None:
        stream_state.cap.release()
    logger.info("Gesture Recognition API stopped")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
