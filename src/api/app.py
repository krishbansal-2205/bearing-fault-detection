"""
Production-ready REST API for bearing fault detection.

Features:
- Real-time inference
- Input validation
- Error handling
- Request logging
- Model versioning
- Health monitoring
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import torch.nn.functional as F
import numpy as np
from io import BytesIO
import logging
from datetime import datetime
from typing import List, Optional, Dict
import time
from contextlib import asynccontextmanager
try:
    import onnxruntime as ort
except ImportError:
    ort = None

from src.models.vibration_cnn import VibrationCNN
from src.data.preprocessing import bandpass_filter, normalize_signal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lifespan manager to replace startup event
@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_model()
    yield

# Initialize FastAPI app
app = FastAPI(
    title="Bearing Fault Detection API",
    description="Real-time bearing health monitoring using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
MODEL = None
ONNX_SESSION = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CLASS_LABELS = {
    0: "Normal",
    1: "Inner_Race_007",
    2: "Inner_Race_014",
    3: "Inner_Race_021",
    4: "Outer_Race_007",
    5: "Outer_Race_014",
    6: "Outer_Race_021",
    7: "Ball_007",
    8: "Ball_014",
    9: "Ball_021"
}

# Pydantic models for request/response validation


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predicted_class: str = Field(..., description="Predicted fault type")
    predicted_class_id: int = Field(..., description="Class ID (0-9)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    alert_level: str = Field(..., description="Alert status")
    top_3_predictions: List[Dict[str, float]
                            ] = Field(..., description="Top 3 predictions")
    processing_time_ms: float = Field(...,
                                      description="Inference time in milliseconds")
    timestamp: str = Field(..., description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    device: str
    version: str
    timestamp: str


async def load_model():
    """Load model on startup."""
    global MODEL, ONNX_SESSION, DEVICE

    try:
        logger.info("Loading model...")

        # Try to load ONNX model first (faster inference)
        try:
            if ort is None:
                raise ImportError("onnxruntime not installed")
            ONNX_SESSION = ort.InferenceSession(
                "models/bearing_fault_detector.onnx",
                providers=['CPUExecutionProvider']
            )
            logger.info("✓ ONNX model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load ONNX model: {e}")

            # Fallback to PyTorch model
            MODEL = VibrationCNN(num_classes=10)
            checkpoint = torch.load(
                'models/best_model.pth',
                map_location=DEVICE,
                weights_only=False
            )
            MODEL.load_state_dict(checkpoint['model_state_dict'])
            MODEL.to(DEVICE)
            MODEL.eval()
            logger.info(f"✓ PyTorch model loaded successfully on {DEVICE}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def preprocess_signal(signal: np.ndarray, fs: int = 12000) -> np.ndarray:
    """
    Preprocess raw vibration signal.

    Args:
        signal: Raw signal array
        fs: Sampling frequency

    Returns:
        Preprocessed signal ready for model input
    """
    # Ensure signal is 1D
    if signal.ndim > 1:
        signal = signal.flatten()

    # Check minimum length
    if len(signal) < 2048:
        raise ValueError(
            f"Signal too short ({len(signal)} samples). Minimum 2048 required.")

    # Take first 2048 samples
    signal = signal[:2048]

    # Apply bandpass filter
    try:
        signal = bandpass_filter(signal, lowcut=10, highcut=5000, fs=fs)
    except Exception as e:
        logger.error(f"Bandpass filter failed: {e}")
        raise ValueError(f"Signal filtering failed: {e}")

    # Normalize
    signal = normalize_signal(signal, method='zscore')

    return signal


def get_alert_level(normal_prob: float, max_fault_prob: float) -> str:
    """
    Determine maintenance alert level.

    Args:
        normal_prob: Probability of normal operation
        max_fault_prob: Maximum probability among fault classes

    Returns:
        Alert level string
    """
    if normal_prob > 0.9:
        return "GREEN - Normal Operation"
    elif max_fault_prob > 0.7:
        return "RED - Critical Fault Detected"
    elif max_fault_prob > 0.4:
        return "YELLOW - Degradation Detected"
    else:
        return "ORANGE - Uncertain - Monitor Closely"


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with basic info."""
    return await health_check()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL is not None or ONNX_SESSION is not None,
        device=DEVICE,
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_fault(
    file: UploadFile = File(...,
                            description="Vibration signal file (.csv or .npy)"),
    sampling_rate: int = 12000
):
    """
    Predict bearing health state from vibration signal.

    **Input**: CSV or NumPy array (.npy) with shape [N] or [N, 1]

    **Output**: JSON with fault prediction and confidence scores

    **Example**:
    ```bash
    curl -X POST "http://localhost:8000/predict" \\
      -F "file=@vibration_signal.npy" \\
      -F "sampling_rate=12000"
    ```
    """
    start_time = time.time()

    try:
        # Read file
        contents = await file.read()

        # Parse based on file type
        if file.filename.endswith('.csv'):
            signal = np.loadtxt(BytesIO(contents), delimiter=',')
        elif file.filename.endswith('.npy'):
            signal = np.load(BytesIO(contents))
        elif file.filename.endswith('.txt'):
            signal = np.loadtxt(BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400,
                detail="File must be .csv, .npy, or .txt format"
            )

        # Preprocess signal
        try:
            signal_processed = preprocess_signal(signal, fs=sampling_rate)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Prepare input tensor
        signal_tensor = torch.FloatTensor(
            signal_processed).unsqueeze(0).unsqueeze(0)

        # Inference
        if ONNX_SESSION is not None:
            # ONNX inference
            onnx_input = {ONNX_SESSION.get_inputs(
            )[0].name: signal_tensor.numpy()}
            logits = ONNX_SESSION.run(None, onnx_input)[0]
            probs = torch.softmax(torch.from_numpy(
                logits), dim=1).squeeze().numpy()
        else:
            # PyTorch inference
            with torch.no_grad():
                signal_tensor = signal_tensor.to(DEVICE)
                logits = MODEL(signal_tensor)
                probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        # Get top-3 predictions
        top3_idx = np.argsort(probs)[-3:][::-1]

        predicted_class_id = int(top3_idx[0])
        confidence = float(probs[predicted_class_id])

        # Determine alert level
        normal_prob = float(probs[0])
        max_fault_prob = float(np.max(probs[1:]))
        alert_level = get_alert_level(normal_prob, max_fault_prob)

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # ms

        # Build response
        response = PredictionResponse(
            predicted_class=CLASS_LABELS[predicted_class_id],
            predicted_class_id=predicted_class_id,
            confidence=confidence,
            alert_level=alert_level,
            top_3_predictions=[
                {
                    "class": CLASS_LABELS[int(idx)],
                    "class_id": int(idx),
                    "confidence": float(probs[idx])
                } for idx in top3_idx
            ],
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )

        # Log prediction
        logger.info(
            f"Prediction: {CLASS_LABELS[predicted_class_id]} "
            f"(conf={confidence:.3f}, time={processing_time:.1f}ms)"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    sampling_rate: int = 12000
):
    """
    Batch prediction endpoint for multiple signals.

    **Input**: List of vibration signal files

    **Output**: List of predictions
    """
    results = []

    for file in files:
        try:
            result = await predict_fault(file, sampling_rate=sampling_rate)
            results.append({
                "filename": file.filename,
                "prediction": result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return {"predictions": results}


@app.get("/classes")
async def get_classes():
    """Get list of supported fault classes."""
    return {
        "classes": CLASS_LABELS,
        "num_classes": len(CLASS_LABELS)
    }


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time*1000:.2f}ms"
    )

    return response


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
