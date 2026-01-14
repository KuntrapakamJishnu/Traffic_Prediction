from fastapi import FastAPI
from pydantic import BaseModel, ValidationError
from typing import List, Optional
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import numpy as np
import psycopg2
import joblib
import tensorflow as tf
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
import json
from datetime import datetime
import asyncio


model = tf.keras.models.load_model(
    "model/gru_lstm_model.h5",
    compile=False
)
scaler = joblib.load("model/scaler.pkl")

# ==============================
# FASTAPI INITIALIZATION
# ==============================
app = FastAPI(title="Traffic Intelligence API")

# Mount a small static folder for websocket demo
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception:
    # static folder may not exist in some environments; ignore
    pass


# Simple WebSocket connection manager to broadcast predictions
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        try:
            self.active_connections.remove(websocket)
        except ValueError:
            pass

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception:
                # ignore websocket send errors
                pass


manager = ConnectionManager()

# Async broadcast infrastructure
ASYNC_LOOP = None
BROADCAST_QUEUE: "asyncio.Queue[str]" = None
SSE_CLIENT_QUEUES: set = set()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # keep the connection alive; we do not expect client messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.on_event("startup")
async def _startup_event():
    global ASYNC_LOOP, BROADCAST_QUEUE
    ASYNC_LOOP = asyncio.get_event_loop()
    BROADCAST_QUEUE = asyncio.Queue()

    async def _broadcast_loop():
        while True:
            msg = await BROADCAST_QUEUE.get()
            # broadcast to websocket clients
            try:
                await manager.broadcast(msg)
            except Exception:
                pass
            # push to SSE client queues
            for q in list(SSE_CLIENT_QUEUES):
                try:
                    # don't await put if queue is full, use create_task
                    await q.put(msg)
                except Exception:
                    pass

    asyncio.create_task(_broadcast_loop())


@app.get("/sse")
async def sse_endpoint(request: Request):
    """Server-Sent Events endpoint. Clients receive newline-separated 'data: ...' messages."""
    q: asyncio.Queue = asyncio.Queue()
    SSE_CLIENT_QUEUES.add(q)

    async def event_generator():
        try:
            print("[SSE] client connected")
            while True:
                # If client disconnects, break cleanly
                if await request.is_disconnected():
                    print("[SSE] client disconnected")
                    break

                try:
                    # wait for a message with timeout so we can send heartbeats
                    msg = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {msg}\n\n"
                except asyncio.TimeoutError:
                    # heartbeat comment to keep connection alive
                    yield ": heartbeat\n\n"
                except asyncio.CancelledError:
                    break
        finally:
            try:
                SSE_CLIENT_QUEUES.discard(q)
            except Exception:
                pass

    from fastapi.responses import StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ==============================
# POSTGRESQL CONFIG (UPDATED)
# ==============================
DB_CONFIG = {
    "dbname": "traffic_db",
    "user": "postgres",
    "password": "postgres123",  
    "host": "localhost",
    "port": 5433                  
}

def log_prediction(predicted_flow, uncertainty, confidence, location):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # detect which column exists in the table and insert accordingly
    cur.execute(
        """
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'traffic_predictions' AND column_name IN ('location', 'location_id')
        """
    )
    cols = {r[0] for r in cur.fetchall()}

    if 'location' in cols:
        cur.execute("""
            INSERT INTO traffic_predictions
            (predicted_flow, uncertainty, confidence, location)
            VALUES (%s, %s, %s, %s)
        """, (predicted_flow, uncertainty, confidence, location))
    elif 'location_id' in cols:
        # try to coerce textual location to integer id, else insert NULL
        try:
            lid = int(location)
        except Exception:
            lid = None
        cur.execute("""
            INSERT INTO traffic_predictions
            (predicted_flow, uncertainty, confidence, location_id)
            VALUES (%s, %s, %s, %s)
        """, (predicted_flow, uncertainty, confidence, lid))
    else:
        # neither column exists; insert without location
        cur.execute("""
            INSERT INTO traffic_predictions
            (predicted_flow, uncertainty, confidence)
            VALUES (%s, %s, %s)
        """, (predicted_flow, uncertainty, confidence))

    conn.commit()
    cur.close()
    conn.close()
    # Also print a short summary for server logs and broadcast to websocket clients
    try:
        summary = {
            "ts": datetime.now().isoformat() + "Z",
            "location": location,
            "predicted_flow": float(predicted_flow),
            "uncertainty": float(uncertainty),
            "confidence": confidence,
        }
        msg = json.dumps(summary)
        print("[PREDICTION_LOG]", msg)
        # enqueue for async broadcast (SSE + websocket) if loop available
        try:
            if ASYNC_LOOP is not None and ASYNC_LOOP.is_running() and BROADCAST_QUEUE is not None:
                try:
                    # schedule put onto the broadcast queue
                    asyncio.run_coroutine_threadsafe(BROADCAST_QUEUE.put(msg), ASYNC_LOOP)
                except Exception:
                    print("[PREDICTION_LOG] Failed to schedule broadcast")
            else:
                print("[PREDICTION_LOG] Broadcast skipped (async loop not available)")
        except Exception:
            pass
    except Exception:
        pass



# ==============================
# REQUEST SCHEMA
# ==============================
class TrafficInput(BaseModel):
    sequence: List[List[float]]
    location: Optional[str] = None  # preferred: textual edge id
    location_id: Optional[int] = None  # legacy numeric id

    def get_location_text(self) -> Optional[str]:
        if self.location:
            return str(self.location)
        if self.location_id is not None:
            return str(self.location_id)
        return None


# Add handler to return useful messages on validation error and log payloads
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        body = await request.json()
    except Exception:
        body = None
    # log to console for debugging
    print("[VALIDATION_ERROR]", exc.errors(), "body=", body)
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": body},
    )

# ==============================
# PREDICTION ENDPOINT
# ==============================
@app.post("/predict")
def predict(data: TrafficInput):

    X = np.array(data.sequence).reshape(
        1, len(data.sequence), 3
    )

    # Monte Carlo Dropout
    preds = [
        model(X, training=True).numpy()[0][0]
        for _ in range(30)
    ]

    mean_pred = np.mean(preds)
    std_pred = np.std(preds)

    # Inverse scaling (flow only)
    dummy = np.zeros((1, 3))
    dummy[0, 0] = mean_pred
    real_flow = scaler.inverse_transform(dummy)[0][0]

    # Confidence logic
    if std_pred < 0.01:
        confidence = "HIGH"
    elif std_pred < 0.03:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Save to PostgreSQL using textual location identifier
    # prefer textual location; fall back to numeric id if provided
    location = data.get_location_text()

    log_prediction(
        predicted_flow=float(real_flow),
        uncertainty=float(std_pred),
        confidence=confidence,
        location=location
    )
    # Additionally return the textual location for clients
    

    return {
        "predicted_flow": float(real_flow),
        "uncertainty": float(std_pred),
        "confidence": confidence,
        "location": location,

    }