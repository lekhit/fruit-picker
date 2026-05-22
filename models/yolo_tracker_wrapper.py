import os
import cv2
import numpy as np
import torch
import logging

from models.base_tracker import BaseTracker
from utils import get_vram_usage, log_exception

logger = logging.getLogger("AppleTrackerBenchmark.yolo_tracker")

YOLO_AVAILABLE = False
try:
    # Ultralytics provides YOLOv8 / YOLOv11 models and the YOLOWorld open-vocabulary API
    from ultralytics import YOLOWorld
    YOLO_AVAILABLE = True
except ImportError:
    logger.warning("Ultralytics library not found. Wrapper will fall back to CPU Centroid Tracker for dry-runs.")

class YOLOTrackerWrapper(BaseTracker):
    """
    Wrapper for YOLOv11-World + ByteTrack / BoT-SORT.
    Uses open-vocabulary detection and standard Kalman-based multi-object tracking (MOT).
    """
    def __init__(self, device="cpu", use_mock=False, model_version="yolov8s-worldv2.pt"):
        # We default to yolov8s-worldv2.pt / yolov11s-world.pt (supported by ultralytics)
        super().__init__(device, use_mock)
        self.model_version = model_version
        self.frame_idx = 0
        self.classes_set = False
        
        # State for Centroid Tracker CPU fallback
        self.next_track_id = 1
        self.disappeared_limit = 10
        # self.active_tracks: dict mapping ID -> {centroid, bbox, mask, disappeared_count}

    def initialize(self, prompt="apple"):
        logger.info(f"Initializing YOLO-World on device: {self.device} (Mock Mode: {self.use_mock or not YOLO_AVAILABLE})")
        
        if self.use_mock or not YOLO_AVAILABLE:
            logger.info("YOLO-World running in CPU Fallback/Mock Mode (Red color segmenter + Centroid Tracker).")
            self.clear_memory()
            return

        try:
            # 1. Load the pre-trained open-vocabulary YOLO-World model
            logger.info(f"Loading YOLO-World model: {self.model_version}")
            self.model = YOLOWorld(self.model_version)
            
            # Move to target device (cpu, cuda, mps)
            self.model.to(self.device)
            
            # 2. Configure open-vocabulary search classes
            logger.info(f"Configuring YOLO-World target concept class: ['{prompt}']")
            self.model.set_classes([prompt])
            self.classes_set = True
            
            self.frame_idx = 0
            mem = get_vram_usage()
            logger.info(f"YOLO-World initialized successfully. VRAM Allocated: {mem['allocated_mb']} MB")
            
        except Exception as e:
            log_exception(logger, "Failed to initialize CUDA YOLO-World. Falling back to CPU Mock mode", e)
            self.use_mock = True
            self.clear_memory()

    def process_frame(self, image):
        """
        Processes a single frame using YOLO-World tracking or CPU fallback.
        """
        if self.use_mock or not YOLO_AVAILABLE:
            return self._process_frame_fallback(image)

        # --- Real YOLO-World + ByteTrack/BoT-SORT Inference ---
        try:
            # Run tracker on the single frame image
            # tracker="bytetrack.yaml" or "botsort.yaml"
            # persist=True ensures tracking IDs are maintained across frames
            results = self.model.track(
                source=image,
                persist=True,
                tracker="bytetrack.yaml",
                device=self.device,
                verbose=False
            )
            
            outputs = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes_data = results[0].boxes
                
                # Verify tracker IDs are assigned
                if boxes_data.id is not None:
                    ids = boxes_data.id.cpu().numpy().astype(int)
                    xyxy = boxes_data.xyxy.cpu().numpy()
                    scores = boxes_data.conf.cpu().numpy()
                    
                    # Generate masks if available (YOLO-World is typically box-only,
                    # so we generate box-based binary masks or return None)
                    H, W, _ = image.shape
                    
                    for i in range(len(ids)):
                        tid = int(ids[i])
                        box = [int(xyxy[i][1]), int(xyxy[i][0]), int(xyxy[i][3]), int(xyxy[i][2])] # [ymin, xmin, ymax, xmax]
                        score = float(scores[i])
                        
                        # Generate box mask
                        box_mask = np.zeros((H, W), dtype=np.uint8)
                        cv2.rectangle(box_mask, (box[1], box[0]), (box[3], box[2]), 1, thickness=-1)
                        
                        outputs.append({
                            "id": tid,
                            "box": box,
                            "mask": box_mask,
                            "score": score
                        })
            
            self.frame_idx += 1
            return outputs

        except Exception as e:
            log_exception(logger, f"Error processing frame {self.frame_idx} in YOLO-World tracking", e)
            return []

    def _process_frame_fallback(self, image):
        """
        CPU Mock Mode: Segment apples by detecting red/green circular shapes in HSV color space
        and associate them across frames using a Centroid Tracker.
        """
        # Mirror the same robust centroid tracking fallback as SAM-3 and DEVA for consistency
        H, W, C = image.shape
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Color boundaries
        lower_red1, upper_red1 = np.array([0, 70, 50]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 70, 50]), np.array([180, 255, 255])
        lower_green, upper_green = np.array([35, 40, 40]), np.array([85, 255, 255])
        
        mask = (cv2.inRange(hsv, lower_red1, upper_red1) | 
                cv2.inRange(hsv, lower_red2, upper_red2) | 
                cv2.inRange(hsv, lower_green, upper_green))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 150 < area < 20000:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                centroid = np.array([x + w_box/2.0, y + h_box/2.0])
                
                inst_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(inst_mask, [cnt], -1, 1, thickness=-1)
                
                detections.append({
                    "centroid": centroid,
                    "box": [y, x, y + h_box, x + w_box],
                    "mask": inst_mask,
                    "score": round(min(0.5 + area/50000.0, 0.99), 2)
                })

        updated_tracks = {}
        unused_detections = list(range(len(detections)))
        
        if len(self.active_tracks) > 0 and len(detections) > 0:
            track_ids = list(self.active_tracks.keys())
            track_centroids = np.array([self.active_tracks[tid]["centroid"] for tid in track_ids])
            det_centroids = np.array([det["centroid"] for det in detections])
            
            dists = np.linalg.norm(track_centroids[:, np.newaxis] - det_centroids, axis=2)
            rows = dists.min(axis=1).argsort()
            cols = dists.argmin(axis=1)
            
            used_rows, used_cols = set(), set()
            for r in rows:
                if r in used_rows:
                    continue
                c = cols[r]
                if c in used_cols:
                    continue
                    
                if dists[r, c] < 120.0:
                    tid = track_ids[r]
                    det = detections[c]
                    updated_tracks[tid] = {
                        "centroid": det["centroid"],
                        "box": det["box"],
                        "mask": det["mask"],
                        "score": det["score"],
                        "disappeared_count": 0
                    }
                    used_rows.add(r)
                    used_cols.add(c)
                    if c in unused_detections:
                        unused_detections.remove(c)
                        
            for r in range(len(track_ids)):
                if r not in used_rows:
                    tid = track_ids[r]
                    tdata = self.active_tracks[tid]
                    tdata["disappeared_count"] += 1
                    if tdata["disappeared_count"] <= self.disappeared_limit:
                        updated_tracks[tid] = tdata

        elif len(self.active_tracks) > 0:
            for tid, tdata in self.active_tracks.items():
                tdata["disappeared_count"] += 1
                if tdata["disappeared_count"] <= self.disappeared_limit:
                    updated_tracks[tid] = tdata
                    
        for c in unused_detections:
            det = detections[c]
            updated_tracks[self.next_track_id] = {
                "centroid": det["centroid"],
                "box": det["box"],
                "mask": det["mask"],
                "score": det["score"],
                "disappeared_count": 0
            }
            self.next_track_id += 1

        self.active_tracks = updated_tracks
        
        results = []
        for tid, tdata in self.active_tracks.items():
            if tdata["disappeared_count"] == 0:
                results.append({
                    "id": tid,
                    "box": tdata["box"],
                    "mask": tdata["mask"],
                    "score": tdata["score"]
                })
        
        self.frame_idx += 1
        return results

    def clear_memory(self):
        logger.info("Clearing YOLO tracking history state.")
        self.frame_idx = 0
        self.active_tracks = {}
        self.next_track_id = 1
        
        if not self.use_mock and YOLO_AVAILABLE and self.model is not None:
            # YOLO internally maintains tracker state within its predictor session,
            # resetting classes or reinstantiating resets it
            if self.classes_set:
                try:
                    self.model.predictor = None # Clears internal predictor history
                    logger.info("YOLO active predictor session reset.")
                except Exception as e:
                    logger.warning(f"Error during YOLO internal reset: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
