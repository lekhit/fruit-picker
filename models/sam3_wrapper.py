import os
import cv2
import numpy as np
import torch
import logging

from models.base_tracker import BaseTracker
from utils import get_vram_usage, log_exception

logger = logging.getLogger("AppleTrackerBenchmark.sam3")

# Optional import of official SAM-3 library
SAM3_AVAILABLE = False
try:
    # This matches the official Meta facebookresearch/sam3 library structure
    from sam3.build_sam import build_sam3_video_predictor
    SAM3_AVAILABLE = True
except ImportError:
    logger.warning("Official 'sam3' library not found. Wrapper will fall back to CPU Centroid Tracker for dry-runs.")

class SAM3Wrapper(BaseTracker):
    """
    Wrapper for Meta's SAM-3 / SAM-3.1 (Segment Anything Model 3).
    Enforces deep learning memory optimizations:
      1. float16/bfloat16 precision
      2. torch.inference_mode()
      3. Image downsampling (1024x1024)
      4. Explicit memory bank eviction and GPU cache clearing
    """
    def __init__(self, device="cpu", use_mock=False, checkpoint_path="sam3_hiera_large.pt"):
        super().__init__(device, use_mock)
        self.checkpoint_path = checkpoint_path
        self.predictor = None
        self.inference_state = None
        self.frame_idx = 0
        
        # State for the fallback Centroid Tracker
        self.next_track_id = 1
        self.disappeared_limit = 10  # Max frames to keep track alive when occluded
        # self.active_tracks: dict mapping ID -> {centroid, bbox, mask, disappeared_count}

    def initialize(self, prompt="apple"):
        logger.info(f"Initializing SAM-3 on device: {self.device} (Mock Mode: {self.use_mock or not SAM3_AVAILABLE})")
        
        if self.use_mock or not SAM3_AVAILABLE:
            logger.info("SAM-3 running in CPU Fallback/Mock Mode (Red color segmenter + Centroid Tracker).")
            self.clear_memory()
            return

        # --- Real CUDA SAM-3 Optimization Setup ---
        try:
            # 1. Apply PyTorch VRAM configuration for fragmentation prevention
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            logger.info("Applied PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'")

            # 2. Determine precision (bfloat16 for newer GPUs, float16 for older GPUs like T4)
            self.precision = torch.float16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.precision = torch.bfloat16
            logger.info(f"SAM-3 Precision Selected: {self.precision}")

            # 3. Load SAM-3 model weights onto GPU
            logger.info(f"Loading SAM-3 checkpoint from: {self.checkpoint_path}")
            self.predictor = build_sam3_video_predictor(self.checkpoint_path)
            
            # Enforce half-precision
            self.predictor.model.to(device=self.device, dtype=self.precision)
            
            # Initialize streaming video state
            # SAM-3 holds attention keys/values inside its internal inference_state
            self.inference_state = self.predictor.init_state(video_path=None)
            self.frame_idx = 0
            
            mem = get_vram_usage()
            logger.info(f"SAM-3 successfully initialized. VRAM Allocated: {mem['allocated_mb']} MB, Peak VRAM: {mem['peak_mb']} MB")
            
        except Exception as e:
            log_exception(logger, "Failed to initialize CUDA SAM-3. Falling back to CPU Mock mode", e)
            self.use_mock = True
            self.clear_memory()

    def process_frame(self, image):
        """
        Processes a frame using SAM-3's concept propagation or CPU centroid tracking fallback.
        """
        if self.use_mock or not SAM3_AVAILABLE:
            return self._process_frame_fallback(image)

        # --- Real CUDA SAM-3 Processing ---
        # Wrap everything in inference_mode for VRAM safety
        with torch.inference_mode():
            try:
                H, W, C = image.shape
                
                # Resizing frame to 1024x1024 (native SAM res) to avoid memory spikes
                if H > 1024 or W > 1024:
                    image_resized = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                    logger.debug(f"Resized frame from {W}x{H} to 1024x1024 for VRAM efficiency.")
                else:
                    image_resized = image

                # Convert BGR to RGB (SAM expectation)
                image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
                
                # Add frame to SAM-3's active video tracking buffer
                self.predictor.add_new_frame(
                    inference_state=self.inference_state,
                    frame_idx=self.frame_idx,
                    frame_image=image_rgb
                )

                # For the first frame, we segment the concepts using text prompt ("apple")
                if self.frame_idx == 0:
                    # In SAM-3, we prompt using text or concepts
                    # Returns a list of segments/ids
                    logger.info("Prompting SAM-3 with text concept: 'apple'")
                    self.predictor.add_new_prompt(
                        inference_state=self.inference_state,
                        frame_idx=0,
                        obj_id=None, # Automatically allocate object IDs
                        points=None,
                        labels=None,
                        box=None,
                        text_prompt="apple"
                    )

                # Propagate masks to the current frame
                # propagate_in_video returns active track IDs and their segmentation masks
                out_obj_ids, out_mask_logits = self.predictor.propagate_in_video(
                    inference_state=self.inference_state,
                    start_frame_idx=self.frame_idx,
                    max_frame_num_to_track=10 # Keep attention span restricted for VRAM safety
                )

                results = []
                for obj_id, mask_logit in zip(out_obj_ids, out_mask_logits):
                    # Convert logit to binary mask
                    mask = (mask_logit > 0.0).cpu().numpy().astype(np.uint8)[0] # shape: [1024, 1024]
                    
                    # Resize mask back to original resolution
                    if H != 1024 or W != 1024:
                        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

                    # Calculate bounding box from mask
                    ys, xs = np.where(mask > 0)
                    if len(xs) > 0 and len(ys) > 0:
                        box = [int(np.min(ys)), int(np.min(xs)), int(np.max(ys)), int(np.max(xs))]
                        results.append({
                            "id": int(obj_id),
                            "box": box,
                            "mask": mask,
                            "score": 0.95 # Base confidence placeholder
                        })

                self.frame_idx += 1
                return results

            except Exception as e:
                log_exception(logger, f"Error processing frame {self.frame_idx} in CUDA SAM-3", e)
                # Recover by freeing cache
                torch.cuda.empty_cache()
                return []

    def _process_frame_fallback(self, image):
        """
        CPU Mock Mode: Segment apples by detecting red/green circular shapes in HSV color space
        and associate them across frames using a Centroid Tracker.
        """
        H, W, C = image.shape
        
        # 1. Image preprocessing: Convert BGR to HSV for robust color thresholding
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Apple red color ranges (two ranges in HSV because red wraps around)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2
        
        # Optional: Add Apple green range for green apples (HSV: 35-85)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        combined_mask = red_mask | green_mask
        
        # Morphological operations to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # 2. Extract detected contours as apples
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter contours that are too small or too large to represent apples
            if 150 < area < 20000:
                # Compute bounding box
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                centroid = np.array([x + w_box/2.0, y + h_box/2.0])
                
                # Create a binary mask specifically for this instance
                inst_mask = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(inst_mask, [cnt], -1, 1, thickness=-1)
                
                detections.append({
                    "centroid": centroid,
                    "box": [y, x, y + h_box, x + w_box],
                    "mask": inst_mask,
                    "score": round(min(0.5 + area/50000.0, 0.99), 2)
                })

        # 3. Associate detections with existing tracks (Centroid Tracker)
        updated_tracks = {}
        unused_detections = list(range(len(detections)))
        
        if len(self.active_tracks) > 0 and len(detections) > 0:
            track_ids = list(self.active_tracks.keys())
            track_centroids = np.array([self.active_tracks[tid]["centroid"] for tid in track_ids])
            det_centroids = np.array([det["centroid"] for det in detections])
            
            # Compute distance matrix between tracking history and new detections
            dists = np.linalg.norm(track_centroids[:, np.newaxis] - det_centroids, axis=2)
            
            # Match greedily
            rows = dists.min(axis=1).argsort()
            cols = dists.argmin(axis=1)
            
            used_rows = set()
            used_cols = set()
            
            for r in rows:
                if r in used_rows:
                    continue
                c = cols[r]
                if c in used_cols:
                    continue
                    
                # If distance is within a reasonable pixel radius (e.g. 100px for camera movement)
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
                        
            # Keep track of missed active tracks (occluded)
            for r in range(len(track_ids)):
                if r not in used_rows:
                    tid = track_ids[r]
                    tdata = self.active_tracks[tid]
                    tdata["disappeared_count"] += 1
                    # Keep active until occlusion limit is hit
                    if tdata["disappeared_count"] <= self.disappeared_limit:
                        updated_tracks[tid] = tdata

        elif len(self.active_tracks) > 0:
            # No detections found, increment disappeared count for all active tracks
            for tid, tdata in self.active_tracks.items():
                tdata["disappeared_count"] += 1
                if tdata["disappeared_count"] <= self.disappeared_limit:
                    updated_tracks[tid] = tdata
                    
        # Register new tracks for unused detections
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
        
        # Format output list according to base tracker standard
        results = []
        for tid, tdata in self.active_tracks.items():
            # Only report if currently visible
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
        logger.info("Clearing SAM-3 internal memory bank.")
        self.frame_idx = 0
        self.active_tracks = {}
        self.next_track_id = 1
        
        if not self.use_mock and SAM3_AVAILABLE and self.predictor is not None:
            try:
                # Evict SAM-3 video streaming inference state
                self.predictor.reset_state(self.inference_state)
                self.inference_state = self.predictor.init_state(video_path=None)
                # Force PyTorch allocator cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("CUDA VRAM memory cache successfully flushed.")
            except Exception as e:
                logger.warning(f"Error during physical predictor reset: {e}")
