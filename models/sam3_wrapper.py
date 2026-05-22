import os
import sys
import cv2
import numpy as np
import torch
import logging
import contextlib

from models.base_tracker import BaseTracker
from utils import get_vram_usage, log_exception

logger = logging.getLogger("AppleTrackerBenchmark.sam3")

# Optional import of official SAM-3 library from Hugging Face transformers
SAM3_AVAILABLE = False
try:
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    SAM3_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Official 'sam3' library not found. Reason: {e}. Wrapper will fall back to CPU Centroid Tracker for dry-runs.")

class SAM3Wrapper(BaseTracker):
    """
    Wrapper for Meta's SAM-3 (Segment Anything Model 3) via Hugging Face.
    Integrates promptable concept segmentation with optimized VRAM usage.
    """
    def __init__(self, device="cpu", use_mock=False, checkpoint_path="facebook/sam3"):
        super().__init__(device, use_mock)
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.processor = None
        self.precomputed_results = []
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

        try:
            # Apply PyTorch VRAM configuration for fragmentation prevention
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            logger.info("Applied PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'")

            # Load SAM-3 Hugging Face model and processor
            logger.info(f"Loading SAM-3 model from Hugging Face: {self.checkpoint_path}")
            try:
                self.model = Sam3VideoModel.from_pretrained(
                    self.checkpoint_path,
                    device_map=self.device,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            except Exception as e:
                # If accelerate is missing, load without device_map and move to device manually
                if "accelerate" in str(e) or "device_map" in str(e):
                    logger.info("The 'accelerate' package is missing. Bypassing device_map and transferring model to GPU manually...")
                    self.model = Sam3VideoModel.from_pretrained(
                        self.checkpoint_path,
                        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                    )
                    self.model.to(self.device)
                else:
                    raise e
            self.processor = Sam3VideoProcessor.from_pretrained(self.checkpoint_path)
            
            # Tweak for small objects like apples
            self.model.config.score_threshold_detection = 0.3  # lower for more detections
            self.model.config.new_det_thresh = 0.6
            
            self.frame_idx = 0
            self.precomputed_results = []
            
            # Look up frames in sys.argv to run the pre-computation
            frames_dir = None
            max_frames = -1
            for i, arg in enumerate(sys.argv):
                if arg == "--frames_dir" and i + 1 < len(sys.argv):
                    frames_dir = sys.argv[i+1]
                elif arg == "--max_frames" and i + 1 < len(sys.argv):
                    max_frames = int(sys.argv[i+1])
            
            if frames_dir and os.path.exists(frames_dir):
                logger.info(f"Pre-loading frames from: {frames_dir} for SAM-3 video tracking...")
                valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
                frame_files = sorted([
                    os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
                    if f.lower().endswith(valid_exts)
                ])
                if max_frames > 0:
                    frame_files = frame_files[:max_frames]
                
                # Load all frames
                video_frames = []
                orig_shapes = []
                for fpath in frame_files:
                    frame = cv2.imread(fpath)
                    if frame is not None:
                        h_orig, w_orig = frame.shape[:2]
                        orig_shapes.append((h_orig, w_orig))
                        
                        # Downsample to a maximum dimension of 640 to prevent CUDA OOM
                        # while keeping aspect ratio intact
                        max_dim = 640
                        if max(h_orig, w_orig) > max_dim:
                            scale = max_dim / float(max(h_orig, w_orig))
                            new_w = int(w_orig * scale)
                            new_h = int(h_orig * scale)
                            frame_resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        else:
                            frame_resized = frame
                            
                        # Convert to RGB (Hugging Face expects RGB)
                        video_frames.append(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                
                if len(video_frames) > 0:
                    logger.info(f"Loaded {len(video_frames)} frames. Initializing SAM-3 video session...")
                    inference_session = self.processor.init_video_session(
                        video=video_frames,
                        inference_device=self.device,
                        processing_device="cpu",
                        video_storage_device="cpu",
                    )
                    
                    logger.info(f"Adding text prompt: '{prompt}'")
                    inference_session = self.processor.add_text_prompt(
                        inference_session=inference_session,
                        text=prompt,
                    )
                    
                    logger.info("Propagating and tracking across video frames...")
                    iterator = self.model.propagate_in_video_iterator(
                        inference_session=inference_session,
                        max_frame_num_to_track=len(video_frames)-1,
                    )
                    
                    autocast_context = (
                        torch.autocast(device_type="cuda", dtype=torch.float16) 
                        if self.device == "cuda" 
                        else contextlib.nullcontext()
                    )
                    
                    outputs_per_frame = {}
                    with autocast_context:
                        for model_outputs in iterator:
                            processed = self.processor.postprocess_outputs(inference_session, model_outputs)
                            outputs_per_frame[model_outputs.frame_idx] = processed
                    
                    # Convert to standard format
                    self.precomputed_results = [[] for _ in range(len(video_frames))]
                    for f_idx in sorted(outputs_per_frame.keys()):
                        processed = outputs_per_frame[f_idx]
                        boxes = processed['boxes'].cpu().numpy()
                        masks = processed['masks'].cpu().numpy()
                        ids = processed['object_ids'].cpu().numpy()
                        scores = processed['scores'].cpu().numpy()
                        
                        h_orig, w_orig = orig_shapes[f_idx]
                        h_resized, w_resized = video_frames[f_idx].shape[:2]
                        
                        scale_x = w_orig / float(w_resized)
                        scale_y = h_orig / float(h_resized)
                        
                        frame_results = []
                        for i in range(len(ids)):
                            obj_id = int(ids[i])
                            box = boxes[i] # [x1, y1, x2, y2]
                            mask = masks[i]
                            score = float(scores[i])
                            
                            # Rescale box back to original coordinates
                            x1 = int(box[0] * scale_x)
                            y1 = int(box[1] * scale_y)
                            x2 = int(box[2] * scale_x)
                            y2 = int(box[3] * scale_y)
                            
                            # Convert box to [ymin, xmin, ymax, xmax]
                            ymin, xmin, ymax, xmax = y1, x1, y2, x2
                            
                            # Resize mask back to original resolution if needed
                            mask_bool = mask > 0.5
                            mask_bool = cv2.resize(
                                mask_bool.astype(np.uint8), 
                                (w_orig, h_orig), 
                                interpolation=cv2.INTER_NEAREST
                            ).astype(bool)
                            
                            frame_results.append({
                                "id": obj_id,
                                "box": [ymin, xmin, ymax, xmax],
                                "mask": mask_bool.astype(np.uint8),
                                "score": score
                            })
                        self.precomputed_results[f_idx] = frame_results
                    
                    logger.info("SAM-3 pre-computation complete!")
                else:
                    logger.error("No valid frames could be pre-loaded.")
            else:
                logger.warning("Could not pre-load frames directory from sys.argv. Falling back to empty precomputed results.")
            
            mem = get_vram_usage()
            logger.info(f"SAM-3 successfully initialized. VRAM Allocated: {mem['allocated_mb']} MB, Peak VRAM: {mem['peak_mb']} MB")
            
        except Exception as e:
            log_exception(logger, "Failed to initialize HF SAM-3. Falling back to CPU Mock mode", e)
            self.use_mock = True
            self.clear_memory()

    def process_frame(self, image):
        """
        Processes a frame using SAM-3's precomputed concept propagation or CPU centroid tracking fallback.
        """
        if self.use_mock or not SAM3_AVAILABLE:
            return self._process_frame_fallback(image)

        # Retrieve precomputed frame results
        if self.precomputed_results and self.frame_idx < len(self.precomputed_results):
            results = self.precomputed_results[self.frame_idx]
            self.frame_idx += 1
            return results
        
        self.frame_idx += 1
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
        self.precomputed_results = []
        self.model = None
        self.processor = None
        
        if not self.use_mock and SAM3_AVAILABLE:
            try:
                # Force PyTorch allocator cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("CUDA VRAM memory cache successfully flushed.")
            except Exception as e:
                logger.warning(f"Error during VRAM cache flush: {e}")
