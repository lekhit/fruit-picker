import numpy as np
import logging

logger = logging.getLogger("AppleTrackerBenchmark.metrics")

def compute_unsupervised_metrics(tracking_history, fps_list, peak_vram):
    """
    Computes advanced unsupervised tracking stability and performance metrics.
    Useful when ground-truth annotations are not available.
    
    Parameters:
        tracking_history: List of frames, where each frame is a list of tracking dicts:
                          [ {"id": int, "box": [...], "score": float}, ... ]
        fps_list: List of float values representing processing speed (FPS) for each frame.
        peak_vram: Peak VRAM usage recorded during the run.
    """
    total_frames = len(tracking_history)
    if total_frames == 0:
        return {
            "num_frames": 0,
            "unique_ids": 0,
            "avg_track_length": 0.0,
            "avg_apples_per_frame": 0.0,
            "fragmentation_rate": 0.0,
            "mean_fps": 0.0,
            "peak_vram_mb": 0.0
        }

    # Aggregate tracks
    # track_lifespans maps ID -> count of frames it appeared in
    track_lifespans = {}
    total_detections = 0
    apples_per_frame = []
    
    for frame_idx, detections in enumerate(tracking_history):
        apples_per_frame.append(len(detections))
        for det in detections:
            tid = det["id"]
            track_lifespans[tid] = track_lifespans.get(tid, 0) + 1
            total_detections += 1

    unique_ids = len(track_lifespans)
    avg_track_length = np.mean(list(track_lifespans.values())) if unique_ids > 0 else 0.0
    avg_apples = np.mean(apples_per_frame)
    
    # Fragmentation rate: unique IDs relative to total detections
    # Lower is better (ideal: 1 ID per actual apple, staying active)
    frag_rate = (unique_ids / total_detections) if total_detections > 0 else 0.0
    mean_fps = np.mean(fps_list) if len(fps_list) > 0 else 0.0

    return {
        "num_frames": total_frames,
        "unique_ids": unique_ids,
        "avg_track_length_frames": round(avg_track_length, 2),
        "avg_track_length_pct": round((avg_track_length / total_frames) * 100.0, 2) if total_frames > 0 else 0.0,
        "avg_apples_per_frame": round(avg_apples, 2),
        "fragmentation_rate": round(frag_rate, 4),
        "mean_fps": round(mean_fps, 2),
        "peak_vram_mb": round(peak_vram, 2)
    }

def compute_supervised_metrics(tracking_history, ground_truth):
    """
    Computes standard MOT tracking metrics (IDF1, ID Switches, IDF1 approximation)
    if ground truth annotations are provided.
    
    ground_truth format: dict mapping frame_idx -> list of dicts:
                         { 0: [{"id": int, "box": [ymin, xmin, ymax, xmax]}, ...] }
    """
    logger.info("Computing supervised MOT metrics against ground-truth...")
    
    id_switches = 0
    total_gt_boxes = 0
    total_pred_boxes = 0
    correct_associations = 0
    
    # Track the active mapping from Ground Truth ID -> Predicted ID
    gt_to_pred_map = {}
    
    for frame_idx in range(len(tracking_history)):
        preds = tracking_history[frame_idx]
        gts = ground_truth.get(frame_idx, [])
        
        total_pred_boxes += len(preds)
        total_gt_boxes += len(gts)
        
        if len(gts) == 0 or len(preds) == 0:
            continue
            
        # Match boxes based on simple IoU overlap
        matched_gt = set()
        matched_pred = set()
        
        # We calculate IoU overlap for all pairs
        for g_idx, gt in enumerate(gts):
            gt_box = gt["box"]
            best_iou = 0.0
            best_p_idx = -1
            
            for p_idx, pred in enumerate(preds):
                if p_idx in matched_pred:
                    continue
                pred_box = pred["box"]
                
                # Calculate IoU
                ymin = max(gt_box[0], pred_box[0])
                xmin = max(gt_box[1], pred_box[1])
                ymax = min(gt_box[2], pred_box[2])
                xmax = min(gt_box[3], pred_box[3])
                
                inter = max(0, ymax - ymin) * max(0, xmax - xmin)
                area_gt = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                area_pred = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                union = area_gt + area_pred - inter
                
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_p_idx = p_idx
            
            # If overlap is > 0.3 (apples are small, 0.3-0.5 is standard IoU threshold for fruit detection)
            if best_iou >= 0.3:
                gt_id = gt["id"]
                pred_id = preds[best_p_idx]["id"]
                matched_gt.add(g_idx)
                matched_pred.add(best_p_idx)
                
                # Check for ID Switches
                if gt_id in gt_to_pred_map:
                    if gt_to_pred_map[gt_id] != pred_id:
                        id_switches += 1
                        gt_to_pred_map[gt_id] = pred_id # Update mapping
                else:
                    gt_to_pred_map[gt_id] = pred_id
                    
                correct_associations += 1
                
    precision = correct_associations / total_pred_boxes if total_pred_boxes > 0 else 0.0
    recall = correct_associations / total_gt_boxes if total_gt_boxes > 0 else 0.0
    
    # IDF1 score combines identification precision and recall
    idf1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "idf1": round(idf1, 4),
        "id_switches": id_switches,
        "total_gt_objects": total_gt_boxes,
        "total_pred_objects": total_pred_boxes
    }
