import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger("AppleTrackerBenchmark.visualizer")

# Generate a deterministic set of distinct colors for drawing tracks
np.random.seed(42)
TRACK_COLORS = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)

def draw_overlays(frame, detections, draw_masks=True):
    """
    Draws semi-transparent segmentation masks, bounding boxes, and persistent ID labels
    on a video frame. Ensures each track ID gets a consistent color.
    """
    annotated = frame.copy()
    H, W, _ = frame.shape
    
    for det in detections:
        tid = det["id"]
        box = det["box"] # [ymin, xmin, ymax, xmax]
        mask = det.get("mask", None)
        score = det.get("score", 0.0)
        
        # Select color deterministically based on track ID
        color = TRACK_COLORS[tid % len(TRACK_COLORS)]
        color_bgr = (int(color[0]), int(color[1]), int(color[2]))
        
        # 1. Draw transparent segmentation mask if available and requested
        if draw_masks and mask is not None:
            # Ensure mask matches frame dimensions
            if mask.shape[:2] != (H, W):
                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            
            # Create transparent overlay colored mask
            colored_mask = np.zeros_like(frame)
            colored_mask[mask > 0] = color_bgr
            
            # Blend frame and colored mask
            mask_indices = mask > 0
            annotated[mask_indices] = cv2.addWeighted(
                annotated, 0.6, colored_mask, 0.4, 0
            )[mask_indices]
            
            # Draw fine border around mask
            contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, color_bgr, 1)

        # 2. Draw bounding box
        ymin, xmin, ymax, xmax = box
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color_bgr, 2)
        
        # 3. Draw premium ID label tag
        label = f"ID: {tid} ({int(score * 100)}%)"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        thickness = 1
        
        # Compute text size for tag background box
        (w, h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Tag placement slightly above bounding box
        tag_ymin = max(ymin - h - 6, 0)
        tag_ymax = min(ymin, H)
        tag_xmin = xmin
        tag_xmax = min(xmin + w + 10, W)
        
        # Draw solid tag background box
        cv2.rectangle(annotated, (tag_xmin, tag_ymin), (tag_xmax, tag_ymax), color_bgr, cv2.FILLED)
        # Draw text inside tag
        cv2.putText(annotated, label, (tag_xmin + 5, tag_ymin + h + 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
    return annotated

def generate_video(frames_dir, tracking_history, output_path, fps=15, draw_masks=True):
    """
    Renders the tracking history onto the source frames and compiles them into an MP4 video.
    """
    logger.info(f"Rendering tracking overlays. Compiling video to: {output_path}")
    
    # List and sort frame files in frames_dir
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    frame_files = sorted([
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if f.lower().endswith(valid_exts)
    ])
    
    if len(frame_files) == 0:
        logger.error(f"No valid frames found in: {frames_dir}. Cannot generate video.")
        return
        
    if len(frame_files) != len(tracking_history):
        logger.warning(f"Mismatch between frames found ({len(frame_files)}) and tracking history ({len(tracking_history)}). Output will be truncated to shortest.")
        limit = min(len(frame_files), len(tracking_history))
        frame_files = frame_files[:limit]
        tracking_history = tracking_history[:limit]

    # Read first frame to initialize VideoWriter dimensions
    first_frame = cv2.imread(frame_files[0])
    H, W, _ = first_frame.shape
    
    # Initialize OpenCV VideoWriter (MP4V is widely compatible across platforms)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    try:
        for idx, (frame_file, detections) in enumerate(zip(frame_files, tracking_history)):
            frame = cv2.imread(frame_file)
            if frame is None:
                logger.warning(f"Failed to read frame: {frame_file}. Skipping.")
                continue
                
            annotated = draw_overlays(frame, detections, draw_masks=draw_masks)
            writer.write(annotated)
            
        logger.info("Video rendering successfully completed.")
    finally:
        writer.release()

def plot_benchmark_charts(metrics_dict, output_dir):
    """
    Generates beautiful comparison charts of the metrics achieved by different models
    and saves them as PNG assets.
    
    metrics_dict format:
    {
        "SAM-3": {"unique_ids": 24, "avg_track_length_frames": 45, "mean_fps": 18.5},
        "DEVA": {"unique_ids": 32, "avg_track_length_frames": 38, "mean_fps": 12.2},
        ...
    }
    """
    logger.info("Plotting comparative benchmark charts...")
    os.makedirs(output_dir, exist_ok=True)
    
    models = list(metrics_dict.keys())
    if len(models) == 0:
        logger.warning("No model metrics available to plot.")
        return
        
    # Standard modern color palette (premium HSL look)
    colors = ["#4D96FF", "#6BCB77", "#FF6B6B", "#FFD93D"]
    
    # 1. Chart 1: Unique Track IDs (Lower is better for consistency)
    plt.figure(figsize=(8, 5))
    unique_ids = [metrics_dict[m]["unique_ids"] for m in models]
    bars = plt.bar(models, unique_ids, color=colors[:len(models)], width=0.5)
    plt.title("Total Unique Track IDs (Lower is Better - Less ID Switches)", fontsize=12, fontweight='bold', pad=15)
    plt.ylabel("Number of Unique IDs", fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, str(yval), ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "unique_ids_comparison.png")
    plt.savefig(chart1_path, dpi=200)
    plt.close()
    
    # 2. Chart 2: Average Track Length Percentage (Higher is better)
    plt.figure(figsize=(8, 5))
    lengths = [metrics_dict[m].get("avg_track_length_pct", 0.0) for m in models]
    bars = plt.bar(models, lengths, color=colors[:len(models)], width=0.5)
    plt.title("Mean Track Lifespan (% of Video Length - Higher is Better)", fontsize=12, fontweight='bold', pad=15)
    plt.ylabel("Average Lifespan (%)", fontsize=10)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval}%", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "track_lifespan_comparison.png")
    plt.savefig(chart2_path, dpi=200)
    plt.close()
    
    # 3. Chart 3: Speed (FPS) vs Peak VRAM (GPU Memory)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    fps = [metrics_dict[m]["mean_fps"] for m in models]
    vram = [metrics_dict[m]["peak_vram_mb"] for m in models]
    
    x = np.arange(len(models))
    width = 0.3
    
    # Primary axis - FPS
    rects1 = ax1.bar(x - width/2, fps, width, label='Speed (FPS)', color='#6BCB77')
    ax1.set_ylabel('Processing Speed (FPS)', color='#6BCB77', fontsize=10, fontweight='bold')
    ax1.set_xlabel('Models', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#6BCB77')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    
    # Secondary axis - VRAM
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, vram, width, label='VRAM (MB)', color='#FF6B6B')
    ax2.set_ylabel('Peak VRAM Allocation (MB)', color='#FF6B6B', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#FF6B6B')
    
    plt.title("Performance Efficiency: Speed (FPS) vs VRAM (MB)", fontsize=12, fontweight='bold', pad=15)
    fig.tight_layout()
    chart3_path = os.path.join(output_dir, "efficiency_comparison.png")
    plt.savefig(chart3_path, dpi=200)
    plt.close()
    
    logger.info(f"Charts saved to: {os.path.abspath(output_dir)}")
