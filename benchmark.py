#!/usr/bin/env python3
import os
import sys
import time
import argparse
import json
import torch
import cv2
import logging

from utils import setup_logger, get_vram_usage, print_gpu_diagnostics, log_exception
from metrics import compute_unsupervised_metrics
from visualizer import generate_video, plot_benchmark_charts

# Initialize wrappers
from models.sam3_wrapper import SAM3Wrapper
from models.deva_wrapper import DEVAWrapper
from models.yolo_tracker_wrapper import YOLOTrackerWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Apple Tracking & Persistent Memory Benchmarking Pipeline")
    parser.add_argument(
        "--frames_dir", 
        type=str, 
        required=True, 
        help="Path to folder containing sequential video frames"
    )
    parser.add_argument(
        "--models", 
        type=str, 
        default="sam3,deva,yolo", 
        help="Comma-separated list of models to evaluate (sam3, deva, yolo)"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="apple", 
        help="Text prompt for open-vocabulary detection & VOS"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./output", 
        help="Directory to save evaluation charts and annotated videos"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto", 
        help="Device to run inference on (cuda, mps, cpu, or auto)"
    )
    parser.add_argument(
        "--use_mock", 
        action="store_true", 
        help="Force CPU mock mode (Centroid Tracker fallback) for dry-runs and pipeline testing"
    )
    parser.add_argument(
        "--max_frames", 
        type=int, 
        default=-1, 
        help="Maximum number of sequential frames to process (-1 for all)"
    )
    parser.add_argument(
        "--log_file", 
        type=str, 
        default="benchmark.log", 
        help="Path to save detailed execution and debug logs"
    )
    return parser.parse_args()

def detect_device(device_arg):
    if device_arg != "auto":
        return device_arg
        
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    args = parse_args()
    
    # 1. Initialize Robust Logging
    logger = setup_logger(args.log_file, verbose=True)
    logger.info("=== Starting Apple Tracking Benchmark Framework ===")
    
    # Print hardware configurations
    device = detect_device(args.device)
    logger.info(f"Target Execution Device: {device}")
    print_gpu_diagnostics(logger)
    
    # 2. Frame Directory Validation
    if not os.path.exists(args.frames_dir):
        logger.error(f"Frames directory not found: {args.frames_dir}")
        sys.exit(1)
        
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    frame_files = sorted([
        os.path.join(args.frames_dir, f) for f in os.listdir(args.frames_dir)
        if f.lower().endswith(valid_exts)
    ])
    
    num_frames = len(frame_files)
    if args.max_frames > 0:
        frame_files = frame_files[:args.max_frames]
        num_frames = len(frame_files)
        logger.info(f"Capping frame execution to the first {num_frames} frames as requested.")
        
    if num_frames == 0:
        logger.error(f"No valid frames (.png, .jpg, etc.) found in directory: {args.frames_dir}")
        sys.exit(1)
        
    logger.info(f"Found {num_frames} frames in directory: {args.frames_dir}")
    
    # Inspect frame resolution
    sample_frame = cv2.imread(frame_files[0])
    H, W, C = sample_frame.shape
    logger.info(f"Source Frame Resolution: {W}x{H} (Channels: {C})")
    
    # 3. Model Matching
    selected_models = [m.strip().lower() for m in args.models.split(",")]
    logger.info(f"Selected evaluation models: {selected_models}")
    
    # Dictionary to store wrappers
    model_classes = {
        "sam3": SAM3Wrapper,
        "deva": DEVAWrapper,
        "yolo": YOLOTrackerWrapper
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    all_metrics = {}
    
    # 4. Evaluate each model
    for model_name in selected_models:
        if model_name not in model_classes:
            logger.warning(f"Unknown model '{model_name}'. Skipping.")
            continue
            
        logger.info(f"\n========================================\nEvaluating Model: {model_name.upper()}\n========================================")
        
        # Instantiate Wrapper
        try:
            wrapper_cls = model_classes[model_name]
            tracker = wrapper_cls(device=device, use_mock=args.use_mock)
            
            # Initialize Model
            logger.info("Initializing tracker model and applying VRAM optimization configurations...")
            tracker.initialize(prompt=args.prompt)
            
            tracking_history = []
            fps_list = []
            peak_vram = 0.0
            
            # Iterate through frames
            logger.info(f"Processing {num_frames} frames sequentially...")
            
            for idx, frame_path in enumerate(frame_files):
                # Read image
                img = cv2.imread(frame_path)
                if img is None:
                    logger.warning(f"Frame {idx} failed to load from: {frame_path}. Skipping.")
                    continue
                
                # Perform tracking and measure latency
                t_start = time.perf_counter()
                detections = tracker.process_frame(img)
                t_end = time.perf_counter()
                
                latency = t_end - t_start
                fps = 1.0 / latency if latency > 0 else 100.0
                fps_list.append(fps)
                
                # Query VRAM metrics
                vram_usage = get_vram_usage()
                vram_allocated = vram_usage["allocated_mb"]
                if vram_allocated > peak_vram:
                    peak_vram = vram_allocated
                    
                tracking_history.append(detections)
                
                # Log micro-progress periodically (every 10 frames)
                if (idx + 1) % 10 == 0 or idx == num_frames - 1:
                    logger.info(
                        f"  [Frame {idx+1}/{num_frames}] - Detections: {len(detections)} - "
                        f"FPS: {round(fps, 1)} - VRAM: {vram_allocated} MB (Peak: {round(peak_vram, 1)} MB)"
                    )
            
            # Compute tracking stability metrics
            logger.info(f"Computing tracking consistency metrics for {model_name.upper()}...")
            metrics = compute_unsupervised_metrics(tracking_history, fps_list, peak_vram)
            all_metrics[model_name] = metrics
            
            # Print Model Metrics
            logger.info(f"--- Results for {model_name.upper()} ---")
            for k, v in metrics.items():
                logger.info(f"  {k}: {v}")
            
            # Compile annotated video
            video_out_path = os.path.join(args.output_dir, f"{model_name}_tracking.mp4")
            generate_video(
                frames_dir=args.frames_dir,
                tracking_history=tracking_history,
                output_path=video_out_path,
                fps=15,
                draw_masks=(model_name != "yolo"), # Draw high-resolution masks for SAM-3/DEVA
                draw_boxes=(model_name == "yolo"), # Draw bounding boxes only if we are using YOLO (since it has no masks)
                draw_ids=False # Remove persistent track ID label overlay as requested by the user
            )
            logger.info(f"Tracking video saved to: {video_out_path}")
            
            # Clear VRAM memory of model to prevent leak on next model run
            tracker.clear_memory()
            del tracker
            
        except Exception as e:
            log_exception(logger, f"Execution failed for model {model_name}", e)
            continue
            
    # 5. Summarize Results & Plot charts
    if len(all_metrics) > 0:
        logger.info("\n========================================\nFINAL BENCHMARK COMPARISON SUMMARY\n========================================")
        
        # Save metrics as a JSON file
        json_path = os.path.join(args.output_dir, "benchmark_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=4)
        logger.info(f"Summary metrics saved in JSON format to: {os.path.abspath(json_path)}")
        
        # Print tabular results to log & console
        header = f"{'Model':<10} | {'Frames':<6} | {'Unique IDs':<10} | {'Avg Lifespan %':<14} | {'FPS':<6} | {'Peak VRAM (MB)':<14}"
        logger.info(header)
        logger.info("-" * len(header))
        for model, met in all_metrics.items():
            logger.info(
                f"{model.upper():<10} | {met['num_frames']:<6} | {met['unique_ids']:<10} | "
                f"{met['avg_track_length_pct']:<14} | {met['mean_fps']:<6} | {met['peak_vram_mb']:<14}"
            )
            
        # Plot and save PNG bar charts
        plot_benchmark_charts(all_metrics, args.output_dir)
        logger.info("=== Benchmarking successfully complete. Diagnostics saved in output dir. ===")
    else:
        logger.error("No model evaluation completed successfully. Please check logs for details.")

if __name__ == "__main__":
    main()
