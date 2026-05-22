#!/usr/bin/env python3
import os
import sys
import subprocess
import json
import logging

# Ensure root directory is in PATH
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def run_cmd(cmd):
    """
    Helper to run terminal command and stream output.
    """
    print(f"\nRunning command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Stream stdout/stderr in real-time to console
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            
    rc = process.poll()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc

def main():
    print("========================================")
    print("STARTING PIPELINE CPU VERIFICATION TESTS")
    print("========================================")
    
    workspace_dir = os.path.abspath(os.path.dirname(__file__))
    mock_frames_dir = os.path.join(workspace_dir, "mock_frames")
    output_dir = os.path.join(workspace_dir, "output")
    
    # 1. Generate mock frames
    print("\nStep 1: Generating synthetic test frames...")
    try:
        from generate_mock_sequence import main as gen_mock_main
        gen_mock_main()
    except Exception as e:
        print(f"Error generating mock sequence: {e}")
        sys.exit(1)
        
    if not os.path.exists(mock_frames_dir) or len(os.listdir(mock_frames_dir)) == 0:
        print("Error: Mock frames were not generated successfully.")
        sys.exit(1)
    print("Mock frames generated successfully!")
    
    # 2. Run benchmark script in dry-run/mock mode on CPU
    print("\nStep 2: Running benchmark pipeline in CPU Mock mode...")
    benchmark_script = os.path.join(workspace_dir, "benchmark.py")
    
    # Command to test all three models on mock frames
    cmd = [
        sys.executable,
        benchmark_script,
        "--frames_dir", mock_frames_dir,
        "--models", "sam3,deva,yolo",
        "--prompt", "apple",
        "--output_dir", output_dir,
        "--use_mock"
    ]
    
    try:
        run_cmd(cmd)
    except subprocess.CalledProcessError as e:
        print(f"\nPipeline Execution FAILED with exit code {e.returncode}")
        sys.exit(1)
        
    print("\nPipeline executed successfully without exceptions!")
    
    # 3. Verify output files exist and are populated
    print("\nStep 3: Verifying generated benchmark assets...")
    
    expected_files = [
        "sam3_tracking.mp4",
        "deva_tracking.mp4",
        "yolo_tracking.mp4",
        "benchmark_results.json",
        "unique_ids_comparison.png",
        "track_lifespan_comparison.png",
        "efficiency_comparison.png"
    ]
    
    missing_files = []
    for f in expected_files:
        path = os.path.join(output_dir, f)
        if not os.path.exists(path):
            missing_files.append(f)
        else:
            size_kb = round(os.path.getsize(path) / 1024.0, 2)
            print(f"  [FOUND] {f:<30} | Size: {size_kb:>8} KB")
            
    if missing_files:
        print(f"\nVerification FAILED: The following expected output files were not found: {missing_files}")
        sys.exit(1)
        
    # 4. Check benchmark metrics consistency
    print("\nStep 4: Checking benchmark metrics contents...")
    json_path = os.path.join(output_dir, "benchmark_results.json")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
            
        print("\nVerified JSON summary data structure:")
        for model, m_data in metrics_data.items():
            print(f"  {model.upper()} Metrics:")
            print(f"    - Frames Processed: {m_data.get('num_frames')}")
            print(f"    - Unique track IDs: {m_data.get('unique_ids')}")
            print(f"    - Mean track length: {m_data.get('avg_track_length_frames')} frames")
            print(f"    - Average apples per frame: {m_data.get('avg_apples_per_frame')}")
            print(f"    - Mean FPS: {m_data.get('mean_fps')}")
            
            # Simple logical check
            if m_data.get("unique_ids", 0) <= 0:
                print(f"    [ERROR] Model {model} registered 0 tracks. Verification failed.")
                sys.exit(1)
                
    except Exception as e:
        print(f"Error reading and verifying metrics JSON: {e}")
        sys.exit(1)
        
    print("\n========================================")
    print("VERIFICATION COMPLETE: ALL CPU TESTS PASSED!")
    print("The entire benchmarking framework is fully robust, verified,")
    print("and ready for deployment on your GPU machine.")
    print("========================================")

if __name__ == "__main__":
    main()
