#!/usr/bin/env python3
import os
import cv2
import numpy as np

def main():
    output_dir = "./mock_frames"
    os.makedirs(output_dir, exist_ok=True)
    
    num_frames = 60
    width, height = 640, 480
    
    print(f"Generating {num_frames} synthetic apple-tracking frames in: {output_dir}")
    
    # Define apples (circles) drifting horizontally
    # Format: {"start_pos": [x, y], "velocity": [vx, vy], "radius": r, "color": (B, G, R)}
    apples = [
        {"pos": [50.0, 150.0], "vel": [8.0, 0.0], "radius": 22, "color": (30, 30, 220)},     # Red Apple 1
        {"pos": [120.0, 300.0], "vel": [7.0, 0.5], "radius": 26, "color": (20, 20, 240)},    # Red Apple 2 (larger)
        {"pos": [30.0, 380.0], "vel": [9.0, -0.5], "radius": 18, "color": (40, 200, 40)},    # Green Apple 1
        {"pos": [450.0, 240.0], "vel": [5.0, 0.2], "radius": 20, "color": (15, 15, 210)},    # Red Apple 3
    ]
    
    # Tree trunk column (occlusion region)
    trunk_x_start = 280
    trunk_x_end = 360
    
    for f in range(num_frames):
        # 1. Base forest green background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (15, 35, 10)  # Dark green/brown base background
        
        # Draw some background branches (fine lines)
        cv2.line(frame, (50, 400), (300, 200), (30, 50, 40), 6)
        cv2.line(frame, (580, 420), (320, 180), (35, 45, 35), 8)
        cv2.line(frame, (200, 100), (450, 300), (25, 40, 30), 4)

        # 2. Draw apples FIRST (so trunk can occlude them by being drawn on top)
        for apple in apples:
            x, y = apple["pos"]
            r = apple["radius"]
            color = apple["color"]
            
            # Draw apple body
            cv2.circle(frame, (int(x), int(y)), r, color, -1)
            # Add small leaf (green ellipse) on red apples
            if color[2] > 200: # Red apple
                cv2.ellipse(frame, (int(x - 4), int(y - r - 2)), (8, 4), -30, 0, 360, (20, 160, 20), -1)
            # Add highlight spot (aesthetic micro-detail)
            cv2.circle(frame, (int(x - r/3.0), int(y - r/3.0)), int(r/4.0), (255, 255, 255), -1)
            
            # Animate positions
            apple["pos"][0] += apple["vel"][0]
            apple["pos"][1] += apple["vel"][1]

        # 3. Draw Tree Trunk on top (Occludes apples that pass behind it!)
        # Draw trunk as a textured brown bar
        cv2.rectangle(frame, (trunk_x_start, 0), (trunk_x_end, height), (30, 60, 90), -1)
        # Texture lines on trunk
        for bark_y in range(0, height, 40):
            cv2.line(frame, (trunk_x_start + 15, bark_y), (trunk_x_start + 15, bark_y + 20), (15, 30, 45), 2)
            cv2.line(frame, (trunk_x_start + 50, bark_y + 15), (trunk_x_start + 50, bark_y + 35), (20, 40, 60), 2)
            
        # 4. Save frame file
        filename = f"frame_{f:03d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)

    print(f"Successfully generated {num_frames} frames under {output_dir}/")
    print("Apples will drift from left to right, passing behind a vertical tree trunk to trigger occlusion.")

if __name__ == "__main__":
    main()
