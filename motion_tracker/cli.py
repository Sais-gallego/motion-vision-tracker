"""Command Line Interface for Motion Vision Tracker."""

import os
import sys
import argparse
import cv2
from .engine import MotionTrackingEngine
from .simulator import SyntheticVideoGenerator

def main():
    parser = argparse.ArgumentParser(
        description="Motion Vision Tracker: Motion Detection, Trajectory Tracing & Optical Flow Engine"
    )
    parser.add_argument("--video", "-v", type=str, help="Path to input video file (MP4, AVI)")
    parser.add_argument("--demo", action="store_true", help="Run demonstration with synthetic dynamic stream")
    parser.add_argument("--frames", "-n", type=int, default=120, help="Number of frames to process in demo mode (default: 120)")
    parser.add_argument("--output-dir", "-o", type=str, default="results", help="Directory to save analytical snapshots")
    parser.add_argument("--web", action="store_true", help="Launch real-time web telemetry dashboard")
    parser.add_argument("--port", type=int, default=5055, help="Web dashboard port (default: 5055)")
    parser.add_argument("--flow", action="store_true", help="Enable dense optical flow calculation")
    parser.add_argument("--heatmap", action="store_true", help="Enable cumulative activity heatmap")

    args = parser.parse_args()

    if args.web:
        from .app import run_app
        run_app(port=args.port)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    engine = MotionTrackingEngine(enable_optical_flow=True, enable_heatmap=True)
    engine.setup_default_zones(640, 480)

    if args.video:
        if not os.path.exists(args.video):
            print(f"[!] Error: File not found: {args.video}", file=sys.stderr)
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        print(f"[*] Processing video source: {args.video}")
    else:
        print("[*] Running in DEMO mode with synthetic motion generator...")
        sim = SyntheticVideoGenerator(640, 480, num_targets=3)
        cap = None

    frame_idx = 0
    max_frames = args.frames if not args.video else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 300)
    
    last_res = None
    print(f"[*] Analyzing {max_frames} frames of kinematic motion and optical flow...")
    
    while frame_idx < max_frames:
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = sim.next_frame()

        res = engine.process_frame(
            frame,
            show_boxes=True,
            show_flow=args.flow,
            show_trails=True,
            show_heatmap=args.heatmap,
            show_zones=True
        )
        last_res = res
        frame_idx += 1
        
        if frame_idx % 30 == 0 or frame_idx == max_frames:
            print(f"  -> Frame {frame_idx}/{max_frames} | FPS: {res['fps']} | Moving Objects: {res['detected_count']} | Active Tracks: {res['tracked_count']}")

    if cap:
        cap.release()

    if last_res is not None:
        composite_path = os.path.join(args.output_dir, "motion_tracking_snapshot.png")
        mask_path = os.path.join(args.output_dir, "motion_mask.png")
        heatmap_path = os.path.join(args.output_dir, "cumulative_heatmap.png")
        
        cv2.imwrite(composite_path, last_res["composite_frame"])
        cv2.imwrite(mask_path, last_res["fg_mask"])
        
        if engine.heatmap:
            heat_img = engine.heatmap.generate_heatmap(last_res["composite_frame"], alpha=0.7)
            cv2.imwrite(heatmap_path, heat_img)

        print("\n" + "=" * 50)
        print("      MOTION TRACKING TELEMETRY SUMMARY")
        print("=" * 50)
        print(f"• Total Processed Frames:     {frame_idx}")
        print(f"• Average Pipeline Speed:     {last_res['fps']} FPS")
        print(f"• Persistent Tracked Targets: {last_res['tracked_count']}")
        stats = last_res.get("heatmap_stats", {})
        print(f"• Monitored Activity Area:    {stats.get('active_area_percent', 0)}%")
        print("=" * 50)
        print(f"[+] Tracking Snapshot saved to: {composite_path}")
        print(f"[+] Foreground Mask saved to:   {mask_path}")
        if engine.heatmap:
            print(f"[+] Activity Heatmap saved to:  {heatmap_path}")

if __name__ == "__main__":
    main()
