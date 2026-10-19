# 🛰️ Motion Vision Tracker

[![CI](https://github.com/Sais-gallego/motion-vision-tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/Sais-gallego/motion-vision-tracker/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org)

**Motion Vision Tracker** is an advanced open-source Computer Vision engine built with **OpenCV**, **NumPy**, and **Flask** for real-time kinematic motion detection, Farneback dense optical flow velocity mapping, multi-object trajectory tracing with decaying glow trails, activity accumulation heatmaps, and virtual tripwire perimeter intrusion defense.

---

## 🌟 Key Capabilities

- **Adaptive Foreground Motion Segmentation**:
  - Gaussian Mixture Background Subtraction (`MOG2`) and temporal frame differencing.
  - Morphological noise filtering and minimum/maximum area thresholding.
- **Farneback Dense Optical Flow (360° Color-Wheel)**:
  - Pixel-level velocity vector field calculation.
  - Full-spectrum HSV color mapping (Hue encodes direction of motion, Value encodes speed).
- **Multi-Object Trajectory Tracing**:
  - Persistent Euclidean centroid tracker with automatic target ID assignment.
  - Dynamic glowing trajectory trails that decay over time to indicate velocity and vector orientation.
- **Long-Exposure Activity Heatmap**:
  - Accumulation buffer with decay modeling to map persistent movement hot-zones over time (Turbo/Inferno colormaps).
- **Virtual Tripwires & Perimeter Security Zones**:
  - Configurable polygonal intrusion zones with real-time breach detection and timestamped alarm logs.
- **Built-in Synthetic Kinematic Simulator**:
  - Includes a synthetic multi-target kinematic physics simulator to test, benchmark, and demonstrate the tracker anywhere without requiring a physical camera.
- **Dual Interface**:
  - **CLI**: Batch video processing and telemetry snapshots (`composite_frame`, `fg_mask`, `cumulative_heatmap`).
  - **Live Web Telemetry Dashboard**: MJPEG video streaming server with real-time toggle switches for Bounding Boxes, Optical Flow, Trails, Heatmap, and Tripwires.

---

## 📁 Analytical Output Architecture

```
results/
├── motion_tracking_snapshot.png    # Composite diagnostic snapshot with HUD telemetry
├── motion_mask.png                 # Isolated binary foreground motion mask
└── cumulative_heatmap.png          # Thermal activity density map
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/Sais-gallego/motion-vision-tracker.git
cd motion-vision-tracker
pip install -r requirements.txt
```

### 2. Command Line Usage

Run demonstration on synthetic kinematic stream:
```bash
python -m motion_tracker.cli --demo --frames 120 --flow --heatmap
```

Analyze an external video file:
```bash
python -m motion_tracker.cli --video path/to/security_footage.mp4 --output-dir results/
```

### 3. Launch Real-Time Web Telemetry Dashboard

```bash
python -m motion_tracker.cli --web --port 5055
```
Open [http://localhost:5055](http://localhost:5055) in your browser.

---

## 🧪 Testing

Run automated unit test suite:

```bash
python -m unittest discover tests -v
```

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
