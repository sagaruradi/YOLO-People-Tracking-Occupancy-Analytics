# YOLO People Detection, Tracking & Occupancy Analytics

A computer vision project that detects and tracks people in video using YOLO11 and ByteTrack, performs directional IN/OUT counting using line-crossing logic, and generates occupancy analytics over time.

## Features

- Person detection using YOLO11n Person Detection
- Multi-object tracking using ByteTrack
- Persistent tracking IDs across video frames
- Directional line-crossing detection
- IN and OUT counting
- Estimated occupancy calculation
- Peak visible people detection
- Occupancy-over-time graph
- Annotated output video with tracking IDs and analytics

## Project Architecture

```text
Input Video
    |
    v
OpenCV
    |
    v
YOLO11n Person Detection
    |
    v
ByteTrack
    |
    v
Tracking IDs
    |
    v
Bounding Box Center Points
    |
    v
Line Crossing Detection
    |
    +----------------+
    |                |
    v                v
   IN               OUT
    |                |
    +-------+--------+
            |
            v
   Estimated Occupancy
            |
            v
   Analytics + Graph