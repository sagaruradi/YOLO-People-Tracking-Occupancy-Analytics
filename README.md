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


Technologies Used
Python
OpenCV
Ultralytics YOLO
YOLO11n
ByteTrack
NumPy
Matplotlib
How It Works
1. Person Detection

YOLO11n processes each video frame and detects objects.

The project filters the detections to the person class.

classes=[0]

Class 0 corresponds to the person class in the COCO dataset.

2. Object Tracking

ByteTrack associates detections across consecutive frames.

For example:
Frame 1 → Person ID 12
Frame 2 → Person ID 12
Frame 3 → Person ID 12
Frame 4 → Person ID 12
 
The ID allows the system to follow the same track across frames.

A tracking ID represents a tracked object during the video. It is not a guaranteed real-world identity.

3. Line Crossing

A horizontal counting line is placed in the video.

The center point of each person's bounding box is calculated:
center_x = (x1 + x2) / 2
center_y = (y1 + y2) / 2

The system compares the person's previous center position with the current center position.

Current configuration:

Top → Bottom = IN

Bottom → Top = OUT

If the center moves from one side of the line to the other, a crossing event is recorded.

4. Occupancy

The project estimates occupancy using:

Estimated Occupancy
=
Initial Occupancy
+ IN
- OUT

Initial occupancy is estimated from people already present on the designated inside side when the video begins.

5. Occupancy Analytics

The system stores occupancy for every processed frame and generates:

output/occupancy_graph.png

This graph shows how estimated occupancy changes throughout the video.

Example Results

For the included test video:

Initial Occupancy: 4
Total IN: 29
Total OUT: 26
Final Estimated Occupancy: 7
Peak Visible People: 12
Total Frames Processed: 265

These values are specific to the included test video and can change with different videos or tracking parameters.

Output

The system generates:

output/
├── people_analytics.mp4
└── occupancy_graph.png
Annotated Video

The output video contains:

Person bounding boxes
Tracking IDs
Counting line
IN count
OUT count
Estimated occupancy
Currently visible people
Peak visible people
Occupancy Graph

The occupancy graph shows estimated people occupancy over time.

Project Structure


YOLO-Object-Detection-Tracking/
│
├── images/
│   └── test.jpg
│
├── input/
│
├── output/
│   ├── people_analytics.mp4
│   └── occupancy_graph.png
│
├── src/
│   └── main.py
│
├── trackers/
│   └── bytetrack_custom.yaml
│
├── videos/
│   └── test.mp4
│
├── README.md
├── requirements.txt
└── yolo11n.pt


Installation

Create and activate a virtual environment:

python -m venv venv

Activate it on Windows PowerShell:

.\venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt
Running the Project

Place the input video inside:

videos/

Then run:

python src/main.py

The processed video and occupancy graph will be generated inside:

output/
Tracking Configuration

The project uses a custom ByteTrack configuration:

trackers/bytetrack_custom.yaml

Important parameters include:

track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: 30
match_thresh: 0.8
fuse_score: True

The tracker configuration can be adjusted depending on video quality, object density, occlusion, and detection performance.

Limitations

The system has several practical limitations:

Severe occlusion can cause tracking failures.
A person may receive a new tracking ID after a tracking failure.
Line-crossing accuracy depends on the position of the counting line.
False detections can affect counting.
The estimated occupancy depends on the assumption that the initial occupancy is correctly estimated.
The system does not identify people personally; tracking IDs are temporary IDs generated during the tracking process.
The current implementation uses a fixed horizontal counting line.
Future Improvements

Possible improvements include:

Region-of-interest based counting
Better line placement and configurable counting zones
Track-quality filtering
Real-time webcam support
CSV analytics export
Interactive analytics dashboard
Entry/exit event logging
Detection and tracking performance optimization
Support for multiple counting lines
Learning Outcomes

Through this project, the following concepts were implemented:

Object detection
Confidence thresholds
Bounding boxes
Multi-object tracking
Tracking IDs
IoU-based association concepts
Detection-to-tracking pipeline
Line-crossing algorithms
Occupancy estimation
Video processing with OpenCV
Data visualization with Matplotlib