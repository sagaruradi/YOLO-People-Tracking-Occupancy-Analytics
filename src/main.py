import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


# =========================================
# LOAD YOLO MODEL
# =========================================

model = YOLO("yolo11n.pt")


# =========================================
# OPEN INPUT VIDEO
# =========================================

video = cv2.VideoCapture("videos/test.mp4")


# Get video properties
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))


# =========================================
# COUNTING LINE
# =========================================

LINE_Y = height // 2


# =========================================
# CREATE OUTPUT VIDEO
# =========================================

output = cv2.VideoWriter(
    "output/people_analytics.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)


# =========================================
# TRACKING / COUNTING DATA
# =========================================

previous_positions = {}

counted_in_ids = set()
counted_out_ids = set()

in_count = 0
out_count = 0

unique_ids = set()

peak_visible = 0


# =========================================
# OCCUPANCY DATA
# =========================================

initial_occupancy = None

occupancy_history = []
time_history = []


# =========================================
# PROCESS VIDEO
# =========================================

frame_number = 0


while True:

    success, frame = video.read()

    if not success:
        break

    frame_number += 1


    # =====================================
    # YOLO + BYTE TRACK
    # =====================================

    results = model.track(
        frame,
        conf=0.25,
        classes=[0],
        tracker="trackers/bytetrack_custom.yaml",
        persist=True,
        verbose=False
    )

    result = results[0]


    # =====================================
    # DRAW DETECTIONS + IDs
    # =====================================

    annotated_frame = result.plot()


    # =====================================
    # DRAW COUNTING LINE
    # =====================================

    cv2.line(
        annotated_frame,
        (0, LINE_Y),
        (width, LINE_Y),
        (255, 0, 0),
        3
    )


    cv2.putText(
        annotated_frame,
        "COUNTING LINE",
        (20, LINE_Y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )


    # =====================================
    # PROCESS TRACKED PEOPLE
    # =====================================

    if result.boxes.id is not None:

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        boxes = (
            result.boxes.xyxy
            .cpu()
            .tolist()
        )


        # Current visible people
        current_people = len(track_ids)


        # Store unique IDs
        unique_ids.update(track_ids)


        # Update peak visible people
        if current_people > peak_visible:
            peak_visible = current_people


        # =================================
        # INITIAL OCCUPANCY
        # =================================

        if initial_occupancy is None:

            initial_occupancy = 0

            for box in boxes:

                x1, y1, x2, y2 = box

                center_y = int((y1 + y2) / 2)

                # Bottom side = inside
                if center_y > LINE_Y:

                    initial_occupancy += 1


        # =================================
        # PROCESS EACH PERSON
        # =================================

        for track_id, box in zip(track_ids, boxes):

            x1, y1, x2, y2 = box


            # Calculate bounding-box center
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)


            # Draw center point
            cv2.circle(
                annotated_frame,
                (center_x, center_y),
                5,
                (0, 255, 255),
                -1
            )


            # =================================
            # CHECK LINE CROSSING
            # =================================

            if track_id in previous_positions:

                previous_y = previous_positions[track_id]


                # =================================
                # TOP → BOTTOM = IN
                # =================================

                if (
                    previous_y < LINE_Y
                    and center_y >= LINE_Y
                    and track_id not in counted_in_ids
                ):

                    in_count += 1

                    counted_in_ids.add(track_id)


                # =================================
                # BOTTOM → TOP = OUT
                # =================================

                elif (
                    previous_y > LINE_Y
                    and center_y <= LINE_Y
                    and track_id not in counted_out_ids
                ):

                    out_count += 1

                    counted_out_ids.add(track_id)


            # Store current position
            previous_positions[track_id] = center_y


    else:

        current_people = 0


    # =====================================
    # ESTIMATED OCCUPANCY
    # =====================================

    if initial_occupancy is None:

        estimated_occupancy = 0

    else:

        estimated_occupancy = (
            initial_occupancy
            + in_count
            - out_count
        )


    # Safety protection
    if estimated_occupancy < 0:

        estimated_occupancy = 0


    # =====================================
    # STORE OCCUPANCY HISTORY
    # =====================================

    current_time = frame_number / fps

    time_history.append(current_time)

    occupancy_history.append(
        estimated_occupancy
    )


    # =====================================
    # DISPLAY ANALYTICS
    # =====================================

    cv2.putText(
        annotated_frame,
        f"IN: {in_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )


    cv2.putText(
        annotated_frame,
        f"OUT: {out_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )


    cv2.putText(
        annotated_frame,
        f"Estimated Occupancy: {estimated_occupancy}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )


    cv2.putText(
        annotated_frame,
        f"People Visible: {current_people}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )


    cv2.putText(
        annotated_frame,
        f"Peak Visible: {peak_visible}",
        (20, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )


    # =====================================
    # SAVE FRAME
    # =====================================

    output.write(annotated_frame)


# =========================================
# RELEASE RESOURCES
# =========================================

video.release()
output.release()


# =========================================
# FINAL ANALYTICS
# =========================================

print("\n=========================================")
print("PEOPLE ANALYTICS COMPLETED")
print("=========================================")

print("\nOutput:")
print("output/people_analytics.mp4")

print("\nAnalytics")
print("-----------------------------------------")

print(f"Initial Occupancy: {initial_occupancy}")
print(f"Total unique tracking IDs: {len(unique_ids)}")
print(f"Total IN: {in_count}")
print(f"Total OUT: {out_count}")

final_occupancy = (
    initial_occupancy
    + in_count
    - out_count
)

if final_occupancy < 0:
    final_occupancy = 0

print(f"Final Estimated Occupancy: {final_occupancy}")
print(f"Peak Visible People: {peak_visible}")
print(f"Total Frames Processed: {frame_number}")


# =========================================
# CREATE OCCUPANCY GRAPH
# =========================================

plt.figure(figsize=(10, 5))

plt.plot(
    time_history,
    occupancy_history
)

plt.xlabel("Time (seconds)")
plt.ylabel("Estimated Occupancy")

plt.title(
    "Estimated People Occupancy Over Time"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "output/occupancy_graph.png",
    dpi=150
)

plt.close()


print("\nOccupancy graph saved to:")
print("output/occupancy_graph.png")

print("\n=========================================")