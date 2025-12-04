import cv2
import numpy as np


# HSV range for hand 

LOWER_HSV = np.array([150, 20, 50])  
UPPER_HSV = np.array([180, 70, 100])

# Minimum contour area
MIN_CONTOUR_AREA = 2000

# Distance thresholds relative to frame width

WARNING_REL = 0.30 #Distance less than 30% of frame width
DANGER_REL  = 0.10 #Distance less than 10% of frame width


def get_hand_fingertip(mask):
    """
    Given a binary mask, find the largest contour (hand)
    and return the fingertip (top-most point) + contour.

    Returns (fx, fy, contour) or (None, None, None) if no valid hand is found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None

    # Largest contour by area (assume it's the hand)
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < MIN_CONTOUR_AREA:
        return None, None, None

    # Find the top-most point of the contour (smallest y)
    # contour shape: (N, 1, 2)
    ys = largest[:, 0, 1]
    min_index = np.argmin(ys)
    fx, fy = largest[min_index, 0, 0], largest[min_index, 0, 1]

    return int(fx), int(fy), largest


def compute_point_to_rect_distance(px, py, x1, y1, x2, y2):
    """
    Compute shortest distance from a point (px, py) to axis-aligned rectangle (x1, y1, x2, y2).
    If point is inside rectangle, distance is 0.
    """
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return np.sqrt(dx * dx + dy * dy)


def process_frame(frame):
    """
    Process a single BGR frame:
    - detect hand via color segmentation
    - compute fingertip position
    - compute distance to virtual rectangle
    - classify SAFE / WARNING / DANGER
    - draw overlays

    Returns:
        annotated_frame, state (str), dist (float or None)
    """
    # Flip horizontally to behave like a mirror
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # Define the virtual object rectangle on right side
    rect_width = int(w * 0.2)
    rect_x1 = w - rect_width - 20
    rect_x2 = w - 20
    rect_y1 = int(h * 0.2)
    rect_y2 = int(h * 0.8)

    # Convert to HSV and threshold for hand/glove color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 5)

    # Find fingertip
    fx, fy, contour = get_hand_fingertip(mask)

    # Draw virtual rectangle
    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 0, 0), 2)
    cv2.putText(
        frame, "VIRTUAL OBJECT", (rect_x1, rect_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
    )

    state = "NO HAND"
    dist = None

    warning_thresh = WARNING_REL * w
    danger_thresh  = DANGER_REL * w

    if fx is not None and fy is not None:
        # Draw contour and fingertip
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        cv2.circle(frame, (fx, fy), 7, (0, 0, 255), -1)
        cv2.putText(
            frame, f"Fingertip ({fx}, {fy})", (fx + 10, fy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
        )

        # Compute distance to rectangle
        dist = compute_point_to_rect_distance(fx, fy, rect_x1, rect_y1, rect_x2, rect_y2)

        # State machine
        if dist <= 0.0:
            state = "DANGER DANGER"
        else:
            if dist > warning_thresh:
                state = "SAFE"
            elif dist > danger_thresh:
                state = "WARNING"
            else:
                state = "DANGER"

    # Choose color based on state
    if state == "SAFE":
        state_color = (0, 255, 0)
    elif state == "WARNING":
        state_color = (0, 255, 255)
    elif state == "DANGER":
        state_color = (0, 0, 255)
    else:  # NO HAND
        state_color = (255, 255, 255)

    # Overlay state text
    cv2.putText(
        frame, f"STATE: {state}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, state_color, 3
    )

    # Distance text (if available)
    if dist is not None:
        cv2.putText(
            frame, f"Distance: {dist:.1f}px", (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2
        )

    # Big warning for DANGER
    if state == "DANGER":
        cv2.putText(
            frame, "DANGER DANGER", (int(w * 0.15), int(h * 0.5)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4
        )

    return frame, state, dist


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Optionally set a smaller resolution to boost FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        annotated, state, dist = process_frame(frame)

        cv2.imshow("Real time Fingertip Tracking - Virtual Object", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
