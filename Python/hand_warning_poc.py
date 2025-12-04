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


def process_frame(frame):

    #Flip horizontally to behave like a mirror
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    #Define the virtual object rectangle on right side
    rect_width = int(w * 0.2)
    rect_x1 = w - rect_width - 20
    rect_x2 = w - 20
    rect_y1 = int(h * 0.2)
    rect_y2 = int(h * 0.8)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 5)

    fx, fy, contour = get_hand_fingertip(mask)




def main():
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    #Smaller resolution to boost FPS
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


    