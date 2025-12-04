import cv2
import numpy as np

def on_mouse(event, x, y, flags, param):
    global frame
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get BGR from frame
        pixel = frame[y, x]  # BGR
        b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])
        hsv_pixel = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])
        print(f"Clicked at ({x},{y}) -> BGR=({b},{g},{r}) HSV=({h},{s},{v})")

cap = cv2.VideoCapture(0)
cv2.namedWindow("Calibrate")
cv2.setMouseCallback("Calibrate", on_mouse)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1) 
    cv2.imshow("Calibrate", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
