import cv2
import numpy as np

num_spaces = 13

# Attempt 4
pt_top_left = (75, 270)
pt_top_right = (1045, 270)
pt_bottom_left = (-10, 480)
pt_bottom_right = (1165, 480)

pts_top = np.array([
    np.linspace(pt_top_left[0], pt_top_right[0], num_spaces + 1),
    np.linspace(pt_top_left[1], pt_top_right[1], num_spaces + 1)
]).T

pts_bottom = np.array([
    np.linspace(pt_bottom_left[0], pt_bottom_right[0], num_spaces + 1),
    np.linspace(pt_bottom_left[1], pt_bottom_right[1], num_spaces + 1)
]).T

cap = cv2.VideoCapture('DLIP_parking_test_video.avi')
cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
ret, frame = cap.read()
cap.release()

if ret:
    for i in range(num_spaces):
        poly = np.array([
            pts_top[i],
            pts_top[i+1],
            pts_bottom[i+1],
            pts_bottom[i]
        ], np.int32)
        cv2.polylines(frame, [poly], True, (0, 255, 0), 2)
        cx = int(np.mean(poly[:, 0]))
        cy = int(np.mean(poly[:, 1]))
        cv2.putText(frame, str(i+1), (cx-10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite('parking_spaces_test4.jpg', frame)
