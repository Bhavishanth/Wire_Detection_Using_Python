import cv2
from util import get_limits

blue = [255, 0, 0]
yellow = [0, 255, 255]
red = [0, 0, 255]
green = [0, 255, 0]
black = [0, 0, 0]
cap = cv2.VideoCapture("testVideo1.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if no frame is captured

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color limits
    lowerYellow, upperYellow = get_limits(color=yellow)
    lowerBlue, upperBlue = get_limits(color=blue)
    lowerRed, upperRed = get_limits(color=red)
    lowerGreen, upperGreen = get_limits(color=green)
    # lowerBlack, upperBlack = get_limits(color=black)

    # Create masks for each color
    maskYellow = cv2.inRange(hsvImage, lowerYellow, upperYellow)
    maskBlue = cv2.inRange(hsvImage, lowerBlue, upperBlue)
    maskRed = cv2.inRange(hsvImage, lowerRed, upperRed)
    maskGreen = cv2.inRange(hsvImage, lowerGreen, upperGreen)
    # maskBlack = cv2.inRange(hsvImage, lowerBlack, upperBlack)

    # Function to draw contours and label them
    def draw_contours_and_label(mask, color, label):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter out small contours
                cv2.drawContours(frame, [contour], -1, color, 7)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(frame, label, (x + w // 2, y + h // 2), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2, cv2.LINE_AA)

    # Draw and label contours for each color
    draw_contours_and_label(maskYellow, (0, 255, 255), "Yellow")
    draw_contours_and_label(maskBlue, (200, 0, 0), "Blue")
    draw_contours_and_label(maskRed, (0, 0, 255), "Red")
    draw_contours_and_label(maskGreen, (0, 255, 0), "Green")
    # draw_contours_and_label(maskBlack, (255, 255, 255), "Black")

    # Display the frame
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
