import cv2
import numpy as np


def main():
    img = cv2.imread('/Users/aayushgupta/yolov5/IMG_0142.JPG')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    # Perform adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # Find the contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the detected contours
    img_contours = img.copy()
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours', img_contours)

    # Perform edge detection using the Canny algorithm
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using the Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

    # Draw the detected lines
    img_lines = img.copy()
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Hough Lines', img_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
