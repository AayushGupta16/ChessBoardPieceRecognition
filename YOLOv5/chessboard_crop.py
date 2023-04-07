import cv2
import numpy as np

def main():
    img = cv2.imread('/Users/aayushgupta/yolov5/IMG_0142.JPG')

    # Convert the input image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur with a larger kernel size
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    # Perform adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # Find the contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    max_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    # Find the corners of the largest contour
    corners = cv2.approxPolyDP(largest_contour, 0.05 * cv2.arcLength(largest_contour, True), True)

    # Check if the contour has 4 corners
    if len(corners) == 4:
        src_points = np.float32([corners[0][0], corners[1][0], corners[2][0], corners[3][0]])
        src_points = src_points[np.argsort(src_points[:, 1])]  # Sort by y-coordinate
        if src_points[0][0] > src_points[1][0]:  # Swap points 0 and 1
            src_points[[0, 1]] = src_points[[1, 0]]
        if src_points[2][0] < src_points[3][0]:  # Swap points 2 and 3
            src_points[[2, 3]] = src_points[[3, 2]]

        # Define the destination points
        square_size = 300
        dst_points = np.float32([[0, 0], [square_size - 1, 0], [0, square_size - 1], [square_size - 1, square_size - 1]])

        # Compute the perspective transform matrix and apply it
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        cropped_chessboard = cv2.warpPerspective(img, transform_matrix, (square_size, square_size))

        # Display the cropped chessboard
        cv2.imshow('Cropped Chessboard', cropped_chessboard)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Chessboard not detected. Please try a different image.")

if __name__ == '__main__':
    main()
