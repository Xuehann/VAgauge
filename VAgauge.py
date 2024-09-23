import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image

# Streamlit app
st.title("Gauge Reader")

# Upload image
uploaded_file = st.file_uploader("Upload a gauge image", type=["jpg", "jpeg", "png"])

# Canny and Hough Line Transform Sliders
low_threshold = st.slider('Low threshold for Canny', 0, 255, 50)
high_threshold = st.slider('High threshold for Canny', 0, 255, 150)
min_line_length = st.slider('Min Line Length for Hough Lines', 10, 100, 50)
max_line_gap = st.slider('Max Line Gap for Hough Lines', 1, 50, 10)

# Function to calculate average circles
def avg_circles(circles, b):
    avg_x, avg_y, avg_r = 0, 0, 0
    for i in circles[0, :]:
        avg_x += i[0]
        avg_y += i[1]
        avg_r += i[2]
    avg_x /= len(circles[0])
    avg_y /= len(circles[0])
    avg_r /= len(circles[0])

    cv2.circle(b, (int(avg_x), int(avg_y)), int(avg_r), (0, 255, 0), 2)
    cv2.circle(b, (int(avg_x), int(avg_y)), 2, (0, 0, 255), 3)

    return int(avg_x), int(avg_y), int(avg_r)

# Function to calculate the angle
def calculate_angle(x, y, x_needle, y_needle):
    dx = x_needle - x
    dy = y - y_needle
    angle = math.atan2(dy, dx) * (180 / np.pi)
    angle = (angle + 90) % 360
    angle = (360 - angle) % 360
    return angle

# Function to map angle to value
def map_angle_to_value(angle, min_angle, max_angle, min_value, max_value):
    if angle < min_angle:
        angle += 360
    elif angle > max_angle:
        angle -= 360
    value_range = max_value - min_value
    angle_range = max_angle - min_angle
    value = (angle - min_angle) * value_range / angle_range + min_value
    return value

# Function to get current value
def get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    if lines is not None:
        longest_line = None
        max_length = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if dist > max_length:
                max_length = dist
                longest_line = (x1, y1, x2, y2)

        if longest_line is not None:
            x1, y1, x2, y2 = longest_line
            dist_pt_0 = np.sqrt((x - x1)**2 + (y - y1)**2)
            dist_pt_1 = np.sqrt((x - x2)**2 + (y - y2)**2)

            if dist_pt_0 > dist_pt_1:
                x_needle, y_needle = x1, y1
            else:
                x_needle, y_needle = x2, y2

            needle_angle = calculate_angle(x, y, x_needle, y_needle)
            current_value = map_angle_to_value(needle_angle, min_angle, max_angle, min_value, max_value)

            cv2.line(img, (x, y), (x_needle, y_needle), (255, 0, 0), 3)
            cv2.putText(img, 'Value: %.2f' % current_value, (x - 50, y + r + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            return current_value, img

    return None, img

# Function to calibrate the gauge
def calibrate_gauge(image_path, min_angle=40, max_angle=320, min_value=0, max_value=200, separation=10.0):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, int(img.shape[0] * 0.35), int(img.shape[0] * 0.48))

    if circles is not None:
        x, y, r = avg_circles(circles, img)
        current_value, img_with_needle = get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r)
        return current_value, img_with_needle
    else:
        st.write("No circles found!")
        return None, img

if uploaded_file is not None:
    # Convert the uploaded file to a format OpenCV can work with
    img = Image.open(uploaded_file)
    img_np = np.array(img)
    image_path = "uploaded_image.jpg"
    cv2.imwrite(image_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

    # Calibrate the gauge and display the results
    current_value, result_img = calibrate_gauge(image_path)
    if current_value is not None:
        st.image(result_img, caption=f'Gauge Reading: {current_value:.2f}', channels="BGR")
    else:
        st.write("Failed to process the image.")
