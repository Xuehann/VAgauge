import cv2
import numpy as np
import math

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

def calculate_angle(x, y, x_needle, y_needle):
    dx = x_needle - x
    dy = y - y_needle
    angle = math.atan2(dy, dx) * (180 / np.pi)
    
    # Convert to gauge reading angle
    angle = (angle + 90) % 360
    angle = (360 - angle) % 360
    
    # Debug print statements
    print(f"Center of gauge: ({x}, {y})")
    print(f"Needle position: ({x_needle}, {y_needle})")
    print(f"Angle: {angle} degrees")

    return angle

def map_angle_to_value(angle, min_angle, max_angle, min_value, max_value):
    if angle < min_angle:
        angle += 360
    elif angle > max_angle:
        angle -= 360
    
    value_range = max_value - min_value
    angle_range = max_angle - min_angle
    value = (angle - min_angle) * value_range / angle_range + min_value
    print(f"Angle: {angle}, Mapped value: {value}")
    return value

def get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

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

            return current_value

    print("No needle found!")
    return None

def draw_calibration_lines(img, x, y, r, separation):
    interval = int(360 / separation)
    p1 = np.zeros((interval, 2))
    p2 = np.zeros((interval, 2))
    p_text = np.zeros((interval, 2))
    
    for i in range(interval):
        p1[i] = [x + 0.9 * r * np.cos(separation * i * np.pi / 180), 
                 y + 0.9 * r * np.sin(separation * i * np.pi / 180)]
        p2[i] = [x + r * np.cos(separation * i * np.pi / 180), 
                 y + r * np.sin(separation * i * np.pi / 180)]
        p_text[i] = [x + 1.2 * r * np.cos((separation * (i + 9)) * np.pi / 180), 
                     y + 1.2 * r * np.sin((separation * (i + 9)) * np.pi / 180)]

    for i in range(interval):
        cv2.line(img, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])), (0, 255, 0), 2)
        cv2.putText(img, '%s' % (int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1, cv2.LINE_AA)

def calibrate_gauge(image_path, file_type, min_angle=40, max_angle=320, min_value=0, max_value=200, separation=10.0):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image from {image_path}")
        return None, None, None, None, None, None, None, None
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, np.array([]), 100, 50, int(height*0.35), int(height*0.48))
    
    if circles is not None:
        a, b, c = circles.shape
        x, y, r = avg_circles(circles, img)
    else:
        print("No circles found!")
        return
    
    cv2.circle(img, (x, y), r, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.circle(img, (x, y), 2, (0, 255, 0), 3, cv2.LINE_AA)

    draw_calibration_lines(img, x, y, r, separation)
    
    cv2.imwrite(f'gauge-{gauge_number}-calibration.{file_type}', img)
    
    units = "PSI"
    current_value = get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r)
    if current_value is not None:
        print(f"Gauge value: {current_value} {units}")

    cv2.imwrite(f'gauge-{gauge_number}-reading.{file_type}', img)
    
    return min_angle, max_angle, min_value, max_value, units, x, y, r

# Execution
calibrate_gauge(1, "jpg")