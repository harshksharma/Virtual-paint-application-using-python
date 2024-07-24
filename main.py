import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

# Constants for drawing
canvas_width, canvas_height = 888, 555
brush_radius = 5
prev_x, prev_y = 0, 0

# Initialize the canvas and color palette
canvas = np.ones((canvas_height, canvas_width, 3), np.uint8) * 255  # Initialize with white background
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 255, 255),
          (0, 255, 255), (255, 0, 255), (128, 128, 128), (0, 0, 0)]
current_color = colors[0]  # Initial color

# Define finger landmarks (only index fingertip)
finger_tip_landmarks = [8]  # Use the index fingertip landmark

# Smoothness factor for the brushstroke
smoothing_factor = 1  # Adjust this value for the desired smoothness

# Create a color palette image with the available colors
palette = np.zeros((50, canvas_width, 3), np.uint8)
color_palette_width = canvas_width // len(colors)
for i, color in enumerate(colors):
    palette[:, i * color_palette_width:(i + 1) * color_palette_width] = color

# Function to draw on the canvas using fingertip coordinates with smoothing
def draw_on_canvas(x, y):
    global prev_x, prev_y
    smoothed_x = int(smoothing_factor * x + (1 - smoothing_factor) * prev_x)
    smoothed_y = int(smoothing_factor * y + (1 - smoothing_factor) * prev_y)

    cv2.circle(canvas, (smoothed_x, smoothed_y), brush_radius, current_color, -1)
    if prev_x == 0 and prev_y == 0:
        prev_x, prev_y = smoothed_x, smoothed_y
    cv2.line(canvas, (prev_x, prev_y), (smoothed_x, smoothed_y), current_color, brush_radius)
    prev_x, prev_y = smoothed_x, smoothed_y

# Function to handle color selection
def select_color(x):
    global current_color
    color_index = x // color_palette_width
    if color_index < len(colors):
        current_color = colors[color_index]

# Function to update brush radius based on hand gesture
def update_brush_radius(hand_landmarks):
    global brush_radius
    tip_of_thumb = hand_landmarks.landmark[4]  # Thumb tip landmark
    tip_of_index_finger = hand_landmarks.landmark[8]  # Index finger tip landmark

    # Calculate the Euclidean distance between thumb tip and index finger tip
    distance = np.sqrt((tip_of_thumb.x - tip_of_index_finger.x)**2 + (tip_of_thumb.y - tip_of_index_finger.y)**2)

    # Map the distance to brush radius (adjust the scaling factor as needed)
    brush_radius = int(10 + 150 * distance)

# Main loop for video capture and hand tracking
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[8]
            x, y = int(index_finger_tip.x * canvas_width), int(index_finger_tip.y * canvas_height)

            if 0 <= y < 50:
                select_color(x)

            # Update brush radius based on hand gesture
            update_brush_radius(hand_landmarks)

            # Draw using the fingertip
            draw_on_canvas(x, y)

    # Display the self-view video feed
    cv2.imshow('Self-View', frame)

    # Display the canvas
    cv2.imshow('Virtual Paint', canvas)

    # Display the color palette
    cv2.imshow('Color Palette', palette)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

# Release the VideoCapture and destroy OpenCV windows
cap.release()
cv2.destroyAllWindows()
