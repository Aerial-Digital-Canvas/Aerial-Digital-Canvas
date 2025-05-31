import numpy as np
import cv2 as cv
import time
import subprocess
import mainmath as mm
from hands import HandDetector
from canvas import Canvas
import mediapipe as mp
import shapes as s

# Variables for dropdown
shapes = ["Circle", "Square", "Triangle", "Pentagon", "Hexagon", "Heptagon", "Octagon", "Nonagon"]
selected_shape = None
dropdown_visible = False
dropdown_x, dropdown_y = 20, 70
dropdown_width, dropdown_height = 140, 30
# Clear button position
clear_X, clear_Y = 20, 10
clear_width, clear_height = 160, 30
undo_X, undo_Y = 480, 10
redo_X, redo_Y = 560, 10
button_width, button_height = 70, 30

# Variables for hover and toggling
hover_start_time = None
hover_threshold = 2  # 2 seconds to toggle
last_hover_time = 0  # To track when the dropdown was last toggled

# List to store selected shapes and their properties
shapes_list = []
undo_stack = []
redo_stack = []

# Dragging state
dragging = False
drag_start_time = None
dragging_shape = None  # Keep track of the shape being dragged

# Variables for resizing
resizing = False
resizing_shape = None
resize_start_point = None

#keyboard variables:
last_blink_time = time.time()
cursor_visible = True
cursor_blink_interval = 0.5

drawing_mode_k = False
keyboard_mode_k = False
black_screen_k = False
drawn_points_k = []
hover_start_times_k = {}
pressed_keys_k = []
pressed_keys_k_total_lines_k = []
selected_color_k = (255, 255, 255)
text_positions_k = [(50, 100 + i * 50) for i in range(20)]
dragging_line_k = None
smooth_factor_k = 0.1

# Keyboard layout
KEYS = [
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", "Space"],
    ["Z", "X", "C", "V", "B", "N", "M", "Back", "Enter"]
]
KEY_WIDTH_K, KEY_HEIGHT_K, KEY_PADDING_K = 45, 45, 10
HOVER_TIME_K = 0.5

# Button properties
BUTTON_WIDTH_K, BUTTON_HEIGHT_K = 180, 30
BUTTON_X_K, BUTTON_Y_K = 450, 60

def clear_canvas():
    shapes_list.clear()
    undo_stack.clear()
    redo_stack.clear()
    #frame.setTo((0, 0, 0))

# Undo function
def undo_action():
    if shapes_list:
        redo_stack.append(shapes_list.pop())

# Redo function
def redo_action():
    if redo_stack:
        shapes_list.append(redo_stack.pop())

# Function to draw polygons
def draw_polygon(img, position, size, sides, color=(255, 255, 255), thickness=2):
    angle_step = 2 * np.pi / sides
    points = []
    
    for i in range(sides):
        angle = i * angle_step
        x = int(position[0] + size * np.cos(angle))
        y = int(position[1] + size * np.sin(angle))
        points.append((x, y))
    
    points = np.array(points, np.int32)
    cv.polylines(img, [points], isClosed=True, color=color, thickness=thickness)

#drawing buttons
def draw_clear_button(img):
    cv.rectangle(img, (clear_X, clear_Y), (clear_X + clear_width, clear_Y + clear_height), (0, 0, 255), -1)
    cv.putText(img, "Clear Canvas", (clear_X + 10, clear_Y + 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv.rectangle(img, (undo_X, undo_Y), (undo_X + button_width, undo_Y + button_height), (0, 255, 0), -1)
    cv.putText(img, "Undo", (undo_X + 10, undo_Y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv.rectangle(img, (redo_X, redo_Y), (redo_X + button_width, redo_Y + button_height), (255, 0, 0), -1)
    cv.putText(img, "Redo", (redo_X + 10, redo_Y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_shapes(img):
    for shape in shapes_list:
        shape_type = shape['type']
        position = shape['position']
        size = shape['size']
        color = (0, 255, 0) if dragging and dragging_shape == shape else (255, 255, 255)
        thickness = 3 if dragging and dragging_shape == shape else 2
        
        if shape_type == "Circle":
            cv.circle(img, tuple(position), size // 2, color, thickness)
        elif shape_type == "Square":
            top_left = (position[0] - size // 2, position[1] - size // 2)
            bottom_right = (position[0] + size // 2, position[1] + size // 2)
            cv.rectangle(img, top_left, bottom_right, color, thickness)
        elif shape_type == "Triangle":
            pts = np.array([
                (position[0], position[1] - size // 2),
                (position[0] - size // 2, position[1] + size // 2),
                (position[0] + size // 2, position[1] + size // 2)
            ], np.int32)
            cv.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
        elif shape_type in ["Pentagon", "Hexagon", "Heptagon", "Octagon", "Nonagon"]:
            sides = {
                "Pentagon": 5,
                "Hexagon": 6,
                "Heptagon": 7,
                "Octagon": 8,
                "Nonagon": 9
            }
            draw_polygon(img, position, size, sides[shape_type], color, thickness)
# Add a new shape to the list
def add_shape(shape_type):
    shape = {
        'type': shape_type,
        'position': [300, 300],  # Default position for new shapes
        'size': 100
    }
    shapes_list.append(shape)
    undo_stack.append(shape)
    redo_stack.clear()
    
# Function to check if a point is near the corner of a square (for resizing)
def is_near_corner(shape, x, y, tolerance=10):
    top_left = (shape['position'][0] - shape['size'] // 2, shape['position'][1] - shape['size'] // 2)
    bottom_right = (shape['position'][0] + shape['size'] // 2, shape['position'][1] + shape['size'] // 2)

    # Check if the point is within the corner tolerance
    corners = [
        top_left,  # top-left
        (bottom_right[0], top_left[1]),  # top-right
        (top_left[0], bottom_right[1]),  # bottom-left
        bottom_right  # bottom-right
    ]
    
    for corner in corners:
        if abs(corner[0] - x) < tolerance and abs(corner[1] - y) < tolerance:
            return True
    return False

# Handle resizing logic
def handle_resizing(x, y):
    global resizing, resizing_shape, resize_start_point

    def is_near_resize_area(shape, x, y):
        # "Check if the cursor is near the resize area of the shape."
        cx, cy = shape['position']
        if shape['type'] == 'Circle':
            radius = shape['size'] // 2
            return abs(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius) <= 10
        elif shape['type'] in ['Square', "Pentagon", "Hexagon", "Heptagon", "Octagon", "Nonagon"]:
            return is_near_corner(shape, x, y)
        return False

    if resizing:
        # Resize the active shape
        dx, dy = x - resize_start_point[0], y - resize_start_point[1]
        resizing_shape['size'] = max(resizing_shape['size'] + (dx + dy) // 2, 20)  # Minimum size
        resize_start_point = (x, y)

        # Stop resizing if no longer near the resize area
        if not is_near_resize_area(resizing_shape, x, y):
            resizing, resizing_shape, resize_start_point = False, None, None
    else:
        # Start resizing if near any shape's resize area
        for shape in shapes_list:
            if is_near_resize_area(shape, x, y):
                resizing, resizing_shape, resize_start_point = True, shape, (x, y)
                break


# Handle dragging logic
def is_inside_shape(shape, x, y):
    """Check if a point is inside any shape with some tolerance"""
    cx, cy = shape['position']
    size = shape['size']
    tolerance = 20  # Extra pixels around shape for easier selection
    
    if shape['type'] == 'Circle':
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return distance <= (size // 2) + tolerance
    elif shape['type'] == 'Square':
        return (cx - size//2 - tolerance <= x <= cx + size//2 + tolerance and 
                cy - size//2 - tolerance <= y <= cy + size//2 + tolerance)
    elif shape['type'] == 'Triangle':
        # Simple triangle hit detection with expanded bounds
        top = (cx, cy - size//2 - tolerance)
        left = (cx - size//2 - tolerance, cy + size//2 + tolerance)
        right = (cx + size//2 + tolerance, cy + size//2 + tolerance)
        return (y >= top[1] and 
                y <= left[1] and 
                x >= left[0] and 
                x <= right[0])
    else:  # For polygons
        # Approximate with circle for simplicity
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        return distance <= size + tolerance

def handle_dragging(hand_landmarks, height, width):
    global dragging, drag_start_time, dragging_shape
    
    if hand_landmarks is None:
        if dragging:
            dragging = False
            dragging_shape = None
            drag_start_time = None
        return

    # Convert normalized coordinates to pixel coordinates
    x_index = int(hand_landmarks.landmark[8].x * width)
    y_index = int(hand_landmarks.landmark[8].y * height)
    x_middle = int(hand_landmarks.landmark[12].x * width)
    y_middle = int(hand_landmarks.landmark[12].y * height)

    # Calculate center point between index and middle fingers
    mid_x = (x_index + x_middle) // 2
    mid_y = (y_index + y_middle) // 2

    if not dragging:
        # Check if we're hovering over any shape
        for shape in reversed(shapes_list):  # Check from top to bottom
            if is_inside_shape(shape, mid_x, mid_y):
                if drag_start_time is None:
                    drag_start_time = time.time()
                elif time.time() - drag_start_time >= hover_threshold:
                    dragging = True
                    dragging_shape = shape
                    # Store the offset from shape center to finger position
                    dragging_offset = [
                        mid_x - shape['position'][0],
                        mid_y - shape['position'][1]
                    ]
                    break
                return
        else:
            # Not hovering over any shape, reset timer
            drag_start_time = None
    else:
        # Currently dragging - update shape position with offset
        if dragging_shape in shapes_list:
            dragging_shape['position'] = [mid_x - dragging_offset[0], mid_y - dragging_offset[1]]
        else:
            dragging = False
            dragging_shape = None
            drag_start_time = None

# Draw the dropdown menu
def draw_dropdown(img):
    cv.rectangle(img, (dropdown_x, dropdown_y), (dropdown_x + dropdown_width, dropdown_y + dropdown_height), (200, 200, 200), -1)
    cv.putText(img, "Select Shape", (dropdown_x + 10, dropdown_y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if dropdown_visible:
        for i, shape in enumerate(shapes):
            option_y = dropdown_y + dropdown_height + i * 40
            cv.rectangle(img, (dropdown_x, option_y), (dropdown_x + dropdown_width, option_y + 30), (240, 240, 240), -1)
            cv.putText(img, shape, (dropdown_x + 10, option_y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
dropdown_click_time = None

# Detect hand hovering over a button
def is_hovering_button(x, y, button_x, button_y, button_width, button_height):
    return button_x <= x <= button_x + button_width and button_y <= y <= button_y + button_height

#keyboard functions:
def draw_toggle_button(image, is_keyboard_visible):
    label = "Hide Keyboard" if is_keyboard_visible else "Show Keyboard"
    cv.rectangle(image, (BUTTON_X_K, BUTTON_Y_K),
                  (BUTTON_X_K + BUTTON_WIDTH_K, BUTTON_Y_K + BUTTON_HEIGHT_K),
                  (200, 100, 100), -1)
    cv.rectangle(image, (BUTTON_X_K, BUTTON_Y_K),
                  (BUTTON_X_K + BUTTON_WIDTH_K, BUTTON_Y_K + BUTTON_HEIGHT_K),
                  (255, 255, 255), 2)
    text_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = BUTTON_X_K + (BUTTON_WIDTH_K - text_size[0]) // 2
    text_y = BUTTON_Y_K + (BUTTON_HEIGHT_K + text_size[1]) // 2 - 5
    cv.putText(image, label, (text_x, text_y),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_keyboard(image):
    h, w, _ = image.shape
    start_x = (w - (len(KEYS[0]) * (KEY_WIDTH_K + KEY_PADDING_K))) // 2
    start_y = h - ((len(KEYS)) * (KEY_HEIGHT_K + KEY_PADDING_K)) - 50

    key_positions = []
    for row_idx, row in enumerate(KEYS):
        for col_idx, key in enumerate(row):
            x1 = start_x + col_idx * (KEY_WIDTH_K + KEY_PADDING_K)
            y1 = start_y + row_idx * (KEY_HEIGHT_K + KEY_PADDING_K)
            x2 = x1 + KEY_WIDTH_K
            y2 = y1 + KEY_HEIGHT_K

            if key == "Back":
                x2 += KEY_WIDTH_K
            elif key == "Space":
                x2 += KEY_WIDTH_K
            elif key == "Enter":
                x1 += KEY_PADDING_K * 5
                x2 += KEY_WIDTH_K * 2

            if key in hover_start_times_k and time.time() - hover_start_times_k[key] < HOVER_TIME_K:
                cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), -1)
            else:
                cv.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)

            cv.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 2)
            text_size = cv.getTextSize(key, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_center_x = x1 + (x2 - x1 - text_size[0]) // 2
            text_center_y = y1 + (KEY_HEIGHT_K + text_size[1]) // 2
            cv.putText(image, key, (text_center_x, text_center_y),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            key_positions.append(((x1, y1, x2, y2), key))
    return image, key_positions

def is_inside_key(x, y, key_position):
    x1, y1, x2, y2 = key_position
    return x1 <= x <= x2 and y1 <= y <= y2

def display_typed_text(image, typed_text, base_y=100):
    lines = typed_text.split('\n')
    for idx, line in enumerate(lines):
        y = base_y + idx * 50
        cv.putText(image, line, (50, y),
                    cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def display_keyboard(key_positions, keyboard_mode_k, frame,x, y, pressed_keys_k):
    global cursor_visible, last_blink_time
    # cursor_visible=True
    # last_blink_time = time.time()
    if keyboard_mode_k:
        for key_position, key in key_positions:
                        if is_inside_key(x, y, key_position):
                            if key not in hover_start_times_k:
                                hover_start_times_k[key] = time.time()
                            elif time.time() - hover_start_times_k[key] >= HOVER_TIME_K:
                                if key == "Back":
                                    if pressed_keys_k:
                                        pressed_keys_k.pop()
                                elif key == "Space":
                                    pressed_keys_k.append(" ")
                                elif key == "Enter":
                                    if pressed_keys_k:
                                        pressed_keys_k_total_lines_k.append("".join(pressed_keys_k))
                                        pressed_keys_k = []
                                else:
                                    pressed_keys_k.append(key)
                                hover_start_times_k.pop(key, None)
                        else:
                            hover_start_times_k.pop(key, None)
    
    typed_text = "".join(pressed_keys_k)
    # print("Current typed:", "".join(pressed_keys_k))
    # print("Submitted lines:", pressed_keys_k_total_lines_k)
    #cv.putText(frame, "DEBUG", (50, 300), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    display_typed_text(frame, typed_text)

    current_time = time.time()
    if current_time - last_blink_time >= cursor_blink_interval:
        cursor_visible = not cursor_visible
        last_blink_time = current_time

    if typed_text:
        lines = typed_text.split('\n')
        last_line = lines[-1]
        text_width, _ = cv.getTextSize(last_line, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cursor_position = (50 + text_width, 100 + 50 * (len(lines) - 1))
        if cursor_visible:
            cv.line(frame, cursor_position, (cursor_position[0], cursor_position[1] + 20), (255, 255, 255), 2)
    else:
        for idx, line in enumerate(pressed_keys_k_total_lines_k):
            if idx < len(text_positions_k):
                x, y = text_positions_k[idx]
                cv.putText(frame, line, (x, y),
                            cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def main():
# Main loop
    cap = cv.VideoCapture(0)
    width_c = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) + 0.5)
    height_c = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) + 0.5)
    # Initialize all global variables
    global dropdown_visible, dropdown_click_time, hover_start_time
    global resizing, resizing_shape, resize_start_point
    global dragging, drag_start_time, dragging_shape, dragging_offset
    global pressed_keys_k_total_lines_k, text_positions_k

    # Set initial values
    dropdown_visible = False
    dropdown_click_time = None
    hover_start_time = None
    resizing = False
    resizing_shape = None
    resize_start_point = None
    dragging = False
    drag_start_time = None
    dragging_shape = None
    dragging_line_k = None
    dragging_offset = [0, 0]
    global keyboard_mode_k, pressed_keys_k
    keyboard_mode_k = False
    key_positions = []
    cursor_visible = True
    last_blink_time = time.time()
    pressed_keys_k_total_lines_k = []
    text_positions_k = [(50, 100 + i * 50) for i in range(10)]  

    # set the default background mode (CAM/BLACK)
    background_mode = 'BLACK'
    #shape_mode = Shape()
    # initialize the canvas element and hand-detector program
    canvas = Canvas(width_c, height_c)
    detector = HandDetector(background_mode)
    
    while True:
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)
        if background_mode == 'BLACK':
            black_frame = np.zeros((height_c, width_c, 3), dtype="uint8")
            request = detector.determine_gesture(frame, black_frame)
            frame = black_frame
        else:
            request = detector.determine_gesture(frame, frame)
        
        #frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # result = hands.process(frame_rgb)

        height, width, _ = frame.shape

    # Draw dropdown and selected shapes
        draw_dropdown(frame)
        draw_shapes(frame)
        draw_clear_button(frame)
        draw_toggle_button(frame, keyboard_mode_k)
        # hand_landmarks = None
        gesture = request.get('gesture')

        if gesture is not None:
            if 'idx_fing_tip' in request:
                x, y = request['idx_fing_tip'][1], request['idx_fing_tip'][2]
                idx_finger = request['idx_fing_tip']  # coordinates of tip of index finger
                _, c, r = idx_finger
    
                data = {'idx_finger': idx_finger}
            # rows, cols, _ = frame.shape
            # hand_landmarks = request.get('hand_landmarks', None)
            # hand_landmarks = request.get('hand_landmarks')
            # if request and 'idx_fing_tip' in request:
            #     x, y = request['idx_fing_tip'][1], request['idx_fing_tip'][2]
                if is_hovering_button(x, y, clear_X, clear_Y, clear_width, clear_height):
                    if hover_start_time is None:
                        hover_start_time = time.time()
                    elif time.time() - hover_start_time >= hover_threshold:
                        clear_canvas()
                        hover_start_time = None  # Reset hover time after clearing
                elif is_hovering_button(x, y, undo_X, undo_Y, button_width, button_height):
                    if hover_start_time is None:
                        hover_start_time = time.time()
                    elif time.time() - hover_start_time >= hover_threshold:
                        undo_action()
                        hover_start_time = None
                elif is_hovering_button(x, y, redo_X, redo_Y, button_width, button_height):
                    if hover_start_time is None:
                        hover_start_time = time.time()
                    elif time.time() - hover_start_time >= hover_threshold:
                        redo_action()
                        hover_start_time = None
                if is_hovering_button(x, y, BUTTON_X_K, BUTTON_Y_K, BUTTON_WIDTH_K, BUTTON_HEIGHT_K):
                    if 'TOGGLE' not in hover_start_times_k:
                        hover_start_times_k['TOGGLE'] = time.time()
                    elif time.time() - hover_start_times_k['TOGGLE'] >= HOVER_TIME_K:
                        keyboard_mode_k = not keyboard_mode_k
                        hover_start_times_k.pop('TOGGLE')
                else:
                    hover_start_times_k.pop('TOGGLE', None)
                # x = int(hand_landmarks.landmark[8].x * width)
                # y = int(hand_landmarks.landmark[8].y * height)
                # Check if hovering over dropdown button
                if is_hovering_button(x, y, dropdown_x, dropdown_y, dropdown_width, dropdown_height):
                    # If the finger is hovering over the dropdown button, check for click
                    if dropdown_click_time is None:
                        dropdown_click_time = time.time()  # Start timer when dropdown is clicked

                    hover_duration = time.time() - dropdown_click_time
                    if hover_duration >= hover_threshold:
                        # Toggle dropdown visibility when clicked and held for more than 2 seconds
                        dropdown_visible = not dropdown_visible
                        dropdown_click_time = None  # Reset click time after toggle

            # If dropdown is visible, check hover on dropdown options
                # If dropdown is visible, check hover on dropdown options
                if dropdown_visible and dropdown_click_time is not None:  # Only check if click time was set
                    for i, shape in enumerate(shapes):
                        option_y = dropdown_y + dropdown_height + i * 40
                        if dropdown_x <= x <= dropdown_x + dropdown_width and option_y <= y <= option_y + 30:
                            hover_duration = time.time() - dropdown_click_time
                            if hover_duration >= hover_threshold:  # Ensure user selects a shape after hovering
                                add_shape(shape)  # Add the selected shape to the list
                                dropdown_visible = False  # Close dropdown after selecting
                                hover_start_time = None  # Reset hover time
                                dropdown_click_time = None
                
            # Handle dragging and resizing of shapes
                # In your main loop, replace the current handle_dragging call with:
                if 'hand_landmarks' in request:
                    hand_landmarks = request['hand_landmarks']
                    handle_dragging(hand_landmarks, height, width)
                    if 'idx_fing_tip' in request:
                        x, y = request['idx_fing_tip'][1], request['idx_fing_tip'][2]
                        handle_resizing(x, y)
                else:
                    # Reset dragging state if no hand is detected
                    dragging = False
                    dragging_shape = None
                    drag_start_time = None
                handle_resizing(x, y)
            if 'idx_fing_tip' in request:
                _, c, r = request['idx_fing_tip']
                if 0 < c < width and 0 < r < height:
                    if gesture == "DRAW":
                        canvas.push_point((r, c))
                    elif gesture == "ERASE":
                        canvas.end_line()
                        if 'mid_fing_tip' in request and 'idx_mid_radius' in request:
                            _, mid_r, mid_c = request['mid_fing_tip']
                            radius = request['idx_mid_radius']
                            canvas.erase_mode((mid_r, mid_c), int(radius * 0.5))
                    elif gesture == "HOVER":
                        canvas.end_line()
                    elif gesture == "SHAPE_LAUNCH":
                        s.main()
                    elif gesture == "MATH_LAUNCH":
                        mm.main()
                    elif gesture == "MOVE":
                        canvas.end_line()
                        if 'idx_pinky_radius' in request and 'shift' in request:
                            idx_position = (r, c)
                            shift = request['shift']
                            radius = request['idx_pinky_radius']
                            radius = int(radius * 0.8)
                            canvas.translate_mode(idx_position, int(radius * 0.5), shift)

            frame = canvas.draw_dashboard(frame, gesture, data=data)
        else:
            frame = canvas.draw_dashboard(frame)
            canvas.end_line()

        # Draw canvas and display frame
        canvas_image = np.zeros_like(frame)
        canvas_image = canvas.draw_lines(canvas_image)
        frame = cv.addWeighted(frame, 0.8, canvas_image, 0.2, 0)
        # Always display submitted lines (even when keyboard is off)
        for idx, (line_x, line_y) in enumerate(text_positions_k):
            if idx >= len(pressed_keys_k_total_lines_k):
                continue
            line = pressed_keys_k_total_lines_k[idx]
            cv.putText(frame, line, (line_x, line_y), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if keyboard_mode_k:
            # if keyboard_mode_k:
            frame, key_positions = draw_keyboard(frame)
            display_keyboard(key_positions, keyboard_mode_k, frame, x, y, pressed_keys_k)
        else:
            for idx, (line_x, line_y) in enumerate(text_positions_k):
                if idx >= len(pressed_keys_k_total_lines_k):
                    continue
                text_line = pressed_keys_k_total_lines_k[idx]
                text_width, text_height = cv.getTextSize(text_line, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                if line_x <= x <= line_x + text_width and line_y - text_height <= y <= line_y:
                    dragging_line_k = idx
                    break

            if dragging_line_k is not None and dragging_line_k < len(text_positions_k):
                current_x, current_y = text_positions_k[dragging_line_k]
                text_line = pressed_keys_k_total_lines_k[dragging_line_k]
                text_width, _ = cv.getTextSize(text_line, cv.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                target_x = x - text_width // 2
                target_y = y
                new_x = current_x + (target_x - current_x) * smooth_factor_k
                new_y = current_y + (target_y - current_y) * smooth_factor_k
                text_positions_k[dragging_line_k] = (int(new_x), int(new_y))

        cv.namedWindow("Airdraw", cv.WND_PROP_FULLSCREEN)
        cv.setWindowProperty("Airdraw", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow("Airdraw", frame)

    # Handle key presses
        stroke = cv.waitKey(1) & 0xff
        if stroke == ord('b'):  # Switch background
            background_mode = "CAM" if background_mode == 'BLACK' else "BLACK"
            detector.background_mode = background_mode
        # elif stroke == ord('m'):  # HME mode
        #     mm.main()
        # elif stroke == ord('s'):
        #     s.main()
        elif stroke == ord('q') or stroke == 27:  # Quit
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()