import numpy as np
import cv2 as cv
from hands import HandDetector
from canvas_math import Canvas
import shapes as s
from PIL import Image
import imutils
import shapes as s
from keras import Sequential
from keras import models
import json
from imutils.contours import sort_contours
import re
import main as m
import math

expression="null"
position = (90, 30) 
position_exp=(350,30) # (x, y) coordinates
font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (0, 255, 0)  # Green
thickness = 2
line_type = cv.LINE_AA

a=0
with open('class.json', 'r') as f:
    class_labels = json.load(f)
# Load the entire model
model = models.load_model('30_epochs.h5')
print(model.summary())
# class_labels = model.class_labels  # Access stored labels

# Prediction function
def prediction(img):
    img = cv.resize(img, (40, 40))
    norm_image = cv.normalize(img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    norm_image = norm_image.reshape((norm_image.shape[0], norm_image.shape[1], 1))
    case = np.asarray([norm_image])

    pred_probs = model.predict(case)
    pred = np.argmax(pred_probs, axis=-1)

    # Get class label from stored class_labels
    predicted_label = class_labels[str(pred[0])]

    return predicted_label, pred_probs

def build_equation(equation):

    eq=''
    for i in equation:
        eq=eq+i
    return(eq)


def compute(chars):
  c=''
  equation=[]
  for item in chars:
    if item[0].isnumeric():
      c=c+item[0]
      # print(c)
    if item[0]=='+' or item[0]=='-' or item[0]=='*' or item[0]=='%' or item[0]=='dot' or item[0]=='sqrt':
      equation.append(c)
      c=''
      equation.append(item[0])
  if c.isnumeric():
    equation.append(c)
  equation=build_equation(equation)
  equation=equation.replace('%','/')
  equation=equation.replace('dot','-')
  result=solve_bodmas(equation)
  return (result)

def solve_bodmas(expressi):
    """
    Solve the given expression using BODMAS rules.

    """
    num=0
    squareroot='sqrt'
    if squareroot in expressi:
      for i in expressi:
        if i in "0123456789":
          num=num*10+int(i)
      return(math.sqrt(num))
    try:
        # Validate the input expression
        # validate_expression(expression)

        # Replace ^ with ** for power calculation in Python
        global expression
        expression = expressi.replace("^", "**")
        if expressi in 'sqrt':
          print (math.sqrt(9))

        # Evaluate the expression
        result = eval(expression)
        return result
    except Exception as e:
        return "Error"


# def solve(chars):
#     c=''
#     equation=[]
#     for item in chars:
#     if item[0].isnumeric():
#         c=c+item[0]
#         # print(c)
#     if item[0]=='+' or item[0]=='-' or item[0]=='*' or item[0]=='%':
#         equation.append(c)
#         c=''
#         equation.append(item[0])
#     if c.isnumeric():
#     equation.append(c)
#     equation=build_equation(equation)
#     equation=equation.replace('%','/')
#     result=solve_bodmas(equation)
#     print(result)
def sort_function(image_path):
    image = cv.imread(image_path)
    chars=[]

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Could not load image from {image_path}. Please check the file path and ensure the image is not corrupted.")
        return

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Apply Gaussian Blur to reduce noise
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Apply Thresholding
    _, thresh = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY_INV)

    # Morphological operation to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv.findContours(cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    contours = sorted(contours, key=lambda c: cv.boundingRect(c)[0])

    # Create a copy of the original image to draw bounding boxes
    image_with_boxes = image.copy()

    # Process each contour
    for i, contour in enumerate(contours):
        x, y, w, h = cv.boundingRect(contour)

        # Draw bounding box
        cv.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract individual elements
        element = image[y:y+h, x:x+w]
        gray = cv.cvtColor(element, cv.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
        blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to create a binary image
        _, thresh = cv.threshold(blurred, 100, 255, cv.THRESH_BINARY_INV)

    # Find all contours
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find the largest connected component (based on area)
        largest_contour = max(contours, key=cv.contourArea)

    # Create an empty mask
        mask = np.zeros_like(gray)

    # Draw only the largest contour onto the mask
        cv.drawContours(mask, [largest_contour], -1, (255), thickness=cv.FILLED)

    # Apply the mask to retain only the largest component
        result = cv.bitwise_and(thresh, mask)

    # Invert the image to make the background white and the diagonal line black
        result = cv.bitwise_not(result)
        chars.append(prediction(result))
        
    global a
    a=compute(chars)
    # print(a)
    print(" Answer : ",a)


def convert_to_grayscale(path):
    image = cv.imread(path)

    # Convert the image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Increase contrast using histogram equalization
    equalized_img = cv.equalizeHist(gray_image)

    # Invert the image
    inverted_img = cv.bitwise_not(equalized_img)

    # Save and display the modified image
    bw_filename = path
    cv.imwrite(bw_filename, inverted_img)

    # Show the image
    bw_image = Image.open(bw_filename)
    sort_function(path)



def main():
    cv.namedWindow("Airdraw", cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty("Airdraw", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    # Loading the default webcam of PC.
    
    cap = cv.VideoCapture(0)
    
    # width and height for 2-D grid
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT) + 0.5)
    ml=150
    max_x, max_y = 250+ml, 50
    mask = np.ones((480, 640))*255
    mask = mask.astype('uint8')
    # set the default background mode (CAM/BLACK)
    background_mode = 'BLACK'
    
    # initialize the canvas element and hand-detector program
    canvas = Canvas(width, height)
    detector = HandDetector(background_mode)

    while True:
        # Reading the frame from the camera
        ret, frame = cap.read()
        frame = cv.flip(frame, 1)

        if background_mode == 'BLACK':
            black_frame = np.ones((height, width, 3), dtype="uint8")*255
            request = detector.determine_gesture(frame, black_frame)
            frame = black_frame
        else:
            request = detector.determine_gesture(frame, frame)

        gesture = request.get('gesture')
        if gesture is not None:
            idx_finger = request['idx_fing_tip']  # coordinates of tip of index finger
            _, c, r = idx_finger
    
            data = {'idx_finger': idx_finger}
            rows, cols, _ = frame.shape

            if 0 < c < cols and 0 < r < rows:
                if gesture == "DRAW":
                    canvas.push_point((r, c))
                elif gesture == "ERASE":
                    canvas.end_line()
                    radius = request['idx_mid_radius']
                    _, mid_r, mid_c = request['mid_fing_tip']
                    canvas.erase_mode((mid_r, mid_c), int(radius * 0.5))
                    data['mid_fing_tip'] = request['mid_fing_tip']
                    data['radius'] = radius
                elif gesture == "HOVER":
                    canvas.end_line()
                elif gesture == "SHAPE_LAUNCH":
                    s.main()
                elif gesture == "MATH_LAUNCH":
                    m.main()
                elif gesture == "MOVE":
                    canvas.end_line()
                    idx_position = (r, c)
                    shift = request['shift']
                    radius = request['idx_pinky_radius']
                    radius = int(radius * 0.8)
                    canvas.translate_mode(idx_position, int(radius * 0.5), shift)
                    data['radius'] = radius
                elif gesture == "SCREENSHOT":
                    screenshot_filename = "screenshot.png"
                    cv.imwrite(screenshot_filename, canvas_image)
                    path="screenshot.png"
                    convert_to_grayscale(path)
            
            frame = canvas.draw_dashboard(frame, gesture, data=data)
        else:
            frame = canvas.draw_dashboard(frame)
            canvas.end_line()
    
        # Draw the stack on the canvas
        canvas_image = np.zeros_like(frame)
        canvas_image = canvas.draw_lines(canvas_image)
        
        # Smoothly blend the canvas with the frame
        op = cv.bitwise_and(canvas_image, canvas_image, mask=mask)

        # Ensure all channels (B, G, R) are updated properly
        # frame[:, :, 0] = op[:, :, 0]  # Blue channel
        frame[:, :, 1] = op[:, :, 1]  # Green channel
        frame[:, :, 2] = op[:, :, 2]  # Red channel

        # Extract the region of interest (ROI) correctly
        roi = frame[:max_y, ml:max_x].copy()  # Ensure a proper copy is taken

        # Ensure the ROI and frame part have the same size before applying addWeighted
        blended = cv.addWeighted(roi, 0.7, roi, 0.3, 0)

        # Place the blended ROI back into the frame
        frame[:max_y, ml:max_x] = blended

        #frame = cv.addWeighted(frame, 0.8, canvas_image, 0.2, 0)
        # cv.putText(frame, f"Mode: {a}", 
        #         (width_border, int(button_height * 2)),
        #         cv.FONT_HERSHEY_SIMPLEX,
        #         2, self.colors[self.color], 3, cv.LINE_AA)
        cv.rectangle(frame, (0, 0), (650, 50), (169, 169, 169), -1)  # Grey color (RGB: 169,169,169)

        cv.putText(frame,f"ANSWER:{a}", position, font, font_scale, color, thickness, line_type)
        cv.putText(frame,f"Equation:{expression}", position_exp, font, font_scale, color, thickness, line_type)
        cv.imshow("Airdraw", frame)
   
        stroke = cv.waitKey(1) & 0xff
        if stroke == ord('b'):  # Switch background
            background_mode = "CAM" if background_mode == 'BLACK' else "BLACK"
            detector.background_mode = background_mode
        elif stroke == ord('q') or stroke == 27:  # Quit
            break
        # elif stroke == ord('p'):  # Press 'p' to take a screenshot
        #     screenshot_filename = "screenshot.png"
        #     cv.imwrite(screenshot_filename, canvas_image)
        #     path="screenshot.png"
        #     convert_to_grayscale(path)
            # print(f"Screenshot saved as {screenshot_filename}")
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
