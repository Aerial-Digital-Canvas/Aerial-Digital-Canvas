import numpy as np
import cv2 as cv
from hands import HandDetector
from canvas_math import Canvas
from PIL import Image
import json
import matplotlib.pyplot as plt
from keras import models
import main as s
import mainmath as m

expression = "null"
position = (90, 30)
position_exp = (350, 30)
font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 1
color = (0, 255, 0)
thickness = 2
line_type = cv.LINE_AA

with open("label_map.json", "r") as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

# Load model
model = models.load_model('geometric_shapes_recognition3244.h5')
print(model.summary())

def preprocess_image(image):
    img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (224, 224))
    edges = cv.Canny(img, 50, 150)
    img = edges / 255.0
    img = img.reshape(1, 224, 224, 1)
    return img, edges

def predict_shape(image):
    img, edges = preprocess_image(image)
    pred = model.predict(img)
    shape_label = inv_label_map[np.argmax(pred)]
    return shape_label

def main():
    global expression
    
    cv.namedWindow("Airdraw", cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty("Airdraw", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cap = cv.VideoCapture(0)
    ml = 150
    max_x, max_y = 250 + ml, 50
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    mask = np.ones((height, width), dtype="uint8") * 255
    
    background_mode = 'BLACK'
    canvas = Canvas(width, height)
    detector = HandDetector(background_mode)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv.flip(frame, 1)
        black_frame = np.zeros((height, width, 3), dtype="uint8") if background_mode == 'BLACK' else frame.copy()
        request = detector.determine_gesture(frame, black_frame)
        frame = black_frame
        
        gesture = request.get('gesture')
        if gesture:
            idx_finger = request['idx_fing_tip']
            _, c, r = idx_finger
            print(f"Detected Gesture: {gesture} at ({r}, {c})")
            
            if gesture == "DRAW":
                canvas.push_point((r, c))
            elif gesture == "ERASE":
                canvas.end_line()
                radius = request['idx_mid_radius']
                _, mid_r, mid_c = request['mid_fing_tip']
                canvas.erase_mode((mid_r, mid_c), int(radius * 0.5))
            elif gesture == "HOVER":
                canvas.end_line()
            elif gesture == "SHAPE_LAUNCH":
                s.main()
            elif gesture == "MATH_LAUNCH":
                m.main()
            elif gesture == "TRANSLATE":
                canvas.end_line()
                idx_position = (r, c)
                shift = request['shift']
                radius = request['idx_pinky_radius']
                canvas.translate_mode(idx_position, int(radius * 0.5), shift)

        
        frame = canvas.draw_dashboard(frame, gesture, request) if gesture else canvas.draw_dashboard(frame)
        canvas_image = canvas.draw_lines(np.zeros_like(frame))
        # cv.imshow("Canvas Output", canvas_image)
        
        if gesture == "DRAW":
            screenshot = canvas_image.copy()
            expression = predict_shape(screenshot)  # Predict the shape when user draws
            
        cv.putText(frame, f"Shape: {expression}", position, font, font_scale, color, thickness, line_type)
        combined = cv.addWeighted(frame, 0.7, canvas_image, 0.3, 0)
        cv.imshow("Airdraw", combined)


        op = cv.bitwise_and(canvas_image, canvas_image, mask=mask)

        frame[:, :, 1] = op[:, :, 1]  
        frame[:, :, 2] = op[:, :, 2]  

        roi = frame[:max_y, ml:max_x].copy()
        blended = cv.addWeighted(roi, 0.7, roi, 0.3, 0)
        frame[:max_y, ml:max_x] = blended
        
        key = cv.waitKey(1) & 0xff
        if key == ord('b'):
            background_mode = "CAM" if background_mode == "BLACK" else "BLACK"
            detector.background_mode = background_mode
        # elif key == ord('s'):
        #     s.main()
        # elif key == ord('m'):
        #     m.main()
        elif key in [ord('q'), 27]:  # 'q' or ESC to quit
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()