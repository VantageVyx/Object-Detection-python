import cv2
import time
import serial
from ultralytics import YOLO

#connecting to the arduino
try:
    arduino = serial.Serial('COM3', 9600)  #change COM port if needed
    time.sleep(2)
    print("Arduino connected.")
except Exception as e:
    print("Failed to connect to Arduino:", e)
    arduino = None

#loading yolo11 model
model = YOLO("yolo11n.pt")

#set you detection target!
TARGET_LABEL = "cup"
led_state = False  #LED should start as off
MIN_ON_TIME = 1  #set a delay (seconds)
last_on_time = 0

#turn on webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Running YOLO real-time cup detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()

    found_cup = False
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = results[0].names[cls_id]

            print(f"Detected: {label} ({conf:.2f})")

            if label.lower() == TARGET_LABEL and conf > 0.5:
                found_cup = True
                break  #cup detected

    #LED ON/OFF logic
    current_time = time.time()

    if arduino:
        if found_cup:
            if not led_state:
                arduino.write(b'1')
                led_state = True
                last_on_time = current_time
                print("☕ FOUND A CUP! Turning LED ON.")
            else:
                last_on_time = current_time  #refresh timer if still detecting
        elif led_state and (current_time - last_on_time >= MIN_ON_TIME):
            arduino.write(b'0')
            led_state = False
            print("🚫 Cup not found for a while. Turning LED OFF.")

    cv2.imshow("YOLO Cup Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#cleaning up everything before shutting off
cap.release()
if arduino:
    arduino.write(b'0')  #make sure LED is off on exit
    arduino.close()
cv2.destroyAllWindows()
