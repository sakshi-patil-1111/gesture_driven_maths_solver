import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, frame = cap.read()
canvas = np.zeros_like(frame)
prev_x, prev_y = None, None
strokes = []
last_undo_time = 0
undo_interval = 0.5

def draw_buttons(frame):
    cv2.rectangle(frame, (10, 10), (110, 60), (50, 50, 50), -1)
    cv2.putText(frame, 'Clear', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(frame, (130, 10), (230, 60), (50, 50, 50), -1)
    cv2.putText(frame, 'Undo', (140, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def check_button_click(x, y):
    if 10 <= x <= 110 and 10 <= y <= 60:
        return "clear"
    elif 130 <= x <= 230 and 10 <= y <= 60:
        return "undo"
    return None

def redraw_canvas():
    global canvas
    canvas = np.zeros_like(frame)
    for stroke in strokes:
        for i in range(1, len(stroke)):
            cv2.line(canvas, stroke[i - 1], stroke[i], (0, 0, 139), 5)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        draw_buttons(frame)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

            index_up = index_tip.y < index_mcp.y
            middle_up = middle_tip.y < middle_mcp.y

            h, w, c = frame.shape
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)

            button_action = check_button_click(cx, cy)

            if index_up and not middle_up:
                if button_action == "clear":
                    canvas = np.zeros_like(frame)
                    strokes.clear()
                elif button_action == "undo":
                    current_time = time.time()
                    if strokes and (current_time - last_undo_time >= undo_interval):
                        strokes.pop()
                        redraw_canvas()
                        last_undo_time = current_time
                else:
                    if prev_x is None and prev_y is None:
                        prev_x, prev_y = cx, cy
                        strokes.append([(cx, cy)])
                    else:
                        avg_x = int((prev_x + cx) / 2)
                        avg_y = int((prev_y + cy) / 2)
                        cv2.line(canvas, (prev_x, prev_y), (avg_x, avg_y), (0, 0, 139), 5)
                        if strokes:  
                            strokes[-1].append((avg_x, avg_y))
                        prev_x, prev_y = avg_x, avg_y
            else:
                prev_x, prev_y = None, None

            if index_tip.y > 1 or index_tip.x < 0 or index_tip.x > 1:
                prev_x, prev_y = None, None

        frame = cv2.addWeighted(frame, 1, canvas, 0.8, 0)
        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
