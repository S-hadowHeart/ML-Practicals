import cv2
import mediapipe as mp
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Finger landmark indexes
finger_tips = [8, 12, 16, 20]
finger_pips = [6, 10, 14, 18]

# Game variables
COUNTDOWN_TIME = 3
RESULT_TIME = 5

countdown_started = False
show_result = False

countdown_start_time = 0
result_start_time = 0

final_gestures = []

left_score = 0
right_score = 0
last_winner = ""

# -------------------- FUNCTIONS --------------------

def detect_gesture(hand_landmarks):
    fingers = []

    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)

    if fingers == [0, 0, 0, 0]:
        return "STONE"
    elif fingers == [1, 1, 1, 1]:
        return "PAPER"
    elif fingers == [1, 1, 0, 0]:
        return "SCISSORS"
    else:
        return "UNDEFINED"


def decide_winner(g1, g2):
    if g1 == g2:
        return "DRAW"

    if (g1 == "STONE" and g2 == "SCISSORS") or \
       (g1 == "SCISSORS" and g2 == "PAPER") or \
       (g1 == "PAPER" and g2 == "STONE"):
        return "LEFT"

    return "RIGHT"


# -------------------- MAIN LOOP --------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)
    gestures = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks)
            gestures.append(gesture)

    # ---------------- START COUNTDOWN ----------------
    if (
        len(gestures) == 2
        and not countdown_started
        and not show_result
        and "UNDEFINED" not in gestures
    ):
        countdown_started = True
        countdown_start_time = time.time()

    # ---------------- COUNTDOWN DISPLAY ----------------
    if countdown_started:
        elapsed = int(time.time() - countdown_start_time)
        remaining = COUNTDOWN_TIME - elapsed

        if remaining > 0:
            cv2.putText(
                frame, str(remaining),
                (300, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 4,
                (0, 0, 255), 6
            )
        else:
            countdown_started = False

            # Lock result ONLY if 2 gestures exist
            if len(gestures) == 2:
                final_gestures = gestures.copy()
                show_result = True
                result_start_time = time.time()

                last_winner = decide_winner(final_gestures[0], final_gestures[1])

                if last_winner == "LEFT":
                    left_score += 1
                elif last_winner == "RIGHT":
                    right_score += 1
            else:
                final_gestures = []
                show_result = False

    # ---------------- RESULT DISPLAY ----------------
    if show_result and len(final_gestures) == 2:
        if time.time() - result_start_time < RESULT_TIME:
            if last_winner == "DRAW":
                text = "DRAW"
            else:
                text = f"{last_winner} HAND WINS"

            cv2.putText(
                frame, text,
                (140, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (255, 0, 0), 4
            )
        else:
            show_result = False
            final_gestures = []
            last_winner = ""

    # ---------------- GESTURE LABELS ----------------
    for i, g in enumerate(gestures):
        label = "LEFT" if i == 0 else "RIGHT"
        cv2.putText(
            frame, f"{label}: {g}",
            (10, 70 + i * 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 255, 0), 2
        )

    # ---------------- SCOREBOARD ----------------
    cv2.rectangle(frame, (0, 0), (640, 50), (40, 40, 40), -1)

    cv2.putText(
        frame, f"LEFT: {left_score}",
        (40, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 1,
        (255, 255, 255), 2
    )

    cv2.putText(
        frame, f"RIGHT: {right_score}",
        (380, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 1,
        (255, 255, 255), 2
    )

    cv2.imshow("Stone Paper Scissors", frame)

    # Exit on 'e'
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
