import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands and Drawing Utils

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

prevtime = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip and convert the image to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    resultsPose = pose.process(rgb_frame)
    resultsHands = hands.process(rgb_frame)
    resultsFaceMesh = face_mesh.process(rgb_frame)

    # Draw landmarks on the original BGR frame
    if resultsPose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, resultsPose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    if resultsHands.multi_hand_landmarks:
        for hand_landmarks in resultsHands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )

        # List to store 0 (down) or 1 (up) for each finger
        fingers = []
        
        # Thumb (special condition based on horizontal movement)
        if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Other 4 Fingers (vertical comparison)
        tip_ids = [12, 16, 20, 8] # Middle, Ring, Pinky, Index
        for id in tip_ids:
            if hand_landmarks.landmark[id].y < hand_landmarks.landmark[id - 2].y:
                fingers.append(1) # Up
            else:
                fingers.append(0) # Down
                
        total_fingers = fingers.count(1)
        cv2.putText(frame, str(total_fingers), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (75, 3, 89), 3, cv2.LINE_AA)
    
    if resultsFaceMesh.multi_face_landmarks:
        for face_landmarks in resultsFaceMesh.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                face_landmarks, 
                mp_face_mesh.FACEMESH_TESSELATION
            )
    
    # Calculate FPS: 1 / (current_time - previous_time)
    new_time = time.time()
    fps = 1 / (new_time - prevtime)
    prevtime = new_time    

    cv2.putText(frame, 'fps: '+str("{:.2f}".format(fps)), (900, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (75, 3, 89), 3, cv2.LINE_AA)

    cv2.imshow('Hand Tracking', frame)
    
    cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Tracking", 1600, 900)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty("Hand Tracking", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()