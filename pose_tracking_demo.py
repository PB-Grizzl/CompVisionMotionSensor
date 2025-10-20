import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Start video capture from webcam
cap = cv2.VideoCapture(0)

print("Starting pose tracking...")
print("Press 'q' to quit")

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from camera")
            break

        # Flip the frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = pose.process(rgb_frame)

        # Draw pose landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Get specific landmarks for arms and head
            landmarks = results.pose_landmarks.landmark

            # Head (nose)
            nose = landmarks[mp_pose.PoseLandmark.NOSE]

            # Left arm
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

            # Right arm
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Display coordinates on screen
            h, w, _ = frame.shape
            cv2.putText(frame, f"Head: ({nose.x:.2f}, {nose.y:.2f})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Left Wrist: ({left_wrist.x:.2f}, {left_wrist.y:.2f})",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Right Wrist: ({right_wrist.x:.2f}, {right_wrist.y:.2f})",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Pose Tracking Demo - Press Q to quit', frame)

        # Exit on 'q' key
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Pose tracking stopped")
