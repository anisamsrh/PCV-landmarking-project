import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Holistic
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Setup Webcam (biasanya index 0 atau 1)
cap = cv2.VideoCapture(0)

# Gunakan model Holistic
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_face_landmarks=True) as holistic: # True agar mendeteksi iris mata
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. Konversi warna BGR (OpenCV) ke RGB (MediaPipe)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False # Optimasi performa
        
        # 2. Proses Deteksi
        results = holistic.process(image)
        
        # 3. Gambar ulang untuk visualisasi
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Gambar Wajah (Mesh)
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
        
        # Gambar Tubuh (Pose)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
        # Gambar Tangan Kiri & Kanan
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.imshow('VTuber Tracking Backend', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()