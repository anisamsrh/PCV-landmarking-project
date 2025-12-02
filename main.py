import cv2
import mediapipe as mp
import pygame
import numpy as np
import utils.get_dist as get_dist
import utils.get_body_rotation as get_body_rotation
import utils.blit_rotate as blit_rotate
import utils.get_angle_diff as get_angle_diff

# --- THRESHOLD ---
HEIGHT_OPEN_THRESH = 0.001  # for mouth
HEIGHT_BIG_THRESH  = 0.05 # for mouth
RATIO_ROUND_THRESH = 0.15 # for mouth
EYE_CLOSE_THRESH = 0.012
MOVEMENT_SENSITIVITY = 1.0

# --- INIT SYSTEM ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_face_landmarks=True, # Tetap True agar iris mata terdeteksi
    model_complexity=0)

pygame.init()
WINDOW_SIZE = (500, 500)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("VTuber")
font = pygame.font.SysFont("Arial", 24)
AVATAR_SIZE = (1200, 1200)
OFFSET = (-300,-200)

# --- LOAD ASET ---
assets = {}
background_img = None 
try:
    bg_raw = pygame.image.load("assets/bg.jpg") 
    background_img = pygame.transform.scale(bg_raw, WINDOW_SIZE)
    assets["IDLE EYES"] = pygame.transform.scale(pygame.image.load("assets/v3/v3_idle.png"), AVATAR_SIZE)
    assets["IDLE MOUTH"] = pygame.transform.scale(pygame.image.load("assets/v3/v3_mouth_idle.png"), AVATAR_SIZE)
    assets["A"]    = pygame.transform.scale(pygame.image.load("assets/v3/v3_mouth_a.png"), AVATAR_SIZE)
    assets["I"]    = pygame.transform.scale(pygame.image.load("assets/v3/v3_mouth_e.png"), AVATAR_SIZE) # I and E
    assets["U"]    = pygame.transform.scale(pygame.image.load("assets/v3/v3_mouth_u.png"), AVATAR_SIZE) # U and O
    assets["BLINK"] = pygame.transform.scale(pygame.image.load("assets/v3/v3_blink.png"), AVATAR_SIZE)
    assets["RIGHT WINK"] = pygame.transform.scale(pygame.image.load("assets/v3/v3_wink_right.png"), AVATAR_SIZE)
    assets["LEFT WINK"] = pygame.transform.scale(pygame.image.load("assets/v3/v3_wink_left.png"), AVATAR_SIZE)
except Exception as e:
    print("Error:", e)
    print("make sure the assets are there")
    exit()

cap = cv2.VideoCapture(0)

clock = pygame.time.Clock()
running = True

smooth_x, smooth_y, smooth_angle = 250, 250, 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1) 
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Draw Landmark
    # mp_drawing.draw_landmarks(
    #     image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
    #     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
    #     mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    # )
    # mp_drawing.draw_landmarks(
    #     image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # cv2.imshow('VTuber Tracking Landmark', image)
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     break

    current_img = assets["IDLE EYES"]
    cureent_mouth = assets["IDLE MOUTH"]
    target_x, target_y = OFFSET[0], OFFSET[1] # Default Position
    target_angle = 0
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # shoulder left and right
        sh_left = landmarks[11]
        sh_right = landmarks[12]
        target_angle = get_body_rotation(sh_left, sh_right)

        nose = landmarks[0]
        center_x = (nose.x - 0.5) * WINDOW_SIZE[0] * MOVEMENT_SENSITIVITY # x H
        center_y = (nose.y - 0.5) * WINDOW_SIZE[1] * MOVEMENT_SENSITIVITY # x W
        
        target_x = center_x + OFFSET[0]
        target_y = center_y + OFFSET[1]
    
    if results.face_landmarks:
        # Sama seperti kode sebelumnya, logika deteksi mulut & mata
        landmarks = results.face_landmarks.landmark
        
        # Mata (Indeks sama dengan kode sebelumnya)
        left_eye  = get_dist(landmarks[159], landmarks[145])
        right_eye = get_dist(landmarks[386], landmarks[374])
        avg_eye = (left_eye + right_eye) / 2
        
        # Mulut
        mouth_h = get_dist(landmarks[13], landmarks[14])
        mouth_w = get_dist(landmarks[61], landmarks[291])
        ratio = mouth_h / mouth_w

        if left_eye < EYE_CLOSE_THRESH and right_eye < EYE_CLOSE_THRESH:
            current_img = assets["BLINK"]
        elif left_eye < EYE_CLOSE_THRESH:
            current_img = assets["LEFT WINK"]
        elif right_eye < EYE_CLOSE_THRESH:
            current_img = assets["RIGHT WINK"]
        else :
            current_img = assets["IDLE EYES"]

        if mouth_h > HEIGHT_BIG_THRESH:
            current_mouth = assets["A"]
        elif mouth_h > HEIGHT_OPEN_THRESH:
             current_mouth = assets["I"] if ratio > RATIO_ROUND_THRESH else assets["U"]
        else:
            current_mouth = assets["IDLE MOUTH"]

    # For smooth movement of the body
    smooth_x += (target_x - smooth_x) * 0.2
    smooth_y += (target_y - smooth_y) * 0.2
    diff = get_angle_diff(target_angle, smooth_angle)
    smooth_angle += diff * 0.5
    
    # --- RENDER ---
    screen.blit(background_img, (0, 0))
    
    # Show image based on the state
    blit_rotate(screen, current_img, (smooth_x, smooth_y), smooth_angle)
    blit_rotate(screen, current_mouth, (smooth_x, smooth_y), smooth_angle)
    
    pygame.display.flip()
    clock.tick(30)

cap.release()
pygame.quit()