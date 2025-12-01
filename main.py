import cv2
import mediapipe as mp
import pygame
import numpy as np
import utils.get_dist as get_dist

# --- THRESHOLD ---
HEIGHT_OPEN_THRESH = 0.001  # for mouth
HEIGHT_BIG_THRESH  = 0.05 # for mouth
RATIO_ROUND_THRESH = 0.15 # for mouth
EYE_CLOSE_THRESH = 0.012

# --- INIT SYSTEM ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

pygame.init()
WINDOW_SIZE = (400, 400)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("VTuber")
font = pygame.font.SysFont("Arial", 24)
AVATAR_SIZE = (300, 300)

# --- LOAD ASET ---
assets = {}
background_img = None 
try:
    bg_raw = pygame.image.load("assets/bg.jpg") 
    background_img = pygame.transform.scale(bg_raw, (400, 400))
    assets["IDLE"] = pygame.transform.scale(pygame.image.load("assets/idle.png"), AVATAR_SIZE)
    assets["A"]    = pygame.transform.scale(pygame.image.load("assets/a.png"), AVATAR_SIZE)
    assets["I"]    = pygame.transform.scale(pygame.image.load("assets/e.png"), AVATAR_SIZE) # I and E
    assets["U"]    = pygame.transform.scale(pygame.image.load("assets/u.png"), AVATAR_SIZE) # U and O
    assets["BLINK"] = pygame.transform.scale(pygame.image.load("assets/blink.png"), AVATAR_SIZE)
    assets["RIGHT WINK"] = pygame.transform.scale(pygame.image.load("assets/blink_right.png"), AVATAR_SIZE)
    assets["LEFT WINK"] = pygame.transform.scale(pygame.image.load("assets/blink_left.png"), AVATAR_SIZE)
except Exception as e:
    print("Error:", e)
    print("make sure the assets are there")
    exit()

cap = cv2.VideoCapture(0)

def calculate_state(landmarks):
    # Upper Lip: 13, Lower Lip: 14
    # Left Lip Corner: 61, Right Lip Corner: 291
    # Left Eye Upper and Lower : 159 and 145
    # Right Eye Upper and Lower : 386 and 374

    # --- EYES ---
    left_eye  = get_dist(landmarks[159], landmarks[145])
    right_eye = get_dist(landmarks[386], landmarks[374])

    if left_eye < EYE_CLOSE_THRESH and right_eye < EYE_CLOSE_THRESH:
        return "BLINK", left_eye, 0.0 

    if left_eye < EYE_CLOSE_THRESH:
        return "LEFT WINK", left_eye, 0.0 

    if right_eye < EYE_CLOSE_THRESH:
        return "RIGHT WINK", right_eye, 0.0 
    
    # --- MOUTH ---
    top = landmarks[13]
    bottom = landmarks[14]
    left = landmarks[61]
    right = landmarks[291]

    height = np.sqrt((top.x - bottom.x)**2 + (top.y - bottom.y)**2)
    width = np.sqrt((left.x - right.x)**2 + (left.y - right.y)**2)
    ratio = height / width
    
    if height < HEIGHT_OPEN_THRESH:
        return "IDLE", height, ratio
    
    if height > HEIGHT_BIG_THRESH:
        return "A", height, ratio
    else:
        if ratio < RATIO_ROUND_THRESH:
            return "U", height, ratio
        else:
            return "I", height, ratio

clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ret, frame = cap.read()
    if not ret: break
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    
    state = "IDLE"
    debug_h = 0
    debug_r = 0
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            state, debug_h, debug_r = calculate_state(face_landmarks.landmark)
    
    # --- RENDER ---
    screen.blit(background_img, (0, 0))
    
    # Show image based on the state
    if state in assets:
        screen.blit(assets[state], (50, 50))
    
    # --- DEBUG TEXT ---
    text_state = font.render(f"State: {state}", True, (0, 0, 0))
    text_debug = font.render(f"H: {debug_h:.3f} | R: {debug_r:.2f}", True, (0, 0, 0))
    
    screen.blit(text_state, (10, 10))
    screen.blit(text_debug, (10, 35))
    
    pygame.display.flip()
    clock.tick(30)

cap.release()
pygame.quit()