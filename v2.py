import cv2
import mediapipe as mp
import pygame
import numpy as np
import math

# --- KONFIGURASI THRESHOLD ---
HEIGHT_OPEN_THRESH = 0.03
HEIGHT_BIG_THRESH  = 0.05
RATIO_ROUND_THRESH = 0.5
EYE_CLOSE_THRESH   = 0.015  # Sesuaikan hasil kalibrasi mata Anda sebelumnya

# Sensitivitas Gerakan Badan
# Semakin kecil nilainya, semakin "nempel" gerakannya
MOVEMENT_SENSITIVITY = 1.5 

# --- INIT MEDIAPIPE HOLISTIC ---
# Kita ganti dari FaceMesh ke Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_face_landmarks=True, # Tetap True agar iris mata terdeteksi
    model_complexity=0) # 0 = Ringan (utk RAM 4GB), 1 = Akurat, 2 = Berat

# --- INIT PYGAME ---
pygame.init()
W, H = 500, 500
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("VTuber: Body + Face")
font = pygame.font.SysFont("Arial", 18)

# --- LOAD ASET ---
assets = {}
try:
    scale = (300, 300)
    assets["IDLE"]  = pygame.transform.scale(pygame.image.load("assets/idle.png"), scale)
    assets["A"]     = pygame.transform.scale(pygame.image.load("assets/mouth_a.png"), scale)
    assets["I"]     = pygame.transform.scale(pygame.image.load("assets/mouth_i.png"), scale)
    assets["U"]     = pygame.transform.scale(pygame.image.load("assets/mouth_u.png"), scale)
    assets["BLINK"] = pygame.transform.scale(pygame.image.load("assets/blink.png"), scale)
except Exception as e:
    print("Error:", e)
    exit()

cap = cv2.VideoCapture(0)

# Fungsi Matematika Jarak
def get_dist(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# [BARU] Fungsi Menghitung Rotasi Badan (Roll)
def get_body_rotation(p11, p12):
    # Hitung selisih koordinat
    delta_x = p12.x - p11.x
    delta_y = p12.y - p11.y
    
    # Hitung sudut dalam radian, lalu konversi ke derajat
    angle_rad = math.atan2(delta_y, delta_x)
    angle_deg = math.degrees(angle_rad)
    
    # Normalisasi (karena atan2 hasilnya bisa terbalik tergantung kamera)
    # Biasanya perlu dikurangi offset atau dibalik tanda (+/-)
    return angle_deg 

# Fungsi Memutar Gambar tanpa mengubah pusat (Pivot Center)
def blit_rotate(surf, image, pos, angle):
    # Rotasi gambar
    rotated_image = pygame.transform.rotate(image, -angle) # Minus agar arahnya natural
    # Ambil rectangle baru setelah rotasi
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = pos).center)
    # Render
    surf.blit(rotated_image, new_rect.topleft)

clock = pygame.time.Clock()
running = True

# Variabel Smoothing (Agar gerakan tidak patah-patah)
smooth_x, smooth_y, smooth_angle = 250, 250, 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ret, frame = cap.read()
    if not ret: break
    
    # Flip frame horizontal agar seperti cermin
    frame = cv2.flip(frame, 1) 
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Proses Holistic
    results = holistic.process(image)
    
    current_img = assets["IDLE"]
    target_x, target_y = 100, 100 # Posisi default (tengah window, dikira-kira)
    target_angle = 0
    
    # --- 1. LOGIKA BADAN (POSE) ---
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Ambil bahu kiri (11) dan kanan (12)
        sh_left = landmarks[11]
        sh_right = landmarks[12]
        
        # Hitung Sudut Miring
        target_angle = get_body_rotation(sh_left, sh_right)
        
        # Hitung Posisi Tengah (Rata-rata hidung/bahu) untuk gerak kiri-kanan
        # Kita pakai hidung (landmark 0) sebagai pusat posisi
        nose = landmarks[0]
        
        # Mapping koordinat normalisasi (0.0 - 1.0) ke Pixel Layar
        # Kita kurangi 0.5 agar titik tengah jadi 0, lalu kali sensitivitas
        center_x = (nose.x - 0.5) * W * MOVEMENT_SENSITIVITY
        center_y = (nose.y - 0.5) * H * MOVEMENT_SENSITIVITY
        
        target_x = 100 + center_x
        target_y = 100 + center_y

    # --- 2. LOGIKA WAJAH (FACE) ---
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

        if avg_eye < EYE_CLOSE_THRESH:
            if "BLINK" in assets: current_img = assets["BLINK"]
        elif mouth_h > HEIGHT_BIG_THRESH:
            current_img = assets["A"]
        elif mouth_h > HEIGHT_OPEN_THRESH:
             current_img = assets["U"] if ratio > RATIO_ROUND_THRESH else assets["I"]
        else:
            current_img = assets["IDLE"]

    # --- 3. SMOOTHING (Interpolasi Linear / LERP) ---
    # Agar gerakan tidak jittery/bergetar
    smooth_x += (target_x - smooth_x) * 0.2
    smooth_y += (target_y - smooth_y) * 0.2
    smooth_angle += (target_angle - smooth_angle) * 0.2

    # --- RENDER ---
    screen.fill((0, 255, 0)) # Green Screen
    
    # Gambar dengan Rotasi
    # Posisi (smooth_x, smooth_y)
    blit_rotate(screen, current_img, (smooth_x, smooth_y), smooth_angle)
    
    # Debug Info
    screen.blit(font.render(f"Angle: {smooth_angle:.1f}", True, (0,0,0)), (10, 10))
    
    pygame.display.flip()
    clock.tick(30) # Limit 30 FPS untuk hemat RAM/CPU

cap.release()
pygame.quit()