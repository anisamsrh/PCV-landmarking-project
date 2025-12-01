# [BARU] To Calculate Body Rotation (Roll)
import math
def get_body_rotation(p11, p12):
    delta_x = p12.x - p11.x
    delta_y = p12.y - p11.y
    
    # Hitung sudut dalam radian, lalu konversi ke derajat
    angle_rad = math.atan2(delta_y, delta_x)
    angle_deg = math.degrees(angle_rad)
    
    # Normalisasi (karena atan2 hasilnya bisa terbalik tergantung kamera)
    # Biasanya perlu dikurangi offset atau dibalik tanda (+/-)
    return angle_deg 