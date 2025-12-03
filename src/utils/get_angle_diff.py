def get_angle_diff(target, current):
    diff = target - current

    diff = (diff + 90) % 360 - 90
    
    return diff