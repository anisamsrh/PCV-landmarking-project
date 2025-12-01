import pygame

def blit_rotate(surf, image, pos, angle):
    rotated_image = pygame.transform.rotate(image, -angle - 180)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = pos).center)

    # Render
    surf.blit(rotated_image, new_rect.topleft)