import os
import cv2
from skimage import io


def green_channel(image):
    retina = io.imread(image)
    retina_green = retina.copy()

    green = retina_green[:, :, (0, 2)] = 0
    retina_green[green] = [0, 255, 0]
    cv2.imwrite('greenImage.png')
