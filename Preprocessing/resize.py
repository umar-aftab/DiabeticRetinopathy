import os
from PIL import Image


def resize(image_path):
    im = Image.open(image_path)
    original_size = im.size

    # Want the new image to be 1300 x 1000
    desired_ratio = 1300 / 1000
    original_ratio = original_size[0] / original_size[1]

    # if the aspect ratio is > 1.3 the width of the image is the restricting variable
    if original_ratio >= desired_ratio:
        ratio = float(1300) / max(original_size)
    # if the aspect ratio is < 1.3 the height of the image is the restricting variable
    else:
        ratio = float(1000) / min(original_size)

    # resize image to new dimensions while preserving original aspect ratio
    new_size = tuple([int(x * ratio) for x in original_size])
    im = im.resize(new_size, Image.ANTIALIAS)

    # Create black image 1300 x 1000
    new_image = Image.new("RGB", (1300, 1000))

    # Paste resized image onto black image
    new_image.paste(im, ((1300 - new_size[0]) // 2,
                         (1000 - new_size[1]) // 2))

    new_image.save('newImage.png')
