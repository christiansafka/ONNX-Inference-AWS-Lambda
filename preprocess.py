import numpy as np
import io
from PIL import Image

def center_crop(image, new_width, new_height):

    left = int(image.size[0]/2-new_width/2)
    upper = int(image.size[1]/2-new_height/2)
    right = left +new_width
    lower = upper + new_height

    return image.crop((left, upper,right,lower))

def preprocess_image(b64image, model_expected_im_size):
    image = Image.open(io.BytesIO(b64image))

    smallest_side = min(image.width, image.height)

    if smallest_side >= model_expected_im_size:
        maxwidth = model_expected_im_size
        maxheight = model_expected_im_size
        i = min(maxwidth/image.width, maxheight/image.height)
        a = max(maxwidth/image.width, maxheight/image.height)
        image.thumbnail((maxwidth*a/i, maxheight*a/i), Image.ANTIALIAS) # Antialias might be slow, can try removing
    else:
        # scale up
        scale_factor = (model_expected_im_size / smallest_side)
        image = image.resize((int(image.width*scale_factor), int(image.height*scale_factor)))

    # Center crop
    image = center_crop(image, model_expected_im_size, model_expected_im_size)
    image_np = np.array(image)

    if image.mode != "RGB":
        if(len(image_np.shape)<3):
            # Grayscale
            rgbimg = Image.new("RGBA", image.size)
            rgbimg.paste(image)
            image_np = np.array(rgbimg)
            image_np = image_np[...,:3]
        else:
            # Other (RGBA)
            image_np = image_np[...,:3]

    # Use to debug crop/scale issues
    #Image.fromarray(image_np.astype(np.uint8)).save("cropped_test_image.png")

    # Normalize for input to efficientNet
    image_np = image_np / 255
    image_np = image_np - [0.485, 0.456, 0.406]
    image_np = image_np / [0.229, 0.224, 0.225]

    # Channels goes first, not last
    image_np = np.moveaxis(image_np, -1, 0)

    # Add batch dimension to the front
    image_np = image_np[np.newaxis, ...]
    return image_np.astype(np.float32)