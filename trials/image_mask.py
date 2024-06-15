import numpy as np
from PIL import Image, ImageDraw
import random

img_height = 224
img_width = 224
min_shape_px = 20
max_shape_px = 40

def random_color():
    """Generate a random color from RGB."""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    return random.choice(colors)

def random_point():
    """Generate a random point within the image dimensions."""
    return random.randint(0, img_height-1), random.randint(0, img_width-1)

def random_size():
    """Generate a random size between min_shape_px and max_shape_px pixels for both width and height."""
    width = random.randint(min_shape_px, max_shape_px)
    height = random.randint(min_shape_px, max_shape_px)
    return width, height

def draw_random_shape(draw, mask_draw):
    """Draw a random shape (ellipse or rectangle) on the given drawing object.
       Also draw rectangles on the mask drawing object."""
    shape_type = random.choice(['ellipse', 'rectangle'])
    color = random_color()

    while True:
        x1, y1 = random_point()
        width, height = random_size()
        x2 = x1 + width
        y2 = y1 + height
        if x2 <= img_height and y2 <= img_width:
            break

    if shape_type == 'ellipse':
        draw.ellipse([x1, y1, x2, y2], fill=color)
    elif shape_type == 'rectangle':
        draw.rectangle([x1, y1, x2, y2], fill=color)
        mask_draw.rectangle([x1, y1, x2, y2], fill='white')

def create_image_and_mask_with_random_shapes():
    """Create an image of size img_width x img_width with random ellipses and rectangles,
       and a mask for rectangles only."""
    
    image = Image.new('RGB', (img_height, img_width), 'white')
    draw = ImageDraw.Draw(image)
    
    mask = Image.new('L', (img_height, img_width), 'black')
    mask_draw = ImageDraw.Draw(mask)

    # min 5 max 10 shapes     
    for _ in range(random.randint(5, 10)):  
        draw_random_shape(draw, mask_draw)

    return image, mask


image, mask = create_image_and_mask_with_random_shapes()
# image.show()
# mask.show()
image.save('random_shapes_image.png')
mask.save('rectangle_mask.png')
