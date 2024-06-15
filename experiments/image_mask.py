"""Image and corresponding mask generation"""

import random
from PIL import Image, ImageDraw


IMG_HEIGHT = 224
IMG_WIDTH = 224
MIN_SHAPE_PX = 20
MAX_SHAPE_PX = 40

def random_color():
    """Generate a random color from RGB."""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    return random.choice(colors)

def random_point():
    """Generate a random point within the image dimensions."""
    return random.randint(0, IMG_HEIGHT-1), random.randint(0, IMG_WIDTH-1)

def random_size():
    """Generate a random size between min_shape_px and 
    max_shape_px pixels for both width and height."""
    width = random.randint(MIN_SHAPE_PX, MAX_SHAPE_PX)
    height = random.randint(MIN_SHAPE_PX, MAX_SHAPE_PX)
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
        if x2 <= IMG_HEIGHT and y2 <= IMG_WIDTH:
            break

    if shape_type == 'ellipse':
        draw.ellipse([x1, y1, x2, y2], fill=color)
    elif shape_type == 'rectangle':
        draw.rectangle([x1, y1, x2, y2], fill=color)
        mask_draw.rectangle([x1, y1, x2, y2], fill='white')

def create_image_and_mask_with_random_shapes():
    """Create an image of size img_width x img_width with random ellipses and rectangles,
       and a mask for rectangles only."""
        
    image = Image.new('RGB', (IMG_HEIGHT, IMG_WIDTH), 'white')
    draw = ImageDraw.Draw(image)
    
    mask = Image.new('L', (IMG_HEIGHT, IMG_WIDTH), 'black')
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
