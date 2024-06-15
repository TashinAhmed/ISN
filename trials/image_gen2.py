import numpy as np
from PIL import Image, ImageDraw
import random

def random_color():
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    return random.choice(colors)

def random_point():
    return random.randint(0, 223), random.randint(0, 223)

def random_size():
    width = random.randint(20, 40)
    height = random.randint(20, 40)
    return width, height

def draw_random_shape(draw):
    shape_type = random.choice(['ellipse', 'rectangle'])
    color = random_color()

    while True:
        x1, y1 = random_point()
        width, height = random_size()
        x2 = x1 + width
        y2 = y1 + height
        if x2 <= 224 and y2 <= 224:
            break
    
    if shape_type == 'ellipse':
        draw.ellipse([x1, y1, x2, y2], fill=color)
    elif shape_type == 'rectangle':
        draw.rectangle([x1, y1, x2, y2], fill=color)

def create_image_with_random_shapes():
    image = Image.new('RGB', (224, 224), 'white')
    draw = ImageDraw.Draw(image)
    
    for _ in range(random.randint(5, 10)):  
        draw_random_shape(draw)

    return image

image = create_image_with_random_shapes()
image.show()
image.save('random_shapes_image.png')
