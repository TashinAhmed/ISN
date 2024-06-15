import numpy as np
from PIL import Image, ImageDraw
import random

def random_color():
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    return random.choice(colors)

def random_point():
    return random.randint(0, 223), random.randint(0, 223)

def draw_random_shape(draw):
    shape_type = random.choice(['ellipse', 'rectangle', 'triangle'])
    color = random_color()
    
    if shape_type == 'ellipse':
        x1, y1 = random_point()
        x2, y2 = random_point()
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        draw.ellipse([x1, y1, x2, y2], fill=color)
    
    elif shape_type == 'rectangle':
        x1, y1 = random_point()
        x2, y2 = random_point()
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        draw.rectangle([x1, y1, x2, y2], fill=color)
    
    elif shape_type == 'triangle':
        points = [random_point() for _ in range(3)]
        draw.polygon(points, fill=color)

def create_image_with_random_shapes():
    image = Image.new('RGB', (224, 224), 'white')
    draw = ImageDraw.Draw(image)
    
    for _ in range(random.randint(5, 10)):  
        draw_random_shape(draw)

    return image

image = create_image_with_random_shapes()
image.show()
image.save('random_shapes_image.png')
