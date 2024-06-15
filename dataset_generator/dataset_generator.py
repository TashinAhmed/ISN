"""Your code has been rated at 9.86/10 (previous run: 8.99/10, +0.87)
    default values:
    img_height and width 224,
    min and max shapes 20-40
    num_images=100,
    image_dir=dataset/images,
    mask_dir=dataset/masks"""

import argparse
import random
import os
from PIL import Image, ImageDraw


class ShapeImageGenerator:
    """5-10 numbers of shape generation for N numbers of
    images and masks generator"""

    def __init__(
        self,
        img_height=224,
        img_width=224,
        min_shape_px=20,
        max_shape_px=40,
        num_images=100,
        image_dir="data/images",
        mask_dir="data/masks",
    ):
        self.img_height = img_height
        self.img_width = img_width
        self.min_shape_px = min_shape_px
        self.max_shape_px = max_shape_px
        self.num_images = num_images
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)

    def random_color(self):
        """Generate a random color from RGB."""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        return random.choice(colors)

    def random_point(self):
        """Generate a random point within the image dimensions."""
        return random.randint(0, self.img_height - 1), random.randint(
            0, self.img_width - 1
        )

    def random_size(self):
        """Generate a random size between min_shape_px and
        max_shape_px pixels for both width and height."""
        width = random.randint(self.min_shape_px, self.max_shape_px)
        height = random.randint(self.min_shape_px, self.max_shape_px)
        return width, height

    def draw_random_shape(self, draw, mask_draw):
        """Draw a random shape (ellipse or rectangle) on the given drawing object.
        Also draw rectangles on the mask drawing object."""
        shape_type = random.choice(["ellipse", "rectangle"])
        color = self.random_color()

        while True:
            x1, y1 = self.random_point()
            width, height = self.random_size()
            x2 = x1 + width
            y2 = y1 + height
            if x2 <= self.img_height and y2 <= self.img_width:
                break

        if shape_type == "ellipse":
            draw.ellipse([x1, y1, x2, y2], fill=color)
        elif shape_type == "rectangle":
            draw.rectangle([x1, y1, x2, y2], fill=color)
            mask_draw.rectangle([x1, y1, x2, y2], fill="white")

    def create_image_and_mask_with_random_shapes(self):
        """Create an image of size img_width x img_width
        with random ellipses and rectangles,
        and a mask for rectangles only."""
        image = Image.new("RGB", (self.img_height, self.img_width), "white")
        draw = ImageDraw.Draw(image)

        mask = Image.new("L", (self.img_height, self.img_width), "black")
        mask_draw = ImageDraw.Draw(mask)

        # generate min 5 shapes and max 10 shapes
        for _ in range(random.randint(5, 10)):
            self.draw_random_shape(draw, mask_draw)

        return image, mask

    def generate_and_save_images_and_masks(self):
        """Generate and save N images and masks
        with specified configurations."""
        for i in range(1, self.num_images + 1):
            image, mask = self.create_image_and_mask_with_random_shapes()
            image_path = os.path.join(self.image_dir, f"{i}_image.png")
            mask_path = os.path.join(self.mask_dir, f"{i}_mask.png")

            image.save(image_path)
            mask.save(mask_path)

            print(f"Saved {image_path} and {mask_path}")


def main():
    """N numberso of image and masks generator main function"""
    parser = argparse.ArgumentParser(
        description="Generate random shape images and masks."
    )

    parser.add_argument(
        "--img_height", type=int, default=224, help="Height of the generated images."
    )
    parser.add_argument(
        "--img_width", type=int, default=224, help="Width of the generated images."
    )
    parser.add_argument(
        "--min_shape_px", type=int, default=20, help="Minimum size of the shapes."
    )
    parser.add_argument(
        "--max_shape_px", type=int, default=40, help="Maximum size of the shapes."
    )
    parser.add_argument(
        "--num_images", type=int, default=100, help="Number of images to generate."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/images",
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="data/masks",
        help="Directory to save generated masks.",
    )

    args = parser.parse_args()

    generator = ShapeImageGenerator(
        img_height=args.img_height,
        img_width=args.img_width,
        min_shape_px=args.min_shape_px,
        max_shape_px=args.max_shape_px,
        num_images=args.num_images,
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
    )

    generator.generate_and_save_images_and_masks()


if __name__ == "__main__":
    main()
