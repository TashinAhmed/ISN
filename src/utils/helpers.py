import numpy as np


def reverse_one_hot(image):
    x = np.argmax(image, axis=0)
    return x


def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x


def main():
    print("Helper functions are ready to use.")


if __name__ == "__main__":
    main()
