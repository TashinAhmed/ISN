import os

DATA_DIR = "../data/preprocessed"
CLASS_DICT_FILE = os.path.join(DATA_DIR, "class_dict.csv")


def main():
    print(f"Data directory: {DATA_DIR}")
    print(f"Class dictionary file: {CLASS_DICT_FILE}")


if __name__ == "__main__":
    main()
