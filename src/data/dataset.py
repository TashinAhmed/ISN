import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


class ISNSet(Dataset):
    def __init__(self, df, transform=None, preprocess_fn=None, class_rgb_values=None):
        self.image_paths = df["image_path"].tolist()
        self.mask_paths = df["mask_path"].tolist()
        self.transform = transform
        self.preprocess_fn = preprocess_fn
        self.class_rgb_values = class_rgb_values

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_file = self.image_paths[idx]
        mask_file = self.mask_paths[idx]

        img = Image.open(img_file).convert("RGB")
        img = np.array(img)

        mask = Image.open(mask_file).convert("RGB")
        mask = one_hot_encode(mask, self.class_rgb_values).astype("float")

        if self.preprocess_fn:
            img = self.preprocess_fn(img)
            img = np.array(img, dtype=np.float32)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask


def main():
    print("Dataset class is ready to use.")


if __name__ == "__main__":
    main()
