import os
import pandas as pd

from constants import DATA_DIR, CLASS_DICT_FILE


def load_metadata(data_dir):
    metadata_path = os.path.join(data_dir, "metadata.csv")
    df = pd.read_csv(metadata_path)
    df = df[df["split"] == "experiment"]
    df["image_path"] = df["image_path"].apply(lambda x: os.path.join(data_dir, x))
    df["mask_path"] = df["mask_path"].apply(lambda x: os.path.join(data_dir, x))
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def split_data(df):
    valid_df = df.sample(frac=0.1, random_state=42)
    remaining_df = df.drop(valid_df.index)
    test_df = remaining_df.sample(frac=0.1, random_state=42)
    train_df = remaining_df.drop(test_df.index)
    return train_df, valid_df, test_df


def load_class_info(data_dir):
    class_df = pd.read_csv(CLASS_DICT_FILE)
    class_names = class_df["name"].tolist()
    class_rgb_values = class_df[["r", "g", "b"]].values.tolist()
    return class_names, class_rgb_values


def main():
    data_dir = DATA_DIR
    metadata_df = load_metadata(data_dir)
    train_df, valid_df, test_df = split_data(metadata_df)
    class_names, class_rgb_values = load_class_info(data_dir)
    print(
        f"Train size: {len(train_df)}, Validation size: {len(valid_df)}, Test size: {len(test_df)}"
    )
    print(f"Classes: {class_names}")
    print(f"Class RGB values: {class_rgb_values}")


if __name__ == "__main__":
    main()
