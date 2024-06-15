import os
import shutil
from tqdm import tqdm
import argparse


class DatasetPreProcessor:
    def __init__(self, data_dir, experiment_dir, holding_dir):
        self.data_dir = data_dir
        self.experiment_dir = experiment_dir
        self.holding_dir = holding_dir
        self.holding_images_dir = os.path.join(holding_dir, "holding_images")
        self.holding_masks_dir = os.path.join(holding_dir, "holding_masks")

    def split_dataset(self):
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(self.holding_images_dir, exist_ok=True)
        os.makedirs(self.holding_masks_dir, exist_ok=True)

        images_dir = os.path.join(self.data_dir, "images")
        masks_dir = os.path.join(self.data_dir, "masks")


        image_files = sorted(os.listdir(images_dir))
        mask_files = sorted(os.listdir(masks_dir))

        assert len(image_files) == len(
            mask_files
        ), "Number of images and masks must be the same."

        total_samples = len(image_files)
        split_point = int(0.9 * total_samples)

        for i in tqdm(range(total_samples)):
            prefix = os.path.splitext(image_files[i])[0].split("_")[
                0
            ]  # Get prefix (e.g., '22')
            image_src = os.path.join(images_dir, image_files[i])
            mask_src = os.path.join(masks_dir, mask_files[i])

            if i < split_point:
                # Copy to experiment directory
                image_dst = os.path.join(self.experiment_dir, f"{prefix}_image.png")
                mask_dst = os.path.join(self.experiment_dir, f"{prefix}_mask.png")
            else:
                # Copy to holding directory
                image_dst = os.path.join(self.holding_images_dir, f"{prefix}_image.png")
                mask_dst = os.path.join(self.holding_masks_dir, f"{prefix}_mask.png")

            shutil.copyfile(image_src, image_dst)
            shutil.copyfile(mask_src, mask_dst)

        print(
            f"Dataset splitting and creation completed at {self.experiment_dir} and {self.holding_dir}."
        )


def main(args):
    splitter = DatasetPreProcessor(args.data_dir, args.experiment_dir, args.holding_dir)
    splitter.split_dataset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset into experiment and holding sets."
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/raw", help="Path to data directory"
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="data/preprocessed/experiment",
        help="Path to experiment directory",
    )
    parser.add_argument(
        "--holding_dir",
        type=str,
        default="data/preprocessed/holding",
        help="Path to holding directory",
    )

    args = parser.parse_args()

    main(args)
