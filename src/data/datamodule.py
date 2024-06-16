import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import ISNSet
from src.config import WORKERS


class ISNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        valid_df,
        test_df,
        batch_size=2,
        img_size=(256, 256),
        preprocess_fn=None,
        class_rgb_values=None,
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.img_size = img_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(size=self.img_size),
            ]
        )
        self.preprocess_fn = preprocess_fn
        self.class_rgb_values = class_rgb_values

    def setup(self, stage=None):
        self.trainset = ISNSet(
            self.train_df,
            transform=self.transform,
            preprocess_fn=self.preprocess_fn,
            class_rgb_values=self.class_rgb_values,
        )
        self.validset = ISNSet(
            self.valid_df,
            transform=self.transform,
            preprocess_fn=self.preprocess_fn,
            class_rgb_values=self.class_rgb_values,
        )
        self.testset = ISNSet(
            self.test_df,
            transform=self.transform,
            preprocess_fn=self.preprocess_fn,
            class_rgb_values=self.class_rgb_values,
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=WORKERS,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=WORKERS,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=WORKERS,
            pin_memory=True,
        )


def main():
    print("Data module is ready to use.")


if __name__ == "__main__":
    main()
