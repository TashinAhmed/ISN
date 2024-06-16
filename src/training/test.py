import os

import segmentation_models_pytorch as smp
from torch import nn

import pytorch_lightning as pl
from src.models.unet_model import create_unet_model
from src.models.segmentation_model import SegmentationModel
from src.data.datamodule import ISNDataModule
from src.data.data_preprocessing import load_metadata, split_data, load_class_info
from src.config import *
from src.utils.checkpoint_utils import get_latest_checkpoint


def test():
    metadata_df = load_metadata(DATA_DIR)
    _, _, test_df = split_data(metadata_df)
    class_names, class_rgb_values = load_class_info(DATA_DIR)

    net = create_unet_model()
    loss = nn.CrossEntropyLoss()
    preprocess_input = smp.encoders.get_preprocessing_fn(
        ENCODER, pretrained=ENCODER_WEIGHTS
    )
    isn_data = ISNDataModule(
        train_df=None,
        valid_df=None,
        test_df=test_df,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        preprocess_fn=preprocess_input,
        class_rgb_values=class_rgb_values,
    )

    # segmodel = SegmentationModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
    latest_checkpoint = get_latest_checkpoint(CHECKPOINT_DIR, "lightning_trained")
    if latest_checkpoint:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
        print(f"Loading checkpoint: {checkpoint_path}")
        segmodel = SegmentationModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path
        )
    else:
        raise FileNotFoundError("No checkpoint file found.")

    trainer = pl.Trainer(accelerator=DEVICE, devices=NUM_DEVICES)
    trainer.test(segmodel, datamodule=isn_data)


def main():
    test()


if __name__ == "__main__":
    main()
