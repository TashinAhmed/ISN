import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import segmentation_models_pytorch as smp
from torch import nn

from src.models.unet_model import create_unet_model
from src.models.segmentation_model import SegmentationModel
from src.data.datamodule import ISNDataModule
from src.data.data_preprocessing import load_metadata, split_data, load_class_info
from src.config import *


def train():
    metadata_df = load_metadata(DATA_DIR)
    train_df, valid_df, test_df = split_data(metadata_df)
    class_names, class_rgb_values = load_class_info(DATA_DIR)

    net = create_unet_model()
    loss = nn.CrossEntropyLoss()

    early_stop_callback = EarlyStopping(
        monitor="valid_loss", min_delta=0.00001, patience=5, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1, dirpath=OUTPUT_DIR, filename="lightning_trained"
    )
    logger = CSVLogger(OUTPUT_DIR, name="lightning_logs")

    trainer = pl.Trainer(
        accelerator=DEVICE,
        devices=NUM_DEVICES,
        max_epochs=EPOCHS,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
    )

    segmodel = SegmentationModel(net, loss, LR)
    preprocess_input = smp.encoders.get_preprocessing_fn(
        ENCODER, pretrained=ENCODER_WEIGHTS
    )
    isn_data = ISNDataModule(
        train_df,
        valid_df,
        test_df,
        BATCH_SIZE,
        IMG_SIZE,
        preprocess_input,
        class_rgb_values,
    )

    checkpoint_file = os.path.join(CHECKPOINT_DIR, "lightning_trained-v1.ckpt")
    if os.path.isfile(checkpoint_file):
        print("Resuming training from previous checkpoint...")
        trainer.fit(segmodel, datamodule=isn_data, ckpt_path=checkpoint_file)
    else:
        print("Starting training from scratch...")
        trainer.fit(segmodel, datamodule=isn_data)


def main():
    train()


if __name__ == "__main__":
    main()
