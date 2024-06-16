import segmentation_models_pytorch as smp
from src.config import ENCODER, ENCODER_WEIGHTS, CHANNELS, CLASSES, ACTIVATION


def create_unet_model():
    net = smp.UnetPlusPlus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=CHANNELS,
        classes=CLASSES,
        activation=ACTIVATION,
    )
    return net


def main():
    model = create_unet_model()
    print("UNet model created successfully.")


if __name__ == "__main__":
    main()
