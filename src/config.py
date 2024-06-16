import multiprocessing as mp
import torch


ENCODER = "efficientnet-b2"
ENCODER_WEIGHTS = "imagenet"
CHANNELS = 3
CLASSES = 2  # including background
ACTIVATION = "sigmoid"
LR = 0.001
BATCH_SIZE = 8
IMG_SIZE = (224, 224)
DEVICE, NUM_DEVICES = (
    ("cuda", torch.cuda.device_count())
    if torch.cuda.is_available()
    else ("cpu", mp.cpu_count())
)
WORKERS = mp.cpu_count()
EPOCHS = 3
OUTPUT_DIR = "../working"
DATA_DIR = "../data/preprocessed"
CHECKPOINT_DIR = '../working/savedckpt/'


def main():
    print(f"Running on {NUM_DEVICES} {DEVICE}(s)")


if __name__ == "__main__":
    main()
