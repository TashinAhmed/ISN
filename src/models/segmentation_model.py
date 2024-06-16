import pytorch_lightning as pl
import segmentation_models_pytorch as smp

import torch
from torch import nn


class SegmentationModel(pl.LightningModule):
    def __init__(self, net, loss, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.loss = loss
        self.net = net

    def forward(self, x):
        return self.net(x)

    def shared_step(self, preds, labels):
        tp, fp, fn, tn = smp.metrics.get_stats(
            preds, labels.long(), mode="binary", threshold=0.5
        )
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        return {
            "iou_score": iou_score,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "recall": recall,
        }

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        metrics = self.shared_step(preds, labels)
        self.log("train_iou_score", metrics["iou_score"], on_epoch=True)
        self.log("train_f1_score", metrics["f1_score"], on_epoch=True)
        self.log("train_accuracy", metrics["accuracy"], on_epoch=True)
        self.log("train_recall", metrics["recall"], on_epoch=True)
        loss = self.loss(preds, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        metrics = self.shared_step(preds, labels)
        self.log("valid_iou_score", metrics["iou_score"], on_epoch=True)
        self.log("valid_f1_score", metrics["f1_score"], on_epoch=True)
        self.log("valid_accuracy", metrics["accuracy"], on_epoch=True)
        self.log("valid_recall", metrics["recall"], on_epoch=True)
        loss = self.loss(preds, labels)
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        metrics = self.shared_step(preds, labels)
        self.log("test_iou_score", metrics["iou_score"], on_epoch=True)
        self.log("test_f1_score", metrics["f1_score"], on_epoch=True)
        self.log("test_accuracy", metrics["accuracy"], on_epoch=True)
        self.log("test_recall", metrics["recall"], on_epoch=True)
        loss = self.loss(preds, labels)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        return [opt]


def main():
    print("Segmentation model is ready to use.")


if __name__ == "__main__":
    main()
