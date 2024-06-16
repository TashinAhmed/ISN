import segmentation_models_pytorch as smp


def compute_metrics(preds, labels):
    tp, fp, fn, tn = smp.metrics.get_stats(
        preds, labels.long(), mode="binary", threshold=0.5
    )
    metrics = {
        "iou_score": smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro"),
        "f1_score": smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro"),
        "accuracy": smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro"),
        "recall": smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise"),
    }
    return metrics


def main():
    print("Metrics utilities are ready to use.")


if __name__ == "__main__":
    main()
