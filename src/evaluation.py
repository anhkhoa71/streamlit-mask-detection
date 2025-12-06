import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection.ssd import ssd300_vgg16, VGG16_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def load_model(model_path, device, model_name, num_classes):
    if model_name == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features 
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
    elif model_name == "ssd300":
        # backbone_weights = VGG16_Weights.IMAGENET1K_V1
        model = ssd300_vgg16(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes
        )

    else:
        return None
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    return model

def evaluate_detection(model, dataloader, device, score_thresh=0.5, iou_thresh=0.5):
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")

    all_precisions, all_recalls, all_f1s = [], [], []
    all_ious = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]

            targets = [
                {
                    "boxes": t["boxes"].to(device),
                    "labels": t["labels"].to(device)
                }
                for t in targets
            ]

            outputs = model(images)

            preds = []
            for o in outputs:
                preds.append({
                    "boxes": o["boxes"].detach().cpu().float(),
                    "scores": o["scores"].detach().cpu().float(),
                    "labels": o["labels"].detach().cpu().long()
                })

            gts = []
            for t in targets:
                gts.append({
                    "boxes": t["boxes"].detach().cpu().float(),
                    "labels": t["labels"].detach().cpu().long()
                })
            
            for p, g in zip(preds, gts):
                metric.update([p], [g])

            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes']
                pred_scores = output['scores']
                keep = pred_scores >= score_thresh
                pred_boxes = pred_boxes[keep]

                gt_boxes = target['boxes']

                if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                    ious = box_iou(pred_boxes, gt_boxes)
                    max_iou_per_pred, _ = ious.max(dim=1)
                    all_ious.extend(max_iou_per_pred.cpu().numpy())

                    TP = ((ious >= iou_thresh).sum(dim=1) > 0).sum().item()
                    FP = len(pred_boxes) - TP
                    FN = len(gt_boxes) - ((ious >= iou_thresh).sum(dim=0) > 0).sum().item()
                else:
                    TP = 0
                    FP = len(pred_boxes)
                    FN = len(gt_boxes)

                precision = TP / (TP + FP + 1e-6)
                recall = TP / (TP + FN + 1e-6)
                f1 = 2 * precision * recall / (precision + recall + 1e-6)

                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f1s.append(f1)

    results = metric.compute()

    final_results = {
        "mAP": results['map'].item(),
        "mAP_50": results['map_50'].item(),
        "mAP_75": results['map_75'].item(),
        "IoU_mean": sum(all_ious)/len(all_ious) if all_ious else 0,
        "Precision": sum(all_precisions)/len(all_precisions) if all_precisions else 0,
        "Recall": sum(all_recalls)/len(all_recalls) if all_recalls else 0,
        "F1": sum(all_f1s)/len(all_f1s) if all_f1s else 0
    }

    return final_results

def save_img(metrics, frcnn, ssd, save_path='results.png'):
    x = np.arange(len(metrics))
    width = 0.35
    frcnn_vals = [frcnn[m] for m in metrics]
    ssd_vals   = [ssd[m] for m in metrics]
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, frcnn_vals, width, label='Faster RCNN')
    plt.bar(x + width/2, ssd_vals, width, label='SSD300')

    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title("Comparison of Object Detection Metrics")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)