import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.transforms as T
from PIL import Image

def inference(image_path, model, device, score_thresh=0.5, input_size=None, class_names=None):
    model.eval()

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    if input_size is not None:
        transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225))
        ])
        img_tensor = transform(img).to(device)
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225))
        ])
        img_tensor = transform(img).to(device)

    with torch.no_grad():
        outputs = model([img_tensor])[0]

    keep = outputs["scores"] >= score_thresh
    boxes = outputs["boxes"][keep].cpu().numpy()
    scores = outputs["scores"][keep].cpu().numpy()
    labels = outputs["labels"][keep].cpu().numpy()

    if input_size is not None:
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * (orig_w / input_size)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * (orig_h / input_size)

    outputs_scaled = {
        "boxes": boxes,
        "scores": scores,
        "labels": labels
    }

    if class_names is None:
        class_names = ["background", "with_mask", "without_mask", "mask_weared_incorrect"]
    show_image_with_boxes(img_tensor, outputs_scaled, class_names)

    return outputs_scaled

def show_image_with_boxes(img_tensor, target, class_names, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):

    img = img_tensor.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.permute(1, 2, 0)
    img = img.clamp(0, 1).cpu().numpy()

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img)

    boxes = target["boxes"]
    labels = target["labels"]
    scores = target["scores"]

    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        label = class_names[labels[i]]
        score = scores[i]
        ax.text(xmin, ymin-5, f"{label} {score*100:.2f}%", color='yellow', fontsize=12,
                bbox=dict(facecolor='red', alpha=0.5, pad=0))

    plt.axis('off')
    plt.show()