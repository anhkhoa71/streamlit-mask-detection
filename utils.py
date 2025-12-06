import time
import torch
import torchvision
from huggingface_hub import hf_hub_download
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st
from PIL import Image, ExifTags
import numpy as np
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"

def correct_orientation(img):
    try:
        exif = img._getexif()
        if exif is not None:
            orientation_key = [k for k, v in ExifTags.TAGS.items() if v == "Orientation"][0]
            orientation = exif.get(orientation_key, None)

            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except:
        pass
    return img

def load_fasterrcnn():
    ckpt_path = hf_hub_download(
        repo_id="anhkhoa71/model_lab05_cs406",
        filename="fasterrcnn_finetune.pth"
    )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 4
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    del state_dict  # Giải phóng state_dict ngay
    
    model.to(device).eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Chỉ dùng half precision cho GPU
    if device == "cuda":
        model = model.half()
    
    return model

def load_ssd300():
    ckpt_path = hf_hub_download(
        repo_id="anhkhoa71/model_lab05_cs406",
        filename="ssd300_finetune.pth"
    )

    model = torchvision.models.detection.ssd300_vgg16(weights=None, num_classes=4)
    
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    del state_dict  # Giải phóng state_dict ngay
    
    model.to(device).eval()
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Chỉ dùng half precision cho GPU
    if device == "cuda":
        model = model.half()
    
    return model

COLOR_MAP = {
    "background": "#6366f1",
    "with_mask": "#10b981",
    "without_mask": "#ef4444",
    "mask_weared_incorrect": "#f59e0b"
}

def draw_boxes(img_tensor, outputs, title, class_names):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # Chuyển sang numpy và denormalize
    img = img_tensor.cpu().numpy()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]

    img = np.clip(img.transpose(1, 2, 0), 0, 1)

    # Tạo figure với DPI thấp hơn để giảm RAM
    fig, ax = plt.subplots(1, figsize=(10, 7.5), dpi=80)
    ax.imshow(img)
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#0e0e0e')

    boxes = outputs.get("boxes", [])
    scores = outputs.get("scores", [])
    labels = outputs.get("labels", [])

    if len(boxes) > 0:
        for box, score, label in zip(boxes, scores, labels):
            xmin, ymin, xmax, ymax = box
            w, h = xmax - xmin, ymax - ymin
            class_name = class_names[label]
            color = COLOR_MAP.get(class_name, "#ffffff")

            rect = patches.Rectangle(
                (xmin, ymin), w, h,
                linewidth=3,
                edgecolor=color,
                facecolor='none',
                linestyle='-'
            )
            ax.add_patch(rect)

            ax.text(
                xmin, ymin - 10,
                f"{class_name.replace('_', ' ').title()} {score*100:.1f}%",
                color='white',
                fontsize=11,
                fontweight='bold',
                bbox=dict(
                    facecolor=color,
                    alpha=0.85,
                    edgecolor='white',
                    linewidth=1.5,
                    boxstyle='round,pad=0.5'
                )
            )

    ax.axis("off")
    plt.tight_layout(pad=0)
    
    # Hiển thị và đóng figure ngay
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    
    # Giải phóng memory
    del fig, ax, img
    gc.collect()


def run_inference(image, model, score_thresh=0.5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    img_tensor = transform(image).to(device)
    
    # Chỉ dùng half precision cho GPU
    if device == "cuda":
        img_tensor = img_tensor.half()

    start = time.time()
    with torch.no_grad():
        if device == "cuda":
            with torch.amp.autocast("cuda"):
                output = model([img_tensor])[0]
        else:
            output = model([img_tensor])[0]
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    infer_time = time.time() - start

    keep = output["scores"] >= score_thresh
    outputs = {
        "boxes": output["boxes"][keep].cpu().numpy(),
        "scores": output["scores"][keep].cpu().numpy(),
        "labels": output["labels"][keep].cpu().numpy(),
    }
    
    # Giải phóng output tensor ngay
    del output, keep

    return {
        "img_tensor": img_tensor.cpu().float(),  # Chuyển về CPU ngay
        "outputs": outputs,
        "infer_time": infer_time
    }