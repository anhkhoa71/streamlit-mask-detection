import random
from glob import glob
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms.functional as F
from torchvision.models.detection.ssd import ssd300_vgg16, VGG16_Weights

class Transforms:
    def __init__(self, subset="train"):
        if subset == "train":
            self.transforms = A.Compose([
                A.Resize(300, 300, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2,
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.9),
                ],p=0.9),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                rotate_limit=10, border_mode=0, p=0.5),
                A.Blur(blur_limit=3, p=0.1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(p=1.0)],
                bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_area=0,
                min_visibility=0.0
                )
            )
        else:
             self.transforms = A.Compose([A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                        ToTensorV2(p=1.0)],
                                        bbox_params=A.BboxParams(
                                        format="pascal_voc",
                                        label_fields=["labels"])
             )
    def __call__(self, image, target):
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        boxes = target["boxes"].clone()
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
        boxes_list = boxes.tolist()
        labels_list = target["labels"].tolist()

        if len(boxes_list) == 0:
            transformed_image = ToTensorV2()(image=image_np)["image"]
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
            return transformed_image, target

        transformed = self.transforms(image=image_np, bboxes=boxes_list, labels=labels_list)

        new_boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
        new_labels = torch.tensor(transformed["labels"], dtype=torch.int64)

        if new_boxes.numel() > 0:
            new_boxes[:, 0::2] = new_boxes[:, 0::2].clamp(0, w)
            new_boxes[:, 1::2] = new_boxes[:, 1::2].clamp(0, h)
            widths = new_boxes[:, 2] - new_boxes[:, 0]
            heights = new_boxes[:, 3] - new_boxes[:, 1]
            valid_mask = (widths > 0) & (heights > 0)
            new_boxes = new_boxes[valid_mask]
            new_labels = new_labels[valid_mask]

        area = (new_boxes[:, 2] - new_boxes[:, 0]) * (new_boxes[:, 3] - new_boxes[:, 1])
        target["boxes"] = new_boxes
        target["labels"] = new_labels
        target["area"] = area
        target["iscrowd"] = torch.zeros((len(new_boxes),), dtype=torch.int64)

        return transformed["image"], target



class DetectionDataset(Dataset):
    def __init__(self, root, classes, subset="train", transforms=None):
        self.root = root
        self.subset = subset
        self.classes_to_id = {class_name: i for i, class_name in enumerate(classes)}
        self.transforms = transforms

        self.xml_files = sorted(glob(f"{root}/{subset}/annotations/*.xml"))
        self.img_files = sorted(glob(f"{root}/{subset}/images/*.png"))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        xml_file = self.xml_files[idx]
        img_file = self.img_files[idx]

        image = Image.open(img_file).convert("RGB")
        tree = ET.parse(xml_file)
        boxes = []
        labels = []

        for obj in tree.findall("object"):
            cls_name = obj.find("name").text
            cls_id = self.classes_to_id[cls_name]

            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(cls_id)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
        else:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": image_id
        }
        original_box_count = len(boxes)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        if original_box_count == 0:
            return image, target

        if len(target["boxes"]) == 0:
            return self.__getitem__(random.randint(0, len(self) - 1))

        return image, target

    @staticmethod
    def collate_fn(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets


def get_dataloaders(
    root="../data",
    classes=["background", "with_mask", "without_mask", "mask_weared_incorrect"],
    batch_size=16,
    num_worker=4
):
    transforms = {s: Transforms(subset=s) for s in ["train", "valid"]}

    dataset = {
        s: DetectionDataset(
            root=root,
            classes=classes,
            subset=s,
            transforms=transforms[s]
        )
        for s in ["train", "valid"]
    }

    loader = {
        s: DataLoader(
            dataset=dataset[s],
            batch_size=batch_size,
            num_workers=num_worker,
            collate_fn=dataset[s].collate_fn
        )
        for s in ["train", "valid"]
    }
    return dataset, loader