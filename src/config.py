import torch
DATA_PATH = '../data'
BATCH_SIZE = 16
NUM_WORKER = 0

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES = ["background", "with_mask", "without_mask", "mask_weared_incorrect"]
NUM_CLASSES = len(CLASSES)
NUM_EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5

MODEL_NAME = ["fasterrcnn", "ssd300"]
MODEL_PATH = ['../models/fasterrcnn_finetune.pth', '../models/ssd300_finetune.pth']


IMG_PATH = '../images'
METRICS = ['mAP', 'mAP_50', 'mAP_75', 'IoU_mean', 'Precision', 'Recall', 'F1']
