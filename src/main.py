import config
from dataset import get_dataloaders
from evaluation import load_model, evaluate_detection, save_img

print("Loading dataloaders...")
_, loader = get_dataloaders(
    root=config.DATA_PATH,
    classes=config.CLASSES,
    batch_size=config.BATCH_SIZE,
    num_worker=config.NUM_WORKER
)

print("Loading models...")
model = {}
for name, path in zip(config.MODEL_NAME, config.MODEL_PATH):
    print(f"Loading model: {name}")
    model[name] = load_model(path, config.DEVICE, name, config.NUM_CLASSES)

print("Running evaluation...")
result = {}
for name in model.keys():
    print(f"Evaluating: {name}")
    result[name] = evaluate_detection(
        model[name],
        loader['valid'],
        config.DEVICE,
        config.SCORE_THRESHOLD,
        config.IOU_THRESHOLD
    )
    print(f"Result: {result[name]}")

print("Saving result images...")
save_img(
    config.METRICS,
    result[config.MODEL_NAME[0]],
    result[config.MODEL_NAME[1]],
    f'{config.IMG_PATH}/results.png'
)

print("Done.")
