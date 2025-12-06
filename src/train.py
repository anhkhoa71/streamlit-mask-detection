from tqdm import tqdm
import torch

def train_model(model, dataloader_dict, dataset_dict, optimizer, num_epochs, device, save_path='ssd300_finetune.pth'):
    Loss_train, Loss_val = [], []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n🔹 Epoch {epoch+1}/{num_epochs}")

        for phase in ['train', 'valid']:
            model.train()
            running_loss = 0.0
            total_samples = len(dataset_dict[phase])
            loop = tqdm(dataloader_dict[phase], desc=f"{phase.capitalize()} phase", leave=False)

            for images, targets in loop:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                if phase == 'train':
                    optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    loss_dict = model(images, targets)
                    loss = sum(loss for loss in loss_dict.values())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * len(images)
                loop.set_postfix({'loss': f'{loss.item():.4f}'})

            epoch_loss = running_loss / total_samples

            if phase == 'train':
                Loss_train.append(epoch_loss)
            else:
                Loss_val.append(epoch_loss)
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    torch.save(model.state_dict(), save_path)
                    print(f"💾 Saved best model (val loss: {best_val_loss:.4f}) to '{save_path}'")

        print(f"✅ Epoch {epoch+1}/{num_epochs} -- "
              f"Train Loss: {Loss_train[-1]:.4f} | Val Loss: {Loss_val[-1]:.4f}")

    print(f"\n🎯 Training complete. Best Val Loss: {best_val_loss:.4f}")
    return Loss_train, Loss_val