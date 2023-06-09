from tqdm import tqdm
import torch


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def train_step(model, data_loader, optimizer, loss_fn, device):
    model.train()
    train_loss = 0

    for (images, _) in tqdm(data_loader):
        images = images.to(device)
        # Forward pass
        encoded, z_mean, z_log_var, decoded = model(images)

        # Calculate loss
        loss = loss_fn(images, z_mean, z_log_var, decoded, weight=1)
        train_loss += loss.item()

        # Zero grad
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # update parameters
        optimizer.step()

    train_loss /= len(data_loader)

    return train_loss


def train(model, train_dataloader, optimizer, loss_fn, epochs, device, save_model=None):
    results = {"train_loss": []}

    model.to(device)
    for epoch in range(epochs):
        train_loss = train_step(model, train_dataloader, optimizer, loss_fn, device)
        results["train_loss"].append(train_loss)

        print(f"EPOCH: {epoch:} | train_loss: {train_loss: .4f}")

        if save_model:
            checkpoint = {"state_dict": model.state_dict(),
                          "optimizer": optimizer.state_dict(),
                          "epoch": epoch}

            save_checkpoint(checkpoint, save_model)

    return results