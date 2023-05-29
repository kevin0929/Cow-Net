import torch
import torch.nn as nn

from utils.dataloder import CowDataset
from classify import CowNet
from torch.utils.data import Dataset, DataLoader


def train(
    model,
    lossfunc,
    optimizer,
    dataloader,
    device,
    num_epochs: int = 100,
) -> None:
    """
    To training cow-net
    """

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")

        loss_value = 0.0
        corrects = 0

        # Iterate dataset
        for inputs, features, labels in dataloader:
            inputs = inputs.to(device)
            features = features.to(device)
            labels = labels.to(device)

            # initial gradient parameter
            optimizer.zero_grad()

            # forwarding
            outputs = model(features, inputs)
            _, preds = torch.max(outputs, 1)
            loss = lossfunc(outputs, labels)

            # backward propagation + optimizer
            loss.backward()
            optimizer.step()

            loss_value = loss_value + loss.item() * input.size(0)
            corrects = corrects + torch.sum(preds == labels.data)

        epoch_loss = loss_value / len(dataloader.dataset)
        epoch_acc = corrects.double() / len(dataloader.dataset)

        print(f"Loss : {epoch_loss:.4f}, ACC : {epoch_acc:.4f}")

    return model


def main():
    """ initialize model / dataset config """

    # model config
    model = CowNet()
    lossfunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # dataset config
    train_image_path = "./data/train/image"
    train_label_path = "./data/train/labels"

    dataset = CowDataset(train_image_path, train_label_path)
    dataloader = DataLoader(dataset, batchsize=16, shuffle=True)

    # train config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # starting training
    model = train(
        model=model,
        lossfunc=lossfunc,
        optimizer=optimizer,
        dataloader=dataloader,
        num_epochs=100,
    )
