"""
A script to carry out full training of the model -- uses similar training functions to HyperTuneTorch
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler


def train_model(config, X_train, y_train, X_val, y_val, model, verbose=True, transform=lambda x: x):
    """
    Train the model for longer periods using predefined model
    :param config: config dict
    :param X_train, y_train: TORCH TENSOR training data
    :param X_val, y_val: TORCH TENSOR validation data
    :param model: model to be used
    :param verbose: whether to print data during training
    :param transform: PyTorch transform for data augmentation -- default (no augmentation) function which returns arg
    :return: history - keys: ['best_val_loss'], ['train_loss_list'], ['val_loss_list']
    """
    # Move validation data to GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_val, y_val = X_val.to(device), y_val.to(device)
    if X_train.size()[1] != 1:
        # Multiple channels
        X_train = X_train.permute(0, 3, 1, 2)  # Assuming X_train has shape (num_samples, height, width, num_channels)
        X_val = X_val.permute(0, 3, 1, 2)  # Assuming X_val has shape (num_samples, height, width, num_channels)
    history = {
        'best_val_loss': float('inf'),
        'train_loss_list': [],
        'val_loss_list': []
    }

    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['mom'])

    # Early stopping parameters
    patience_self = config['patience_self']
    epochs = config['epochs']
    tol = config['tolerance_self']

    best_loss = float('inf')
    early_stop_count = 0

    scaler = GradScaler()  # Less precision for reduced memory use

    # Train model
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), config['batch_size']):
            batch_loss = []
            # Load current batch onto GPU
            batch_X, batch_y = transform(X_train[i:i + config['batch_size']]).to(device), y_train[i:i + config['batch_size']].to(device)
            optimizer.zero_grad()
            with autocast():
                output = model(batch_X)
                loss = criterion(output, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_loss.append(loss.cpu().item())

            # Explicitly delete tensors to free up GPU memory
            del batch_X, batch_y, output
            torch.cuda.empty_cache()

        batch_loss_mean = sum(batch_loss)/len(batch_loss)
        history['train_loss_list'].append(batch_loss_mean)
        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            history['val_loss_list'].append(val_loss.item())

        # Early stopping
        if val_loss < best_loss - tol:
            best_loss = val_loss
            early_stop_count = 0
            history['best_val_loss'] = best_loss.cpu().detach().item()

            # Checkpoint the model
            torch.save({'model': model, 'epoch': epoch, 'loss':val_loss}, 'TorchCheckpoints.pt')

        else:
            early_stop_count += 1
            if early_stop_count >= patience_self:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1} due to not self improving")
                # Load the best checkpoint
                saved = torch.load('TorchCheckpoints.pt')
                print(f"Model loaded from epoch {saved['epoch']} with loss -- {saved['loss']}")
                model = saved['model']
                break

        if verbose:
            print(
                f'Epoch {epoch}/{epochs} -- Train Loss {loss} -- Val Loss -- {val_loss} -- Best Val Loss -- {best_loss}')

    # Explicitly delete tensors to free up GPU memory
    del X_val, y_val, val_outputs
    torch.cuda.empty_cache()

    return history, model


def test_model(x_test, y_test, model, criterion=nn.MSELoss()):
    """
    Test the model on test data
    :param x_test: TORCH TENSOR test data
    :param y_test: TORCH TENSOR test data labels
    :param criterion: Criterion function to use for 'error' default is MSE
    :return: loss
    """
    # Check if GPU available
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.empty_cache()
    else:
        device = 'cpu'

    if x_test.size()[1] != 1:
        # Multiple channels -- need to reorder for torch model
        x_test = x_test.permute(0, 3, 1, 2)  # Assuming X_train has shape (num_samples, height, width, num_channels)

    x_test, y_test = x_test.to(device), y_test.to(device)
    model.to(device)

    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test)
        test_loss = criterion(test_outputs, y_test)
    try:
        return test_loss.item()
    except RuntimeError:
        return test_loss.detach().cpu().numpy()


