"""
Carries out hyperparameter tuning using pytorch and GPU if available

NOTE: this also saves the best configs and data to a file in the form of a dictionary, which due to hash tables etc.
does increase file size, however the searched config space would need to be huge to actually make the file size of this
save an issue (as in millions of configs)

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import ParameterGrid, KFold

import pickle


# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, config):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(config['in_channels'], config['first_channels'], kernel_size=config['kernel_size'])
        self.conv2 = nn.Conv2d(config['first_channels'], config['out_channels'], kernel_size=config['kernel_size'])
        self.conv3 = nn.Conv2d(config['out_channels'], config['out_channels_2'], kernel_size=config['kernel_size'])
        self.pool = nn.MaxPool2d(config['pool_size'], config['strides'])
        self.drop = nn.Dropout(p=config['dropout'])

        # Calculate the size of the output tensor after convolutional and pooling layers
        self.conv_out_size = self._get_conv_out_size(config['input_shape'])

        self.fc1 = nn.Linear(self.conv_out_size, config['dense'])
        self.fc2 = nn.Linear(config['dense'], 6)  # Output layer with 6 units for regression

    def _get_conv_out_size(self, input_shape):
        # Forward pass a dummy tensor to calculate the output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  # Add batch dimension
            dummy_output = self.pool(self.conv3(self.pool(self.conv2(self.pool(self.conv1(dummy_input))))))
            return dummy_output.view(dummy_output.size(0), -1).size(1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


# Trains the model
def train_model(config, X_train, y_train, X_val, y_val, best_cost):
    # Move validation data to GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_val, y_val = X_val.to(device), y_val.to(device)

    # Define model
    model = NeuralNetwork(config).to(device)
    model_params = sum(p.numel() for p in model.parameters())

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['mom'], weight_decay=config['weight_decay'])

    # Early stopping parameters
    patience_self = config['patience_self']
    patience_total = config['patience_total']
    epochs = config['epochs']
    tol = config['tolerance_self']
    tol_mult = config['tol_mult']
    cost_func = config['cost_func']

    best_loss = float('inf')
    model_cost = cost_func(float('inf'), model_params)
    early_stop_count = 0

    # Train model
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), config['batch_size']):
            # Load current batch onto GPU
            batch_X, batch_y = X_train[i:i+config['batch_size']].to(device), y_train[i:i+config['batch_size']].to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            # Explicitly delete tensors to free up GPU memory
            del batch_X, batch_y, outputs
            torch.cuda.empty_cache()

        # Validate
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        model_cost = cost_func(val_loss, model_params)  # Work out model cost according to evaluation function

        # Early stopping
        if val_loss < best_loss - tol:
            best_loss = val_loss
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= patience_self:
                # print(f"Early stopping at epoch {epoch+1} due to not self improving")
                break
            elif model_cost > tol_mult * best_cost and early_stop_count >= patience_total:
                # print(f"Early stopping at epoch {epoch + 1} due to being significantly worse than best model")
                break

    # Explicitly delete tensors to free up GPU memory
    del X_val, y_val, val_outputs
    torch.cuda.empty_cache()
    try:
        return best_loss.item(), model_cost, model_params
    except AttributeError:
        return best_loss, model_cost, model_params


def hyperparameter_tuning(config_space, X, y, num_folds=3, filedir='./HyperTuning/LastRun'):
    """
    Carry out hyperparameter tuning - saves the best model to a file every time a new best is found
    :param config_space: Config space to search over
    :param X: Input data (training)
    :param y: Input labels (training)
    :param num_folds: Number of folds to carry out cross-validation for
    :param filedir: directory to save data to
    :return: saves to file filedir
    """

    # When first starting load the save file
    save = setup_save(filedir)
    print("Starting...")
    _, left = load_dt_run(config_space)
    print(f"{len(left)} configs left out of original {len(config_space)}")

    # Set config space to only configs not already tested
    config_space = left

    best_loss = save['best_loss']

    if save['best_config_by_cost']['cost_func'](15, 23) == config_space[0]['cost_func'](15, 23):
        # Using the same cost function so the best previous cost still valid
        best_cost = save['best_cost']
        best_config = save['best_config_by_cost']
    else:
        # Different cost function
        print("Different Cost function being used")
        best_cost = float('inf')
        best_config = None

    # Carry out k-fold cross validation
    kf = KFold(n_splits=num_folds)
    counts = 0
    for count, config in enumerate(config_space):
        save['configs_tested'].append(config)
        counts += 1
        if counts > 10:
            # Save every 10 models tested
            counts = 0
            save_config(save)
        print(f'Config: {count}/{len(config_space)}')
        loss_sum = 0
        model_cost_sum = 0
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            loss, model_cost, model_params = train_model(config, X_train, y_train, X_val, y_val, best_cost)
            loss_sum += loss
            model_cost_sum += model_cost

        mean_loss = loss_sum / num_folds
        mean_cost = model_cost_sum / num_folds

        # SAVE RATHER THAN PRINT
        # print(f"""
        # Config:{config}
        # Mean loss: {mean_loss}
        # Mean cost: {mean_cost}
        # Model Params: {model_params}
        # """)

        if mean_cost < best_cost:
            print("New Best Model")
            best_cost = mean_cost
            best_config = config
            save['best_config_by_cost'] = config
            save['best_cost'] = mean_cost
            save['best_config_loss'] = mean_loss
            save['best_config_by_cost_params'] = model_params
            save_config(save)

        if mean_loss < best_loss:
            best_loss = mean_loss
            save['best_config_by_loss'] = config
            save['best_loss'] = mean_loss
            save['best_config_by_loss_params'] = model_params
            save_config(save)

    return best_config, best_loss, best_cost


def save_config(save, filedir='./HyperTuning/LastRun'):
    """
    Call when run stopped early to save models already tested and the best performing configs
    :return:
    """
    fl = open(filedir, 'wb')
    pickle.dump(save, fl)
    fl.close()


def load_dt_run(config_space, filedir='./HyperTuning/LastRun'):
    """
    Loads the run in filedir and returns list of configs not already tested
    :return: save, configs_left
    """

    fl = open(filedir, 'rb')
    save_load = pickle.load(fl)
    fl.close()

    # GET ONLY UNTESTED CONFIGS - frozenset for efficiency

    # Convert list2 to a set of frozensets
    set2 = {frozenset(d.items()) for d in save_load['configs_tested']}

    # Remove dictionaries from config_space that are contained in tested_configs
    configs_left = [d for d in config_space if frozenset(d.items()) not in set2]

    return save_load, configs_left


def setup_save(filedir='./HyperTuning/LastRun'):
    """
    Sets up the save param and files when running for first time
    :param filedir: file directory of save to load
    :return:
    """
    # Open the file to load previously tested configs
    try:
        fl_r = open(filedir, 'rb')
        save_dt = pickle.load(fl_r)
        save = save_dt
        fl_r.close()
    except (FileNotFoundError, EOFError):
        # File for storage not yet created
        fl_r = open(filedir, 'x')
        fl_r.close()
        fl_r = open(filedir, 'wb')
        save = {
            'best_config_by_cost': None,
            'best_cost': 100,
            'best_config_loss': 100,
            'best_config_by_cost_params': 0,
            'best_config_by_loss': None,
            'best_loss': 100,
            'best_config_by_loss_params': 0,
            'configs_tested': []
        }
        pickle.dump(save, fl_r)
        fl_r.close()

    return save
