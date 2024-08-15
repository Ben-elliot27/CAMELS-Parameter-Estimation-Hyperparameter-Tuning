"""
Carries out training for models on multiple maps via extra channels
Put in an external script just to keep main notebook a little tidier in the main notebook

Also includes analysis functions which are again very similar to the main code block, just moved into separate file for
neatness.

NOTE: This method randomly generates what maps to train on every time it is run, so in the savefile there will be a
mix of lengths of maps trained on which will not always be the same.
"""

from itertools import combinations
from random import choices
import torch
import CAMELS_LOADER
import HyperTuneTorch
import ModelTraining
import pickle
from matplotlib import pyplot as plt
from collections import defaultdict, Counter
import numpy as np

def get_combinations(lst):
    """
    Returns all combinations of a list
    :param lst:
    :return: list
    """
    combination = []
    for i in range(2, len(lst)):
        combination.extend(combinations(lst, i))
    return combination

def run(comb_per_map_comb, max_len_comb, best_config, min=1):
    """
    Carries out the training and testing of multiple map combinations by channels
    :param comb_per_map_comb: How many combinations for each combination length to carry out training on
    :param max_len_comb: Maximum length of map combinations to carry out on
    :param best_config: best config (to build model from)
    :return: Saves data to file 'MultiCrashFallback' (from save_file func)
    """

    # Get all map combinations
    map_combinations = get_combinations(CAMELS_LOADER.map_types)

    # Reduce the amount of combinations to train on
    grouped_map_comb = {}
    for comb in map_combinations:
        length = len(comb)
        if length > max_len_comb:
            break
        if length < min:
            continue
        if length not in grouped_map_comb:
            grouped_map_comb[length] = []
        grouped_map_comb[length].append(comb)
    reduced_map_combinations = []
    for value in grouped_map_comb.values():
        reduced_map_combinations.append(choices(value, k=comb_per_map_comb))

    # New list of map combinations
    map_combinations = [item for sublist in reduced_map_combinations for item in sublist]

    print("Total Length of set: ", len(map_combinations))

    # Run the training loop for all parameter groups -- CHANNELS
    # Setup Save file
    new_dict = setup_save()

    # Remove elements that were added but never worked out due to run being stopped early
    # NOTE: this is just a safety fallback and should never actually run after fixing the saving method
    mult_map_data_CHANNELS = new_dict.copy()
    for item in new_dict.items():
        if type(item[1][0]) is str:
            mult_map_data_CHANNELS.pop(item[0])

    # Remove scenarios already tested
    # Convert list of strings to set for faster lookup
    maps_set = set(map_combinations)

    # Use set difference operation to remove elements also in dictionary keys
    filtered_maps = list(maps_set.difference(mult_map_data_CHANNELS.keys()))

    # Update the dictionary with new keys and default values
    mult_map_data_CHANNELS.update({key: ['trained model', 'test loss', 'history'] for key in filtered_maps})

    print("Length of set not already tested: ", len(filtered_maps))

    try:
        for maps in filtered_maps[::-1]:
            # ----------------------------------- MODEL TRAINING -----------------------------------
            print('Maps:', maps)

            # Set up data
            CAMELS_dataset = CAMELS_LOADER.CAMELS_Dataset([])
            CAMELS_dataset.add_maps(list(maps), verbose=False)
            CAMELS_dataset.generate_dataset(channels=True)
            CAMELS_dataset.normalise(channels=True)
            # Generate the model
            config = best_config

            # Train for longer
            config['epochs'] = 200
            config['batch_size'] = 64
            config['patience_self'] = 20
            config['tolerance_self'] = 0.0001
            config['input_shape'] = (len(maps), 256, 256)
            config['in_channels'] = len(maps)

            model = HyperTuneTorch.NeuralNetwork(config)

            X_train = CAMELS_dataset.train_x
            y_train = CAMELS_dataset.train_y
            X_val = CAMELS_dataset.val_x
            y_val = CAMELS_dataset.val_y
            x_test = CAMELS_dataset.test_x
            y_test = CAMELS_dataset.test_y

            # Convert data from numpy to torch
            X_train = torch.from_numpy(X_train).float()
            y_train = torch.from_numpy(y_train).float()
            X_val = torch.from_numpy(X_val).float()
            y_val = torch.from_numpy(y_val).float()
            x_test = torch.from_numpy(x_test).float()
            y_test = torch.from_numpy(y_test).float()

            # Train the model
            mult_map_data_CHANNELS[maps][2], model = ModelTraining.train_model(config, X_train, y_train, X_val, y_val, model,
                                                                               verbose=False)

            mult_map_data_CHANNELS[maps][0] = model  # Save the model for future use

            # Evaluate loss on training data
            test_loss = ModelTraining.test_model(x_test, y_test, model)
            mult_map_data_CHANNELS[maps][1] = test_loss
            print(f'Model test loss for field(s) {list(CAMELS_dataset.maps.keys())}: {test_loss}')

            # SAVE DATA
            incremental_save(mult_map_data_CHANNELS)
    except KeyboardInterrupt:
        # Run stopped early
        # Remove elements that were added but never worked out due to run being stopped early
        mult_map_data_CHANNELS_n = mult_map_data_CHANNELS.copy()
        for item in mult_map_data_CHANNELS_n.items():
            if type(item[1][0]) is str:
                mult_map_data_CHANNELS.pop(item[0])
        incremental_save(mult_map_data_CHANNELS)

def setup_save(filedir='MultiCrashFallback'):
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
        fl_r = open(filedir, 'wb')
        save = {}
        pickle.dump(save, fl_r)
        fl_r.close()

    return save

def incremental_save(dt, filedir='MultiCrashFallback'):
    """
    Saves dt to file filedir
    :param dt: data to be saved
    :return:
    """
    fl = open(filedir, 'wb')
    pickle.dump(dt, fl)
    fl.close()


def analysis(single_avg_loss, filedir='MultiCrashFallback'):
    """
    Plot graphs and does data analysis for all trained models in filedir
    :param single_avg_loss: average loss for single map training
    :param filedir: file directory for data
    :return: produces graphs
    """

    # --------- DATA PROCESSING ---------
    # Load data
    fl = open(filedir, 'rb')
    mult_map_data_CHANNELS = pickle.load(fl)
    fl.close()

    mult_map_data_CHANNELS = dict(mult_map_data_CHANNELS)

    print(f"Total number of map combinations tested: {len(mult_map_data_CHANNELS.keys())}")

    # Getting the maps with the best loss
    maps_in_loss_order_C = dict(sorted(mult_map_data_CHANNELS.items(), key=lambda x: x[1][1]))

    # Plotting bar graph of map type and test loss
    test_loss_list_20_C = list(maps_in_loss_order_C.values())[:50]

    # Get all map types and their costs
    loss_elements_C = [sublist[1] for sublist in test_loss_list_20_C]
    mult_maps_list_20_C = list(maps_in_loss_order_C.keys())[:50]
    mult_maps_list_20_C = [str(ma)[1:-1].replace("'", "") for ma in mult_maps_list_20_C]

    # Print standard deviation and mean of the data for future statistical analysis
    print(f"Standard deviation of tested model combinations: {np.std(np.array(loss_elements_C))}")
    print(f"Mean of tested model combinations: {np.mean(np.array(loss_elements_C))}")

    grouped_values_C = defaultdict(list)

    for key, value in mult_map_data_CHANNELS.items():
        key_length = len(key)
        grouped_values_C[key_length].append(value[1])

    # Getting average loss for all map number groups
    for key, value in grouped_values_C.items():
        grouped_values_C[key] = np.mean(value)

    flattened_list = ','.join(mult_maps_list_20_C).split(',')
    # Remove spaces generated
    flattened_list = [string.replace(' ', '') for string in flattened_list]

    # Count occurrences
    counted_elements = Counter(flattened_list)

    # --------- GRAPH OF BEST 20 COMBINATIONS ---------
    fig, ax = plt.subplots()
    ax.bar(mult_maps_list_20_C[:20], loss_elements_C[:20])
    plt.xticks(range(len(mult_maps_list_20_C[:20])), mult_maps_list_20_C[:20], rotation='vertical')
    ax.set_title('Multiple maps - Best 20 combinations')
    ax.set_ylabel('Test Loss')
    ax.set_xlabel('Maps trained on')

    # --------- GRAPH OF LOSS VS NUMBER OF MAPS ---------
    fig, ax = plt.subplots()
    ax.plot(list(grouped_values_C.keys()), list(grouped_values_C.values()), 'x')
    ax.plot(1, single_avg_loss, 'x', color='tab:red', label='Single Map')  # SINGLE MAP
    ax.set_xlabel('Number of Maps')
    ax.set_ylabel('Average Loss')
    ax.set_title('Comparing number of maps model trained on and loss -- extra channels')

    # --------- GRAPH OF OCCURRENCES OF MAP TYPE IN BEST MAPS ---------
    fig, ax = plt.subplots()
    ax.bar(counted_elements.keys(), counted_elements.values())
    plt.xticks(range(len(counted_elements.keys())), counted_elements.keys(), rotation='vertical')
    ax.set_xlabel("Map")
    ax.set_ylabel("Occurrences")
    ax.set_title("Number of occurrences of maps in the 50 best map combinations")

    # --------- GRAPH MEAN LOSS ON EACH COMBINATION CONTAINING EACH MAP TYPE ---------
    search_strings = CAMELS_LOADER.map_types
    # Dictionary to store test losses for each search string
    test_losses_dict = {string: [] for string in search_strings}
    # Iterate through the dictionary items
    for key, value in mult_map_data_CHANNELS.items():
        # Check if any of the search strings appear in the key tuple
        for search_string in search_strings:
            if search_string in key:
                # If found, add test loss to the corresponding list
                test_losses_dict[search_string].append(value[1])

    for key, val in test_losses_dict.items():
        test_losses_dict[key] = np.mean(np.array(val))

    fig, ax = plt.subplots()
    ax.bar(test_losses_dict.keys(), test_losses_dict.values())
    plt.xticks(range(len(test_losses_dict.keys())), test_losses_dict.keys(), rotation='vertical')
    ax.set_xlabel("Map")
    ax.set_ylabel("Mean Loss")
    ax.set_title("Mean loss for map combinations containing each map")


def analysis_2_only(filedir='MultiCrashFallback'):
    """
    Carry out data analysis and graph plotting, but this time only for map combinations of length 2
    :param filedir: directory of data
    :return:
    """
    # Get all map combinations
    map_combinations = get_combinations(CAMELS_LOADER.map_types)
    counter = 0
    # Get map combs of len 2
    for comb in map_combinations:
        if len(comb) == 2:
            counter += 1
    print(f"Total combinations of maps containing 2 different map types: {counter}")

    # Load data
    fl = open(filedir, 'rb')
    mult_map_data_CHANNELS = pickle.load(fl)
    fl.close()

    mult_map_data_CHANNELS = dict(mult_map_data_CHANNELS)

    # Get maps combs with only len 2
    only_2_map = {}
    for key, value in mult_map_data_CHANNELS.items():
        if len(key) == 2:
            only_2_map[key] = value
    print(f"Loaded and tested combinations of maps containing 2 different map types: {len(only_2_map.keys())}")

    # Getting the maps with the best loss
    only_2_map_C = dict(sorted(only_2_map.items(), key=lambda x: x[1][1]))

    # Plotting bar graph of map type and test loss
    only_2_map_C = list(only_2_map_C.keys())[:20]
    only_2_map_C = [str(ma)[1:-1].replace("'", "") for ma in only_2_map_C]

    flattened_list = ','.join(only_2_map_C).split(',')
    # Remove spaces generated
    flattened_list = [string.replace(' ', '') for string in flattened_list]

    # Count occurrences
    counted_elements = Counter(flattened_list)

    # --------- GRAPH OF BEST 20 COMBINATIONS ---------
    fig, ax = plt.subplots()
    ax.bar(counted_elements.keys(), counted_elements.values())
    plt.xticks(range(len(counted_elements.keys())), counted_elements.keys(), rotation='vertical')
    ax.set_xlabel("Map")
    ax.set_ylabel("Occurrences")
    ax.set_title("Number of occurrences of maps in the 20 best map combinations")

    search_strings = CAMELS_LOADER.map_types
    # Dictionary to store test losses for each search string
    test_losses_dict = {string: [] for string in search_strings}

    # Iterate through the dictionary items
    for key, value in only_2_map.items():
        # Check if any of the search strings appear in the key tuple
        for search_string in search_strings:
            if search_string in key:
                # If found, add test loss to the corresponding list
                test_losses_dict[search_string].append(value[1])

    # Calculate mean loss for all test losses
    for key, val in test_losses_dict.items():
        test_losses_dict[key] = np.mean(np.array(val))

    # --------- GRAPH MEAN LOSS ON EACH COMBINATION CONTAINING EACH MAP TYPE ---------
    fig, ax = plt.subplots()
    ax.bar(test_losses_dict.keys(), test_losses_dict.values())
    plt.xticks(range(len(test_losses_dict.keys())), test_losses_dict.keys(), rotation='vertical')
    ax.set_xlabel("Map")
    ax.set_ylabel("Mean Loss")
    ax.set_title("Mean loss for map combinations containing each map - 2 map combinations only")
