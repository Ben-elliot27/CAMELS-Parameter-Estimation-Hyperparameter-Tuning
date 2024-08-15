"""
A class to load data from the CAMELS dataset 1P variant.

NOTE: Remember to call method generate_dataset after every change to values within the dataset
"""


import urllib.request
from urllib.error import HTTPError
import os
import numpy as np

# Different map types in the IllistrisTNG_1P dataset
map_types = ['Mgas', 'Vgas', 'T', 'Z', 'HI', 'ne', 'B', 'MgFe', 'Mcdm', 'Vcdm', 'Mstar', 'Mtot', 'P']
# URL links for the dataset
labelFile = 'params_1P_IllustrisTNG.txt'  # Labels file for the 1P IllustrisTNG 1P simulation
IllistrisTNG_IMG_FILES = ['Maps_' + _ + '_IllustrisTNG_1P_z=0.00.npy' for _ in map_types]  # List of all image files


# A custom dataset class for the CAMELS database
class CAMELS_Dataset:
    # Note this could inherit a different dataset 'prefab' eg. Pytorch Dataset for better data storage and manipulation

    def __init__(self, maps: list[str]):
        """
        :param maps: Initial maps to load dataset with
        """
        self.maps = {
            str(img): img for img in maps
        }
        try:
            self.labels = np.loadtxt('./data/params_1P_IllustrisTNG.txt')
        except FileNotFoundError:
            self.getFile('./data/', 'params_1P_IllustrisTNG.txt')
            self.labels = np.loadtxt('./data/params_1P_IllustrisTNG.txt')

        self.load_files(maps=maps)  # Loads the files onto the system if not already there

        self.train_val_test_split = (0.8, 0.1, 0.1)  # Change this value to change the split between data

        # Initialising properties (example shapes for 1 map)
        self.train_x = np.zeros((792, 256, 256))
        self.train_y = np.zeros((792, 6))
        self.val_x = np.zeros((99, 256, 256))
        self.val_y = np.zeros((99, 6))
        self.test_x = np.zeros((99, 256, 256))
        self.test_y = np.zeros((99, 6))

        self.state = {
            'maps': list(self.maps.keys()),
            'normalised': 'False',
            'train_num': self.train_x.shape[0],
            'val_num': self.val_x.shape[0],
            'test_num': self.test_x.shape[0]
        }

    def __delitem__(self, maps):
        self.del_maps(maps)

    def __len__(self):
        return sum(x.shape[0] for x in [self.train_x, self.val_x, self.test_x])

    def add_maps(self, maps_add: list[str], fileDir='./data/', verbose=True):
        # Add map(s) to the current database
        if type(maps_add) is not list:
            print('Please pass argument as a list of strings')
        maps_add_set = set(maps_add)
        maps_set = set(self.maps.keys())

        # Find common elements between the list and array
        already_added = maps_set.intersection(maps_add_set)

        for item in already_added:
            if verbose:
                print(f'Map {item} already in dataset, skipping...')
            maps_add_set.remove(item)  # Remove the item from the to add set

        self.load_files(fileDir, maps=list(maps_add_set), verbose=verbose)
        if verbose:
            print('All new maps loaded. Call method generate_dataset to regenerate the new dataset for the new maps')

    def del_maps(self, maps_del):
        # Delete map(s) from the current database
        if type(maps_del) is not list:
            print('Please pass argument as a list of strings')
        # Convert list to a set for faster membership checking
        maps_del_set = set(maps_del)
        maps_set = set(self.maps.keys())

        # Find common elements between the list and array
        to_del = maps_set.intersection(maps_del_set)

        for item in to_del:
            del self.maps[item]
            print(f'Map {item} removed from dataset')

    def update_state(self):
        # Updates the state variable
        self.state['maps'] = list(self.maps.keys())
        self.state['train_num'] = self.train_x.shape[0]
        self.state['val_num'] = self.val_x.shape[0]
        self.state['test_num'] = self.test_x.shape[0]

    def get_summary(self):
        # Returns current summary of dataset state
        self.update_state()
        return self.state

    def normalise(self, by_map=True, shuffle=True, channels=False):
        """
        Normalises all images in the dataset to range [-1, 1] (or [0, 1] for grayscale
        :param by_map: by each map individually so don't make one map particularly important over all others
        :param shuffle: Whether to shuffle the dataset when normalising (for plotting or testing)
        :return:
        """
        if by_map:
            if self.state['normalised'] == 'True: By Map':
                print("Dataset already normalised")
            else:
                self.maps = {key: np.float32(value) / np.float32(np.max(np.abs(value))) for key, value in
                             self.maps.items()}
                self.split_train_val_test(shuffle=shuffle, channels=channels)
                self.state['normalised'] = 'True: By Map'
        else:
            if self.state['normalised'] == 'True':
                print("Dataset already normalised")
                return
            else:
                self.generate_dataset(shuffle=shuffle, channels=channels)
                # Find maximum value in all images
                max_val = np.float32(np.max(np.abs(np.concatenate([self.train_x, self.val_x, self.test_x]))))

                self.train_x = np.float32(self.train_x) / max_val
                self.val_x = np.float32(self.val_x) / max_val
                self.test_x = np.float32(self.test_x) / max_val

                self.state['normalised'] = 'True'

    def load_files(self, fileDir='./data/', maps=None, verbose=True):
        # Make map_types not a mutable argument
        if maps is None:
            maps = map_types  # Uses all possible maps

        # Image file names from selected
        IllistrisTNG_IMG_FILES = ['Maps_' + _ + '_IllustrisTNG_1P_z=0.00.npy' for _ in maps]

        # Load the image files
        for i, file in enumerate(IllistrisTNG_IMG_FILES):
            if not os.path.isfile(fileDir + file):
                try:
                    self.getFile(fileDir, file)
                    if verbose:
                        print(f'File {file} loaded')
                    self.maps[maps[i]] = np.array([])  # Add the new map to the loaded maps
                except HTTPError as err:
                    if err.code == 404:
                        if verbose:
                            print(f"Cannot find file: '{file}'. Skipping...")
                    else:
                        raise
            else:
                # If map already downloaded but not in list of maps
                if maps[i] not in self.maps.keys():
                    self.maps[maps[i]] = np.array([])
                if verbose:
                    print(f'File {file} already on system, skipping download...')


    def getFile(self, fileDir, fileName):
        # Copy a network object to a local file
        print(f"""Downloading file 
        'https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/{fileName}'
        to {fileDir + fileName}. This may take a while...""")
        urllib.request.urlretrieve(
            'https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/' + fileName,
            fileDir + fileName)

    def generate_dataset(self, fileDir = './data/', shuffle=True, channels=False):
        """
        Updates the dataset values for current map -- performs a random shuffle on them also
        """
        for key in self.maps.keys():
            self.maps[key] = np.load(fileDir + 'Maps_' + key + '_IllustrisTNG_1P_z=0.00.npy')

        self.split_train_val_test(shuffle, channels)

    def split_train_val_test(self, shuffle=True, channels=False):
        # Splits data into training, validation and test data
        if not channels:
            # Splits the data within maps into training, validation and testing data
            all_data = np.array(list(self.maps.values()))
            # Remove the extra dimension that comes from map to pool all data into one long list
            reshaped_data = all_data.reshape(-1, *all_data.shape[-2:])
            # This gives shape (990, 256, 256) for single map (1980, 256, 256) for 2 maps

            # Extend the labels to make the first index shape of the image data
            labels_ext = np.repeat(self.labels, 15, axis=0)
            # Repeat the labels for every map we have included
            labels_ext = np.tile(labels_ext, (len(self.maps), 1))
        else:
            # Combine all arrays across all keys
            combined_array = []
            for value in self.maps.values():
                if isinstance(value, list):
                    combined_array.extend(value)
                else:
                    combined_array.append(value)

            # Convert the list of arrays to a single numpy array
            reshaped_data = np.stack(combined_array, axis=-1)

            # Extend the labels to make the first index shape of the image data
            labels_ext = np.repeat(self.labels, 15, axis=0)

        train_num = int(reshaped_data.shape[0] * self.train_val_test_split[0])
        val_num = int(reshaped_data.shape[0] * self.train_val_test_split[1])
        # Make remaining images test images
        test_num = reshaped_data.shape[0] - (train_num + val_num)

        if shuffle:
            # Random shuffle
            labels_ext, shuffled_data = self.shuffle_data(labels_ext, reshaped_data)
        else:
            labels_ext, shuffled_data = labels_ext, reshaped_data

        # Split the data into train, validation and test
        self.train_x = shuffled_data[:train_num]

        self.val_x = shuffled_data[train_num:train_num+val_num]
        self.test_x = shuffled_data[train_num+val_num:]

        self.train_y = labels_ext[:train_num]
        self.val_y = labels_ext[train_num:train_num+val_num]
        self.test_y = labels_ext[train_num+val_num:]

        self.state['normalised'] = 'False'


    def shuffle_data(self, labels, data):
        """
        Performs a random shuffle on labels and data that leave both in same groupings of the 66 lines
        :param labels: the image labels
        :param data: image data itself
        :return: shuffled_labels, shuffled_data
        """
        indices = np.arange(labels.shape[0])
        np.random.shuffle(indices)
        shuffled_labels = labels[indices]
        shuffled_data = data[indices]

        return shuffled_labels, shuffled_data

