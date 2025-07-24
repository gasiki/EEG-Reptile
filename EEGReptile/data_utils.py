import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import pickle
import os
from copy import deepcopy
import pathlib
import time


def experiment_storage(experiment_name: str, description=None):
    """
    Creates experiment storage directory with path = /experiment_name/date_time
    :param experiment_name: str - name of the experiment
    :param description: str (will be saved in experiment specific text file)
    :return: path to experiment storage directory in string format
    """
    p = 1
    path = pathlib.Path(pathlib.Path.cwd())
    if not os.path.exists((pathlib.Path(path, str(experiment_name)))):
        os.mkdir(pathlib.Path(path, str(experiment_name)))
    timestr = time.strftime("%Y-%m-%d_%H-%M_")
    experiment_name = experiment_name + "/{}".format(timestr)
    if os.path.exists((pathlib.Path(path, str(experiment_name)))):
        while os.path.exists(pathlib.Path(path, str(experiment_name) + str(p))):
            p += 1
        os.mkdir(pathlib.Path(path, str(experiment_name) + str(p)))
        experiment_name = str(experiment_name) + str(p)
    else:
        os.mkdir(pathlib.Path(path, str(experiment_name)))
    if description is not None and isinstance(description, str):
        path = './' + experiment_name + '/'
        with open(path + 'description.txt', 'w') as f:
            f.write(description)
    else:
        path = './' + experiment_name + '/'
        with open(path + 'description.txt', 'w') as f:
            f.write('Base description is not provided')
    return experiment_name


class SubjDataset(Dataset):   # create train dataset for one of subjects
    def __init__(self, data, actual_data):
        data = data.reset_index(drop=True)
        self.data = data['X']
        self.targets = data['Y'].values
        self.actual_data = actual_data

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        pos = self.data[idx].split(',')
        inputs = np.array(self.actual_data[str(pos[0])][int(pos[1])])
        inputs = inputs.transpose(0, 1)[None, :, :]
        return inputs, self.targets[idx]


class CrossSubjDataset(Dataset):   # create train dataset for grouping model
    def __init__(self, data, actual_data, groups):
        data = data.reset_index(drop=True)
        self.data = data['X']
        self.groups = groups
        self.actual_data = actual_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos = self.data[idx].split(',')
        group_n = 4
        for i in range(len(self.groups)):
            if int(pos[0]) in self.groups[i]:
                group_n = i
        inputs = np.array(self.actual_data[str(pos[0])][int(pos[1])])
        inputs = inputs.transpose(0, 1)[None, :, :]
        return inputs, group_n


class MetaDataset:
    """
    Class to handle EEG data for meta-learning
    """
    def __init__(self, dataset_name: str, description: str = None):
        """
        Initializes MetaDataset
        :param dataset_name: name of the dataset (str)
        :param description: description of dataset (str), needed for creating empty dataset
        """
        self.file = None
        if os.path.isdir('data_' + dataset_name):
            print('loading existing_dataset from: data_' + dataset_name)
            self.dataset_name = dataset_name
            self.data = {}
            self.subjects = []
            self.load_data()
        else:
            print('creating new empty dataset: ' + str(dataset_name))
            self.dataset_name = dataset_name
            self.subjects = []
            if description is None:
                self.description = 'Description not provided'
            else:
                self.description = description
            self.data = {}
            self.save_data()

    def load_data(self, logging=False):
        """
        Loads data (most times handled by meta dataset itself, without user interaction)
        :param logging: Boolean flag for logging or not
        """
        filename = 'data_' + self.dataset_name
        if not os.path.isdir(filename):
            raise ValueError('Specified filename does not exist')
        if logging:
            print('Dataset found')
        if logging:
            print('dataset file contains: ' + str(os.listdir(filename)))
        with open(filename + "/subjects.pkl", "rb") as fp:
            self.subjects = pickle.load(fp)
        if logging:
            print('subjects loaded: ' + str(self.subjects))
        if 'description.txt' in os.listdir(filename):
            with open(filename + "/description.txt", "rb") as fp:
                self.description = fp.read().decode('utf-8')
            if logging:
                print('description loaded: ' + str(self.description))
        else:
            self.description = 'Description not provided'
        self.data['actual_data'] = {}
        for subject in self.subjects:
            if 'subj_' + str(subject) in os.listdir(filename):
                self.data[subject] = {}
                with open(filename + '/subj_' + str(subject) + '/test_data.pkl', "rb") as fp:
                    self.data[subject]['test'] = pd.read_pickle(fp)  # TODO resolve
                with open(filename + '/subj_' + str(subject) + '/train_data.pkl', "rb") as fp:
                    self.data[subject]['train'] = pd.read_pickle(fp)
                with open(filename + '/subj_' + str(subject) + '/shape.pkl', "rb") as fp:
                    self.data['actual_data']['dim_' + str(subject)] = pickle.load(fp)
                self.data['actual_data'][str(subject)] = (
                    np.memmap(filename + '/subj_' + str(subject) + '/actual_data.dat',
                              dtype='float32', mode='r', shape=self.data['actual_data']['dim_' + str(subject)]))
            else:
                raise ValueError('Specified subject does not exist or dataset is broken')

    def save_data(self, subject=False, subject_data=None):
        """
        Saves data from new subject (most times handled by meta dataset itself, without user interaction)
        :param subject: subject id or False
        :param subject_data: Preprocessed data for subject
        """
        filename = 'data_' + self.dataset_name
        if os.path.isdir(filename) and not subject:
            raise ValueError('Specified dataset already exists')
        else:
            if subject:
                subj = self.subjects[-1]
                print('updating the dataset with subject ' + str(subj))
                os.mkdir(filename + '/subj_' + str(subj))
                with open(filename + "/subjects.pkl", "wb") as fp:
                    pickle.dump(self.subjects, fp)
                with open(filename + "/description.txt", "wb") as fp:
                    fp.write(self.description.encode('utf-8'))
                with open(filename + '/subj_' + str(subj) + '/test_data.pkl', 'wb') as fp:
                    self.data[subj]['test'].to_pickle(fp)
                with open(filename + '/subj_' + str(subj) + '/train_data.pkl', 'wb') as fp:
                    self.data[subj]['train'].to_pickle(fp)
                with open(filename + '/subj_' + str(subj) + '/shape.pkl', "wb") as fp:
                    pickle.dump(subject_data.shape, fp)
                fp = np.memmap(filename + '/subj_' + str(subj) + '/actual_data.dat', dtype='float32',
                               mode='w+', shape=subject_data.shape)
                fp[:] = subject_data[:]
                fp.flush()
            else:
                os.mkdir(filename)
                with open(filename + "/subjects.pkl", "wb") as fp:
                    pickle.dump(self.subjects, fp)
                with open(filename + "/description.txt", "wb") as fp:
                    fp.write(self.description.encode('utf-8'))

    def add_subject_from_xy(self, subject_id: int | str, x: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """
        Function for adding new subject data to the meta-dataset. From np arrays x and y.
        :param subject_id: int subject id
        :param x: np.ndarray of shape (n_samples, n_channels, window_len)
        :param y: np.ndarray of shape (n_samples), consists of labels for classes 0 to n_classes - 1
        :param test_size: float test size in range from 0.0 to 1.0 determining the proportion of test set
        """
        if subject_id in self.subjects:
            raise ValueError('Specified subject already exists')
        if not isinstance(x, np.ndarray):
            raise ValueError('X data must be numpy array')
        if len(x.shape) != 3:
            raise ValueError('Shape of X data must be 3D')
        self.subjects.append(subject_id)
        self.data[subject_id] = {}
        x_id = []
        for i in range(len(x)):
            x_id.append('{},{}'.format(subject_id, i))
        data_dict = {'X': x_id, 'Y': y}
        n = list(set(y.tolist()))
        df = pd.DataFrame.from_dict(data_dict)
        t_n = int(len(df.index) * test_size / len(n))
        test_data = []
        for i in n:
            test_data.append(df.loc[df['Y'] == i][-t_n:])
        self.data[subject_id]['test'] = pd.concat(test_data)
        self.data[subject_id]['train'] = df.drop(self.data[subject_id]['test'].index)
        self.save_data(subject=True, subject_data=x)
        self.load_data()

    def get_data_dims(self):
        """
        Function to access data dimensions for X (EEG epoch)
        :return: (int, int) tuple with number of channels and length of signal
        """
        subj = self.subjects[0]
        shape = self.data['actual_data']['dim_' + str(subj)]
        channels = shape[1]
        data_len = shape[2]
        return channels, data_len

    def groups_test_data(self, subjects, groups):
        data = pd.DataFrame()
        for sub in subjects:
            data = pd.concat([data, self.data[sub]['test']], ignore_index=True)
        return CrossSubjDataset(data, self.data['actual_data'], groups)

    def part_data_subj(self, subj: int, rs: int = 42, n: int = 1):
        """
        Function to get random part of the subject data with length n samples
        :param subj: subject id
        :param rs: random state
        :param n: number of samples
        :return: tuple of datasets for train, validation and test sets
        """
        if subj not in self.subjects:
            raise ValueError('Specified subject does not exist in this dataset')
        data = self.data[subj]['train']
        train_data = data.sample(n=n, random_state=rs)
        val_data = data.drop(train_data.index)
        test_data = self.data[subj]['test']
        return (SubjDataset(train_data, self.data['actual_data']), SubjDataset(val_data, self.data['actual_data']),
                SubjDataset(test_data, self.data['actual_data']))

    def all_data_subj(self, subj: int, n: int = 1, mode='epoch', early_stopping=0):
        """
        Function to get all data for specific subject
        :param subj: subject id
        :param n: batch size for each task
        :param mode: specifies mode of meta train (epoch, batch, single_batch)
        :param early_stopping: 0 means no early stopping and val task will be None
        :return: tuple of tasks for train and validation task is array of datasets
        """
        tasks = []
        val = None
        if subj not in self.subjects:
            raise ValueError('Specified subject does not exist in this dataset')
        if early_stopping == 0:
            dat = pd.concat([self.data[subj]['train'], self.data[subj]['test']], ignore_index=True)
        else:
            dat = pd.concat([self.data[subj]['train'], self.data[subj]['test']], ignore_index=True)
            ny = dat['Y'].nunique(dropna=True)
            num_d = dat.shape[0]
            num_d = int(num_d * 0.1 / ny)
            val = pd.DataFrame()
            for i in range(ny):
                val = pd.concat([val, dat.loc[dat['Y'] == i].tail(num_d)])
            dat = dat.drop(val.index)
            val = SubjDataset(val, self.data['actual_data'])
        if mode == 'epoch':
            tasks.append(SubjDataset(dat, self.data['actual_data']))
        elif mode == 'batch':
            ny = dat['Y'].nunique(dropna=True)
            dat = dat.sample(frac=1)
            train_data = []
            dlens = []
            for i in range(ny):
                train_data.append(dat.loc[dat['Y'] == i])
                dlens.append(len(train_data[i].index))
            n = int(n/ny)
            data_len = int(min(dlens)/n)
            for i in range(data_len):
                local_data = pd.DataFrame()
                a = i*n
                b = n*(i+1)
                for tr_data in train_data:
                    local_data = pd.concat([local_data, tr_data[a:b]], ignore_index=True)
                tasks.append(SubjDataset(local_data, self.data['actual_data']))
        elif mode == 'single_batch':
            tasks.append(SubjDataset(dat.sample(n), self.data['actual_data']))
        else:
            raise ValueError('incorrect meta-learning mode specified')
        return tasks, val

    def test_data_subj(self, subj: int):
        """
        function to get test dataset for given subject
        :param subj: int subject id
        :return: test dataset
        """
        if subj not in self.subjects:
            raise ValueError('Specified subject does not exist in this dataset')
        return SubjDataset(self.data[subj]['test'], self.data['actual_data'])

    def other_data_subj(self, subj, subjects):
        data = pd.DataFrame()
        using_subjects = deepcopy(subjects)
        using_subjects.remove(subj)
        for sub in using_subjects:
            data = pd.concat([data, self.data[sub]['train']], ignore_index=True)
        return SubjDataset(data, self.data['actual_data'])

    def multiple_data(self, subjects):
        data = pd.DataFrame()
        for sub in subjects:
            data = pd.concat([data, self.data[sub]['train']], ignore_index=True)
        return SubjDataset(data, self.data['actual_data'])

    def last_n_data_subj(self, subj, train: int, rs=42, return_dataset=True):
        """
        Function to get the last n points of subject data in the dataset or np array
        (with same number of points for each class)
        :param subj: int subject id
        :param train: int length of data
        :param rs: random state
        :param return_dataset: boolean to return dataset (default) or np array
        :return: tuple of (train and test dataset) or (train and test np data)
        """
        if subj not in self.subjects:
            raise ValueError('Specified subject does not exist in this dataset')
        data = self.data[subj]['train']
        n = data['Y'].nunique(dropna=True)
        if train < n:
            n = train
        train_data = pd.DataFrame()
        for i in range(n):
            train_data = pd.concat([train_data,
                                    data.loc[data['Y'] == i].sample(n=int(train/n), random_state=rs)])
        test_data = self.data[subj]['test']
        if return_dataset:
            return SubjDataset(train_data, self.data['actual_data']), SubjDataset(test_data, self.data['actual_data'])
        else:
            return train_data.to_numpy(copy=True), test_data.to_numpy(copy=True)

    def m_data(self, using_subjects: list, groups):
        """
        Function to get dataset for learning grouping model
        :param using_subjects: list of subjects
        :param groups: list of groups
        :return: CrossSubDataset (dataset where target is number of group)
        """
        data = pd.DataFrame()
        groups = deepcopy(groups)
        for sub in using_subjects:
            data = pd.concat([data, self.data[sub]['train']], ignore_index=True)
        return CrossSubjDataset(data, self.data['actual_data'], groups)
