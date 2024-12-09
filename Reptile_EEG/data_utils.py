import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import pickle
import zipfile as zp
import os
from copy import deepcopy
import pathlib
import tempfile
import time


def experiment_storage(experiment_name: str):
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
    return experiment_name


class SubjDataset(Dataset):   # create train dataset for one of subjects
    def __init__(self, data, name):
        self.name = name
        data = data.reset_index(drop=True)
        self.data = data['X']
        self.targets = data['Y'].values

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        pos = self.data[idx].split(',')
        pt = self.name + '/subj_' + str(pos[0]) + '/' + str(pos[1]) + '.pkl'
        with open(pt, mode='rb') as f:
            inputs = np.load(f)
        inputs = inputs.transpose(0, 1)[np.newaxis, :, :]
        return inputs, self.targets[idx]


class MetaDataset:
    def __init__(self, dataset_name: str, description: str = None):
        self.file = None
        if os.path.isfile('data_' + dataset_name + '.zip'):
            print('loading existing_dataset from: data_' + dataset_name + '.zip')
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
        filename = 'data_' + self.dataset_name
        if not os.path.isfile(filename + '.zip'):
            raise ValueError('Specified filename does not exist')
        if logging:
            print('Dataset found')
        with zp.ZipFile(filename + '.zip') as zf:
            if logging:
                print('dataset file contains: ' + str(zf.namelist()))
            with zf.open("subjects.pkl", "r") as fp:
                self.subjects = pickle.load(fp)
            if logging:
                print('subjects loaded: ' + str(self.subjects))
            if 'description.txt' in zf.namelist():
                with zf.open("description.txt", "r") as fp:
                    self.description = fp.read().decode('utf-8')
                if logging:
                    print('description loaded: ' + str(self.description))
            else:
                self.description = 'Description not provided'
            for subject in self.subjects:
                if 'subj_' + str(subject) + '/test_data.pkl' in zf.namelist():
                    self.data[subject] = {}
                    with zf.open('subj_' + str(subject) + '/test_data.pkl') as fp:
                        self.data[subject]['test'] = pd.read_pickle(fp)  # TODO resolve
                    with zf.open('subj_' + str(subject) + '/train_data.pkl') as fp:
                        self.data[subject]['train'] = pd.read_pickle(fp)
                else:
                    raise ValueError('Specified subject does not exist or dataset is broken')
        path = pathlib.Path(pathlib.Path.cwd())
        self.file = tempfile.TemporaryDirectory(dir=path)
        with zp.ZipFile(filename + '.zip', 'r') as zf:
            for sub in self.subjects:
                sub_path = 'subj_{}/'.format(str(sub))
                os.mkdir(self.file.name + "/" + sub_path)
                for name in zf.namelist():
                    if name.startswith(sub_path):
                        if not name.endswith('data.pkl'):
                            with zf.open(name) as fp:
                                buffer = np.load(fp)
                            with open(self.file.name + "/" + name, 'wb') as f:
                                np.save(f, buffer)

    def save_data(self, subject=False, subject_data=None):
        filename = 'data_' + self.dataset_name
        if os.path.isfile(filename + '.zip') and not subject:
            raise ValueError('Specified dataset already exists')
        else:
            if subject:
                print('updating the dataset with subject ' + str(self.subjects[-1]))
                with zp.ZipFile(filename + '_new.zip', 'w') as zf:
                    with zf.open("subjects.pkl", "w") as fp:
                        pickle.dump(self.subjects, fp)
                    with zf.open("description.txt", "w") as fp:
                        fp.write(self.description.encode('utf-8'))
                    subj = self.subjects[-1]
                    with zf.open('subj_' + str(subj) + '/test_data.pkl', 'w') as fp:
                        self.data[subj]['test'].to_pickle(fp)
                    with zf.open('subj_' + str(subj) + '/train_data.pkl', 'w') as fp:
                        self.data[subj]['train'].to_pickle(fp)
                    for data in subject_data:
                        with zf.open('subj_' + str(subj) + '/' + str(data['id']) + '.pkl', 'w') as fp:
                            np.save(fp, data['data'])
                    zp_in = zp.ZipFile(filename + '.zip', 'r')
                    for sub in self.subjects[:-1]:
                        sub_path = 'subj_{}/'.format(str(sub))
                        for name in zp_in.namelist():
                            if name.startswith(sub_path):
                                buffer = zp_in.read(name)
                                zf.writestr(name, buffer)
                    zp_in.close()
                os.remove(filename + '.zip')
                os.rename(filename + '_new.zip', filename + '.zip')
            else:
                with zp.ZipFile(filename + '.zip', 'w') as zf:
                    with zf.open("subjects.pkl", "w") as fp:
                        pickle.dump(self.subjects, fp)
                    with zf.open("description.txt", "w") as fp:
                        fp.write(self.description.encode('utf-8'))

    def add_subject_from_xy(self, subject_id: int, x: list, y: list, test_size: float = 0.2):
        if subject_id in self.subjects:
            raise ValueError('Specified subject already exists')
        self.subjects.append(subject_id)
        self.data[subject_id] = {}
        subj_data = []
        x_id = []
        for i in range(len(x)):
            dat = {'id': i, 'data': x[i]}
            x_id.append('{},{}'.format(subject_id, i))
            subj_data.append(dat)
        data_dict = {'X': x_id, 'Y': y}
        n = len(list(set(y)))
        df = pd.DataFrame.from_dict(data_dict)
        t_n = int(len(df.index) * test_size / n)
        test_data = []
        for i in range(n):
            test_data.append(df.loc[df['Y'] == i][-t_n:])
        self.data[subject_id]['test'] = pd.concat(test_data)
        self.data[subject_id]['train'] = df.drop(self.data[subject_id]['test'].index)
        self.save_data(subject=True, subject_data=subj_data)
        self.load_data()

    def part_data_subj(self, subj: int, rs: int = 42, n: int = 1):
        if subj not in self.subjects:
            raise ValueError('Specified subject does not exist in this dataset')
        data = self.data[subj]['train']
        train_data = data.sample(n=n, random_state=rs)
        val_data = data.drop(train_data.index)
        test_data = self.data[subj]['test']
        return (SubjDataset(train_data, self.file.name), SubjDataset(val_data, self.file.name),
                SubjDataset(test_data, self.file.name))

    def all_data_subj(self, subj: int, n: int = 1, mode='epoch', early_stopping=0):
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
            val = SubjDataset(val, self.file.name)
        if mode == 'epoch':
            tasks.append(SubjDataset(dat, self.file.name))
        elif mode == 'batch':
            dat = dat.sample(frac=1)
            data_len = int(len(dat)/n)
            for i in range(data_len):
                a = i*n
                b = n*(i+1)
                tasks.append(SubjDataset(dat[a:b], self.file.name))
        elif mode == 'single_batch':
            tasks.append(SubjDataset(dat.sample(n), self.file.name))
        else:
            raise ValueError('incorrect meta-learning mode specified')
        return tasks, val

    def test_data_subj(self, subj: int):
        if subj not in self.subjects:
            raise ValueError('Specified subject does not exist in this dataset')
        return SubjDataset(self.data[subj]['test'], self.file.name)

    def other_data_subj(self, subj, subjects):
        data = pd.DataFrame()
        using_subjects = deepcopy(subjects)
        using_subjects.remove(subj)
        for sub in using_subjects:
            data = pd.concat([data, self.data[sub]['train']], ignore_index=True)
        return SubjDataset(data, self.file.name)

    def multiple_data(self, subjects):
        data = pd.DataFrame()
        for sub in subjects:
            data = pd.concat([data, self.data[sub]['train']], ignore_index=True)
        return SubjDataset(data, self.file.name)

    def last_n_data_subj(self, subj, train: int, rs=42, return_dataset=True):
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
            return SubjDataset(train_data, self.file.name), SubjDataset(test_data, self.file.name)
        else:
            return train_data.to_numpy(copy=True), test_data.to_numpy(copy=True)
