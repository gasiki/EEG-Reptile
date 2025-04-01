import os.path
import optuna
from .EEGnet_model import *
from .data_utils import *
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from copy import deepcopy
import matplotlib.pyplot as plt
import joblib
import numpy as np
import multiprocessing
import json
from typing import Union


def meta_step(weights_before: OrderedDict, weights_after: OrderedDict, meta_weights: OrderedDict,  model,
              outestepsize=0.8, outestepsize1: float = None, iteration=None, epochs=None, meta_optimizer=None,
              task_len: int = 1):
    if iteration is None or epochs is None or epochs == 0:
        raise ValueError('iterative error in meta step definition')
    if not meta_optimizer:
        if outestepsize1 is None:
            # outerstepsize0 = outestepsize * (1 - iteration / epochs) / task_len
            # outerstepsize0 = outestepsize / task_len
            outerstepsize0 = outestepsize / task_len / epochs
            state_dict = {name: meta_weights[name] + (weights_after[name] - weights_before[name]) * outerstepsize0
                          for name in weights_before}
        else:
            outerstepsize0 = outestepsize / task_len
            outerstep1 = outestepsize1 / task_len
            state_dict = {name: meta_weights[name] + (
                    weights_after[name] - weights_before[name]) * outerstepsize0 if name.startswith(
                'conv_features') else meta_weights[name] + (weights_after[name] - weights_before[name]) * outerstep1 for
                          name in weights_before}
    else:
        rd = []
        meta_optimizer.zero_grad()
        model.load_state_dict(meta_weights)
        model.train()
        for name in weights_before:
            if name.endswith('weight') or name.endswith('bias'):
                rd.append((weights_after[name] - weights_before[name])/task_len)
        for p, d in zip(model.parameters(), rd):
            p.grad = d
        meta_optimizer.step()
        model.eval()
        state_dict = deepcopy(model.state_dict())
    return state_dict


# todo: add the subdroprate call for exp
def meta_weights_init(model, subjects, meta_dataset, sub_drop_rate=0.15, in_lr=0.0063, in_epochs=20):
    print('Meta weights initialization for subjects: {}'.format(subjects))
    naive_weights = deepcopy(model.state_dict())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    in_epochs = in_epochs*5
    using_subjects = deepcopy(subjects)
    meta_weights = None
    for sub in subjects:
        dat = meta_dataset.all_data_subj(subj=sub, mode='epoch')
        dat = DataLoader(dat[0], batch_size=64, drop_last=True, shuffle=True)
        model.load_state_dict(naive_weights)
        optim = torch.optim.AdamW(model.parameters(), lr=in_lr*10)
        _, _ = train(model, optim, torch.nn.CrossEntropyLoss(), dat, epochs=in_epochs, device=device, logging=False)
        weights_after = deepcopy(model.state_dict())
        if meta_weights is None:
            meta_weights = {name: weights_after[name] / len(subjects) for name in weights_after}
        else:
            meta_weights = {name: meta_weights[name] + (weights_after[name] / len(subjects)) for name in weights_after}
    naive_weights = deepcopy(meta_weights)
    meta_weights = {}
    for sub in subjects:
        dat = meta_dataset.all_data_subj(subj=sub, mode='epoch')
        dat = DataLoader(dat[0], batch_size=64, drop_last=True, shuffle=True)
        model.load_state_dict(naive_weights)
        optim = torch.optim.Adam(model.parameters(), lr=in_lr)
        _, _ = train(model, optim, torch.nn.CrossEntropyLoss(), dat, epochs=in_epochs, device=device, logging=False)
        meta_weights[sub] = deepcopy(model.state_dict())
    sub_drop_rate = round(sub_drop_rate * len(subjects))
    print('Bad subjects dropout rate: {}'.format(sub_drop_rate))
    if sub_drop_rate > 0:
        stat = []
        for sub in subjects:
            dif_sub = deepcopy(subjects)
            dif_sub.remove(sub)
            output_weights = None
            for subj in dif_sub:
                if output_weights is None:
                    output_weights = {name: meta_weights[subj][name] / len(using_subjects) for name in meta_weights[subj]}
                else:
                    output_weights = {name: output_weights[name] + (meta_weights[subj][name] / len(using_subjects))
                                      for name in meta_weights[subj]}
            st = sum((x.cpu() - y.cpu()).abs().sum() for x, y in zip(meta_weights[sub].values(), output_weights.values()))
            stat.append(st)
        stat = np.array(stat)
        stat = np.argpartition(stat, -sub_drop_rate)[-sub_drop_rate:]
        print('{} subjects will be dropped'.format(sub_drop_rate))
        dropped_subjects = []
        for i in stat:
            dropped_subjects.append(using_subjects[i])
        for sub in dropped_subjects:
            using_subjects.remove(sub)
        print('subjects dropped: {}'.format(str(dropped_subjects)))
    output_weights = None
    for sub in using_subjects:
        output_weights = {name: meta_weights[sub][name] / len(using_subjects) for name in meta_weights[sub]}
    return output_weights, using_subjects


def meta_learner(model, subjects, epochs: int, batch_size: int, in_epochs: int, in_lr,
                 meta_optimizer, lr1, lr2, device, mode='epoch', early_stopping=0, metadataset=None, logging=True,
                 sub_drop_rate=0.1, meta_w_init=True):
    if mode == 'epoch':
        n = batch_size
    elif mode == 'batch':
        n = batch_size
    elif mode == 'single_batch':
        n = batch_size
    else:
        raise ValueError('incorrect meta-training mode')
    if metadataset is None:
        raise ValueError('Meta dataset not specified for meta learner')
    flag = 0
    flag2 = 1
    best_stat = 0
    best_stat_epoch = 0
    early_model = copy.deepcopy(model.state_dict())
    meta_weights = None
    # Todo: remove test supplyments
    #full_test_stat = []
    # Todo: remove test supplyments
    #lr_max = in_lr
    lr_max = in_lr*10
    lrstep = 20/epochs
    olr_max = 1
    olr_step = 20/epochs
    ex_o_lr = lr1
    if meta_w_init:
        meta_weights, using_subjects = meta_weights_init(model=deepcopy(model), subjects=subjects,
                                                         meta_dataset=metadataset, sub_drop_rate=sub_drop_rate,
                                                         in_lr=in_lr, in_epochs=in_epochs)
    else:
        meta_weights = deepcopy(model.state_dict())
        using_subjects = deepcopy(subjects)
    if isinstance(lr1, list):
        if not len(lr1) == len(using_subjects):
            raise ValueError('Multiple meta learning rates have wrong size')
    else:
        olr_max = lr1
        #olr_step = 20/epochs
    model = deepcopy(model)
    model.to(device)
    for iteration in range(epochs):
        val = []
        task = []
        ex_in_lr = lr_max * (1-(iteration * lrstep / 21))
        #ex_in_lr = in_lr
        if not isinstance(lr1, list):
            #ex_o_lr = olr_max * (1-(iteration * olr_step / 21))
            k1 = 10
            k2 = 0.01
            ex_o_lr = (k1*lr1-(lr1*(epochs*k1-k2))/(epochs-1)) * (iteration+1) + (epochs*k1-k2)*lr1/(epochs-1)
            #ex_o_lr = 1/(10 * (epochs-1)) * (-99 * lr1 * (iteration+1) + (100 * epochs - 1)*lr1)   #todo: fix if not work
        for i in range(len(using_subjects)):
            # Todo: remove test supplyments
            train_tasks, vals = metadataset.all_data_subj(subj=using_subjects[i], n=n, mode=mode,
                                                          early_stopping=early_stopping)
            #train_tasks, vals = metadataset.all_data_subj(subj=using_subjects[i], n=n, mode=mode,
            #                                              early_stopping=1)
            #val.append(vals)
            # Todo: remove test supplyments
            task.append(train_tasks)
            if early_stopping != 0:
                val.append(vals)
        task_len = len(task[0])
        for j in range(task_len):
            weights_before = deepcopy(meta_weights)
            for i in range(len(task)):
                if isinstance(lr1, list):
                    ex_o_lr = lr1[i]
                    k1 = 1
                    k2 = 0.01
                    ex_o_lr = ((k1*ex_o_lr-(ex_o_lr*(epochs*k1-k2))/(epochs-1)) * (iteration+1) +
                               (epochs*k1-k2)*ex_o_lr/(epochs-1))
                dat = DataLoader(task[i][j], batch_size=n, drop_last=False, shuffle=False)
                model.load_state_dict(weights_before)
                optimizer = torch.optim.AdamW(model.parameters(), lr=ex_in_lr)    #todo: lr_size
                _, _ = train(model, optimizer, torch.nn.CrossEntropyLoss(), dat, epochs=in_epochs, device=device,
                             logging=False)
                model.eval()
                weights_after = deepcopy(model.state_dict())
                meta_weights = meta_step(weights_before=weights_before, weights_after=weights_after,
                                         meta_weights=meta_weights, outestepsize=ex_o_lr,
                                         outestepsize1=lr2, epochs=in_epochs,
                                         iteration=iteration, meta_optimizer=meta_optimizer, model=model,
                                         task_len=len(task))   # fixme size of task?
            # Todo: remove test supplyments
            # test_supp = 1
            # if test_supp:
            #     stat = []
            #     model.load_state_dict(meta_weights)
            #     for val_d in val:
            #         test_data_loader = DataLoader(val_d, batch_size=500, drop_last=False, shuffle=False)
            #         model.eval()
            #         with torch.no_grad():
            #             for batch in test_data_loader:
            #                 inputs, targets = batch
            #             inputs = inputs.to(device=device, dtype=torch.float)
            #             output = model(inputs)
            #             targets = targets.type(torch.LongTensor)
            #             targets = targets.to(device=device, dtype=torch.float)
            #             correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            #             stat.append(torch.sum(correct).item() / len(targets))
            #     stat = np.array(stat)
            #     stat = (np.quantile(stat, 0.25) + 2 * stat.mean())/3
            #     print('Val stat is: {}'.format(stat))
            #     full_test_stat.append(stat)
            # Todo: remove test supplyments
            if early_stopping != 0:
                stat = []
                model.load_state_dict(meta_weights)
                for val_d in val:
                    test_data_loader = DataLoader(val_d, batch_size=500, drop_last=False, shuffle=False)
                    model.eval()
                    with torch.no_grad():
                        for batch in test_data_loader:
                            inputs, targets = batch
                        inputs = inputs.to(device=device, dtype=torch.float)
                        output = model(inputs)
                        targets = targets.type(torch.LongTensor)
                        targets = targets.to(device=device, dtype=torch.float)
                        correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
                        stat.append(torch.sum(correct).item() / len(targets))
                stat = np.array(stat)
                stat = (np.quantile(stat, 0.25) + 2 * stat.mean())/3
                if stat > best_stat:
                    print('in process ' + str(multiprocessing.current_process().name) +
                          '\nnew best stat: {:.3f}, on epoch: {}/{}, batch: {}/{}'.format(stat, iteration, epochs,
                                                                                          j, task_len))
                    best_stat = stat
                    flag = 0
                    flag2 = 1
                if flag == 0:
                    early_model = copy.deepcopy(model.state_dict())
                    best_stat_epoch = iteration
                    flag = 1
                else:
                    flag += 1
                if flag/flag2 >= early_stopping*task_len/3:
                    ex_in_lr = ex_in_lr*0.85
                    lr_max = lr_max*0.85
                    flag2 += 1
                if flag >= early_stopping * task_len:
                    print('Early stopping! With best stat: {}, on epoch: {}'.format(best_stat, best_stat_epoch))
                    model.load_state_dict(early_model)
                    return model
        if logging:
            print('in process ' + str(multiprocessing.current_process().name) +
                  ' ended epoch: {}/{}'.format(iteration, epochs))
        model.load_state_dict(meta_weights)
    if flag != 0:
        model.load_state_dict(early_model)
        return model
    return model


def params_pretrain(trial, model, metadataset: MetaDataset, tr_sub, tst_sub, double_meta_step=False,
                    mode='epoch', meta_optimizer=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = deepcopy(model)
    model.to(device)
    if not meta_optimizer:
        if double_meta_step:
            params = {
                'innerepochs': trial.suggest_int('innerepochs', 1, 20),
                'oterepochs': trial.suggest_int('oterepochs', 4, 20),
                'outerstepsize0': trial.suggest_float('outerstepsize0', 0.1, 0.9),
                'outerstepsize1': trial.suggest_float('outerstepsize1', 0.1, 0.9),
                'in_lr': trial.suggest_float('in_lr', 0.000006, 0.005),
                'in_datasamples': trial.suggest_int('in_datasamples', 4, 50)
            }
        else:
            params = {
                'innerepochs': trial.suggest_int('innerepochs', 1, 20),
                'oterepochs': trial.suggest_int('oterepochs', 4, 30),
                'outerstepsize0': trial.suggest_float('outerstepsize0', 0.1, 0.9),
                'outerstepsize1': None,
                'in_lr': trial.suggest_float('in_lr', 0.00001, 0.009),
                'in_datasamples': trial.suggest_int('in_datasamples', 4, 50)
            }
    else:
        params = {
            'innerepochs': trial.suggest_int('innerepochs', 1, 20),
            'oterepochs': trial.suggest_int('oterepochs', 4, 20),
            'outerstepsize0': trial.suggest_float('outerstepsize0', 0.1, 0.9),
            'outerstepsize1': None,
            'in_lr': trial.suggest_float('in_lr', 0.000006, 0.005),
            'in_datasamples': trial.suggest_int('in_datasamples', 4, 50)
        }
        meta_optimizer = torch.optim.Adam(model.parameters(), lr=params['outerstepsize0'])
    n = params['in_datasamples']
    if tst_sub is not None:
        model = meta_learner(model, tr_sub, int(params['oterepochs']), n, params['innerepochs'], params['in_lr'],
                             meta_optimizer, params['outerstepsize0'], params['outerstepsize1'], device, mode,
                             0, metadataset, logging=False)
        stat = []
        for test_sub in tst_sub:
            train_data, test_data, _ = metadataset.part_data_subj(subj=test_sub, n=n, rs=42)
            test_data_loader = DataLoader(test_data, batch_size=500, drop_last=False, shuffle=False)
            with torch.no_grad():
                for batch in test_data_loader:
                    inputs, targets = batch
                inputs = inputs.to(device=device, dtype=torch.float)
                output = model(inputs)
                targets = targets.type(torch.LongTensor)
                targets = targets.to(device=device, dtype=torch.float)
                correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
                stat.append(torch.sum(correct).item() / len(targets))
        stat = np.array(stat)
        stat = (np.quantile(stat, 0.25) + 2 * stat.mean())/3
    else:
        st = []
        t_s = np.array_split(tr_sub, 2)
        for i in range(2):
            model = meta_learner(model, t_s[i].tolist(), int(params['oterepochs']/4), n, params['innerepochs'], params['in_lr'],
                                 meta_optimizer, params['outerstepsize0'], params['outerstepsize1'], device, mode,
                                 0, metadataset, logging=False)
            stat = []
            for test_sub in t_s[-1-i]:
                train_data, test_data, _ = metadataset.part_data_subj(subj=test_sub, n=n, rs=42)
                test_data_loader = DataLoader(test_data, batch_size=500, drop_last=False, shuffle=False)
                with torch.no_grad():
                    for batch in test_data_loader:
                        inputs, targets = batch
                    inputs = inputs.to(device=device, dtype=torch.float)
                    output = model(inputs)
                    targets = targets.type(torch.LongTensor)
                    targets = targets.to(device=device, dtype=torch.float)
                    correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
                    stat.append(torch.sum(correct).item() / len(targets))
            stat = np.array(stat)
            stat = (np.quantile(stat, 0.25) + 2 * stat.mean())/3
            st.append(stat)
        stat = np.mean(st)
    return stat


def meta_params(metadataset: MetaDataset, model, tr_sub: list, tst_sub=None, trials=50, jobs=1, mode='single_batch',
                double_meta_step=False, meta_optimizer=False, experiment_name='experiment'):
    """
    function used for meta hyper-params search
    :param metadataset: this is working dataset
    :param tr_sub: list of subjects used for training in params search
    :param tst_sub: list of subjects or None used for testing ACC in params search if none 2-k fold of tr_sub is used
    :param model: model for which params search will be performed
    :param trials: number of optuna trials in param search
    :param jobs: how many parallel jobs to use in params search
    :param mode: mode of meta train, may be: single_batch, batch or epoch, single_batch is more time efficient
    :param double_meta_step: boolean flag for double meta step
    :param meta_optimizer: boolean flag for meta optimizer (Adam)
    :param experiment_name: str from the experiment_storage function
    :return: dict of params for meta training
    """
    p = 1
    lib_name = 'params_for_meta_training'
    path = pathlib.Path(pathlib.Path.cwd(), experiment_name)
    while os.path.exists(pathlib.Path(path, lib_name + str(p))):
        p += 1
    os.mkdir(pathlib.Path(path, lib_name + str(p)))
    path = pathlib.Path(path, lib_name + str(p))
    file_path = str(path.resolve()) + "/journal.log"
    lock_obj = optuna.storages.JournalFileOpenLock(file_path)
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(file_path, lock_obj=lock_obj),
    )
    func = lambda trial: params_pretrain(trial, model=model, metadataset=metadataset, tr_sub=tr_sub,
                                         tst_sub=tst_sub, double_meta_step=double_meta_step, mode=mode,
                                         meta_optimizer=meta_optimizer)
    study = optuna.create_study(storage=storage, study_name='optuna_meta_params', direction='maximize',
                                load_if_exists=True)
    study.optimize(func, n_trials=trials, n_jobs=jobs)
    params = study.best_params
    file_path = str(path.resolve()) + "/best_params.txt"
    with open(file_path, 'w') as file:
        file.write(json.dumps(params))
    return params


def group_meta_train(model, subjects, groups: int, epochs: int, batch_size: int, in_epochs: int, in_lr,
                     meta_optimizer, lr1, lr2, device, mode='epoch', early_stopping=0, metadataset=None,
                     logging=True, name=None, path='./', grouping_epochs=50):
    if metadataset is None:
        raise ValueError('Meta dataset not specified for group meta train')
    if groups * 2 > len(subjects):
        raise ValueError('To much groups for proposed amount of subjects')
    if logging:
        print('Started group meta training for subjects:{}'.format(subjects))
    if name is None:
        print('name noy specified, subjects will be used as name descriptor!')
        name = str(subjects)
    small_d_flag = False
    if len(subjects) / groups < 5:
        small_d_flag = True
        if logging:
            print('Dataset is considered small. Small dataset optimisation will be performed!')
    os.mkdir(path + 'groped_models_' + str(name) + '/')
    path = path + 'groped_models_' + str(name) + '/'
    if logging:
        print('Models will be stored in :{}'.format(path))
    model = deepcopy(model)
    model.to(device)
    model = meta_learner(model, subjects, epochs=int(epochs*0.75), batch_size=batch_size, in_epochs=in_epochs, in_lr=in_lr,
                         meta_optimizer=meta_optimizer, lr1=lr1, lr2=lr2, device=device, mode='batch', early_stopping=0,
                         metadataset=metadataset, logging=logging, sub_drop_rate=0.0)
    loss_fn = torch.nn.CrossEntropyLoss()
    per_subj_stats = {}
    all_meta_weights = deepcopy(model.state_dict())
    for subj in subjects:
        data = metadataset.test_data_subj(subj)
        test_data_loader = torch.utils.data.DataLoader(data, batch_size=500, drop_last=False, shuffle=False)
        model.eval()
        with torch.no_grad():
            for batch in test_data_loader:
                inputs, targets = batch
            inputs = inputs.to(device=device, dtype=torch.float)
            output = model(inputs)
            targets = targets.type(torch.LongTensor)
            targets = targets.to(device=device)
            per_subj_stats[subj] = float(loss_fn(output, targets).to('cpu'))
    sorted_output_names = []
    sorted_output_stats = []
    for k, v in sorted(per_subj_stats.items(), key=lambda item: item[1]):
        sorted_output_names.append(k)
        sorted_output_stats.append(v)
    if logging:
        print('Sorted stats:\n {}\n, for subjects:\n {} on meta-model'.format(sorted_output_stats, sorted_output_names))
    grouped = []
    bonus_len = 0
    first_gr_len = len(subjects) // groups
    if len(subjects) % groups > 1:
        bonus_len = (len(subjects) % groups) // 2
        first_gr_len += (len(subjects) % groups - bonus_len)
    if len(subjects) % groups == 1:
        first_gr_len += 1
    grouped.append(deepcopy(sorted_output_names[0:first_gr_len]))
    if logging:
        print('group number 1: {}'.format(grouped[0]))
    new_target = sorted_output_names[first_gr_len]
    acces_subjs = sorted_output_names[first_gr_len+1:]
    for gr in range(1, groups-1):
        inter_model = deepcopy(model)
        inter_model.load_state_dict(model.state_dict())
        inter_model.to(device=device)
        targ_s_data_loader = torch.utils.data.DataLoader(metadataset.all_data_subj(new_target)[0][0], batch_size=64,
                                                         drop_last=False, shuffle=True)
        optimizer = torch.optim.Adam(inter_model.parameters(), lr=in_lr)
        best_model, stats = train(inter_model, optimizer, torch.nn.CrossEntropyLoss(), targ_s_data_loader,
                                  epochs=50, device='cuda', logging=logging)
        sec_per_subj_stats = {}
        for subj in acces_subjs:
            data = metadataset.test_data_subj(subj)
            test_data_loader = torch.utils.data.DataLoader(data, batch_size=500, drop_last=False, shuffle=False)
            inter_model.eval()
            with torch.no_grad():
                for batch in test_data_loader:
                    inputs, targets = batch
                inputs = inputs.to(device=device, dtype=torch.float)
                output = inter_model(inputs)
                targets = targets.type(torch.LongTensor)
                targets = targets.to(device=device)
                sec_per_subj_stats[subj] = float(loss_fn(output, targets).to('cpu'))
            sorted_output_names = []
            sorted_output_stats = []
            for k, v in sorted(sec_per_subj_stats.items(), key=lambda item: item[1]):
                sorted_output_names.append(k)
                sorted_output_stats.append(v)
        grouped.append([new_target] + sorted_output_names[0:len(subjects) // groups + bonus_len - 1])
        new_target = sorted_output_names[len(subjects) // groups+bonus_len-1]
        acces_subjs = sorted_output_names[len(subjects) // groups+bonus_len:]
        bonus_len = 0
        if logging:
            print('group number {}: {}'.format(gr+1, grouped[gr]))
    grouped.append([new_target] + acces_subjs)
    if logging:
        print('last group: {}'.format(grouped[-1]))
        print('total number of groups: {}'.format(len(grouped)))
    dt_cross = metadataset.m_data(subjects, grouped)
    cross_data_loader = torch.utils.data.DataLoader(dt_cross, batch_size=64, drop_last=False, shuffle=True)
    in_channels, in_data_len = metadataset.get_data_dims()
    grouping_model_params = {
        'n_groups': len(grouped),
        'n_channels': in_channels,
        'n_data_points': in_data_len
    }
    grouping_model = inEEG_Net(num_classes=grouping_model_params['n_groups'],
                               num_channels=grouping_model_params['n_channels'],
                               time_points=grouping_model_params['n_data_points'],
                               depth_multiplier=2, point_reducer=8)
    grouping_model.to(device)
    gr_optimizer = torch.optim.Adam(grouping_model.parameters(), lr=in_lr)
    best_model, stats = train(grouping_model, gr_optimizer, torch.nn.CrossEntropyLoss(), cross_data_loader,
                              epochs=grouping_epochs, device=device, logging=logging)

    stat = []
    t_d_e = metadataset.groups_test_data(subjects, grouped)
    test_data_loader = torch.utils.data.DataLoader(t_d_e, batch_size=128, drop_last=False, shuffle=False)
    grouping_model.eval()
    with torch.no_grad():
        for batch in test_data_loader:
            inputs, targets = batch
            inputs = inputs.to(device=device, dtype=torch.float)
            output = grouping_model(inputs)
            targets = targets.type(torch.LongTensor)
            targets = targets.to(device=device, dtype=torch.float)
            correct = torch.eq(torch.max(torch.nn.functional.softmax(output, dim=1), dim=1)[1], targets)
            stat.append(torch.sum(correct).item() / len(targets))
    stat = np.array(stat)
    gr_stat = stat.mean()
    if logging:
        print('Accuracy for grouping model is: {}'.format(gr_stat))
    torch.save(grouping_model.state_dict(), path + 'grouping_model.pkl')
    with open(path + 'g_model_params.txt', 'w') as f:
        json.dump(grouping_model_params, f)
    with open(path + 'groups.txt', 'w') as f:
        f.write('Groups:\n')
        for gri in range(len(grouped)):
            f.write('Group ' + str(gri) + ': ' + str(grouped[gri]) + '\n')
        f.write('Grouping stat: ' + str(gr_stat))
    if logging:
        print('Grouping model and its params is saved!')
        print('Learning meta-models for groups started')
    for i in range(len(grouped)):
        meta_model = deepcopy(model)
        meta_model.load_state_dict(all_meta_weights)
        u_sub = grouped[i]
        o_lr = lr1/10
        if small_d_flag:
            o_lr = []
            u_sub = deepcopy(grouped[i])
            #target_o_lr = 0.3 / len(u_sub) * lr1 / 10
            target_o_lr = 0.9 * lr1
            for k in range(len(u_sub)):
                o_lr.append(target_o_lr)
            other_sub = [x for x in subjects if x not in u_sub]
            #untarget_o_lr = 0.7 / len(other_sub) * lr1 / 10
            untarget_o_lr = 0.1 * lr1
            for k in range(len(other_sub)):
                o_lr.append(untarget_o_lr)
            u_sub.extend(other_sub)

        meta_model = meta_learner(model=model, subjects=u_sub, epochs=int(epochs*0.25), batch_size=batch_size,
                                  in_epochs=in_epochs, in_lr=in_lr/10, meta_optimizer=meta_optimizer, lr1=o_lr, lr2=lr2,
                                  device=device, mode=mode, early_stopping=early_stopping,
                                  metadataset=metadataset, logging=logging, sub_drop_rate=0.0, meta_w_init=False)
        meta_model.to('cpu')
        torch.save(meta_model.state_dict(), path + 'model_' + str(i) + '.pkl')
        if logging:
            print('meta model for group {} trained and saved'.format(i))
    return path


def gr_m_load(path, dt_loader, device: Union[str, torch.device] = 'cpu', one_batch: bool = True,
              return_num: bool = False):
    with open(path + 'g_model_params.txt', 'r') as f:
        grouping_model_params = json.load(f)
    grouping_model = inEEG_Net(num_classes=grouping_model_params['n_groups'],
                               num_channels=grouping_model_params['n_channels'],
                               time_points=grouping_model_params['n_data_points'],
                               depth_multiplier=2, point_reducer=8)
    grouping_model.load_state_dict(torch.load(path + 'grouping_model.pkl'))
    grouping_model.to(device)
    grouping_model.eval()
    with torch.no_grad():
        for batch in dt_loader:
            inputs, targets = batch
            inputs = inputs.to(device=device, dtype=torch.float)
            output = grouping_model(inputs)
            output = torch.max(torch.mean(torch.nn.functional.softmax(output, dim=1), dim=0), dim=0)[1].item()
            if one_batch:
                break
    if return_num:
        return output
    else:
        state_dict = torch.load(path + 'model_' + str(output) + '.pkl')
        return state_dict


def meta_train(params: dict, model, metadataset: MetaDataset, wal_sub, path=None, name: str = None,
               mode='single_batch', meta_optimizer=False, subjects: list = None, loging=True, baseline=True,
               early_stopping=0, groups=None):
    if name is None:
        name = str(wal_sub)
    if subjects is None:
        subjects = metadataset.subjects
    if 'outerstepsize1' not in params.keys():
        params.update(outerstepsize1=None)
    if path is None:
        raise ValueError('Path for meta-trained model not specified for meta train')
    if loging:
        print("meta train for sub: " + str(subjects) + " started  in process"
              + str(multiprocessing.current_process().name))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = deepcopy(model)
    model.to(device)
    bs_model = deepcopy(model)
    bs_weights = deepcopy(model.state_dict())
    if meta_optimizer:
        meta_optimizer = torch.optim.AdamW(model.parameters(), lr=params['outerstepsize0'])
    n = params['in_datasamples']
    gr_model_path = None
    if groups is None:
        model = meta_learner(model, subjects, params['oterepochs'], n, params['innerepochs'], params['in_lr'],
                             meta_optimizer, params['outerstepsize0'], params['outerstepsize1'], device, mode,
                             early_stopping, metadataset)
        torch.save(model.state_dict(), (path + str(name) + "-reptile.pkl"))
        if loging:
            print("meta train for sub: " + str(subjects) + "completed")
    else:
        gr_model_path = group_meta_train(model, subjects, groups, epochs=params['oterepochs'], batch_size=n,
                                         in_epochs=params['innerepochs'], in_lr=params['in_lr'],
                                         meta_optimizer=meta_optimizer, lr1=params['outerstepsize0'],
                                         lr2=params['outerstepsize1'], device=device,
                                         mode=mode, early_stopping=early_stopping, metadataset=metadataset,
                                         logging=loging, name=name, path=path, grouping_epochs=50)
    test_data = metadataset.test_data_subj(subj=wal_sub)
    test_data_loader = DataLoader(test_data, batch_size=500, drop_last=False, shuffle=False)
    if groups is not None and gr_model_path is not None:
        model_state_dict = gr_m_load(path=gr_model_path, dt_loader=test_data_loader)
        model.load_state_dict(model_state_dict)
    with torch.no_grad():
        for batch in test_data_loader:
            inputs, targets = batch
        inputs = inputs.to(device=device, dtype=torch.float)
        output = model(inputs)
        targets = targets.type(torch.LongTensor)
        targets = targets.to(device=device, dtype=torch.float)
        correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
        stat = torch.sum(correct).item()/len(targets)
    if baseline:
        model = bs_model
        model.load_state_dict(bs_weights)
        optim = torch.optim.Adam(model.parameters(), lr=params['in_lr'])
        data = metadataset.multiple_data(subjects)
        data = DataLoader(data, batch_size=64, drop_last=True, shuffle=True)
        _, _ = train(model, optim, torch.nn.CrossEntropyLoss(), data, epochs=params['oterepochs'], device=device,
                     logging=loging)
        torch.save(model.state_dict(), (path + name + "-baseline.pkl"))
        test_data = metadataset.test_data_subj(subj=wal_sub)
        test_data_loader = DataLoader(test_data, batch_size=500, drop_last=False, shuffle=False)
        with torch.no_grad():
            for batch in test_data_loader:
                inputs, targets = batch
            inputs = inputs.to(device=device, dtype=torch.float)
            output = model(inputs)
            targets = targets.type(torch.LongTensor)
            targets = targets.to(device=device, dtype=torch.float)
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
            stat1 = torch.sum(correct).item()/len(targets)
        return [stat, stat1]
    else:
        return stat


def meta_exp(params: dict, model, target_sub: list, metadataset: MetaDataset, mode='batch',
             meta_optimizer=False, num_workers=1, experiment_name='experiment', all_subjects: list = None,
             baseline=True, early_stopping=0, groups=None):
    if all_subjects is None:
        all_subjects = metadataset.subjects
    path = './' + experiment_name + '/'
    if not os.path.exists(path + 'models/'):
        os.mkdir(path + 'models/')
    path = path + 'models/'
    stat = []
    stat1 = []
    task = []
    # this need to be here for multiprocessing
    model.share_memory()
    for sub in target_sub:
        target_subjects = deepcopy(all_subjects)
        target_subjects.remove(sub)
        task.append(tuple((params, model, metadataset, sub, path, None, mode, meta_optimizer, target_subjects, True,
                           baseline, early_stopping, groups)))
    con = multiprocessing.get_context('spawn')
    with con.Pool(num_workers) as p:
        res = p.starmap(meta_train, task)
        p.close()
        p.join()
    for st in res:
        stat.append(st[0])
        if baseline:
            stat1.append(st[1])
    print("Pretraining competed with mean cold ACC = " + str(sum(stat)/len(stat)))
    with open(path+'reptile_cold_stats.txt', 'w') as fp:
        for i in range(len(target_sub)):
            fp.write("reptile on subject " + str(target_sub[i]) + " ACC is " + str(stat[i]) + '\n\n')
        fp.write("Mean ACC = " + str(sum(stat)/len(stat)) + '\n\n')
        fp.write("Params: " + '\n\n')
        for k, v in params.items():
            fp.write(str(k) + ' == ' + str(v) + '\n\n')
        if baseline:
            for i in range(len(target_sub)):
                fp.write("baseline on subject " + str(target_sub[i]) + " ACC is " + str(stat1[i]) + '\n\n')
    return sum(stat)/len(stat)


def p_aftrain(trial, model, metadataset: MetaDataset, tst_sub, experiment_name, length=50, last_layer=False):
    a = trial.suggest_int('a_ep', 2, 12)
    ji = 0
    params = {
              'lr': trial.suggest_float('lr', 0.000001, 0.001),
              'a': a,
              'b': trial.suggest_int('b_ep', 0, 4*a-1)
              }
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = deepcopy(model)
    model.to(device)
    cn = None
    if last_layer:
        for n, p in model.named_parameters():
            p.requires_grad = False
            if n.startswith('out_features'):
                p.requires_grad = True
    stat = 0
    for sub in tst_sub:
        if cn is None:
            cn = metadataset.data[sub]['train']['Y'].nunique(dropna=True)
        ji = 0
        j = 0
        while cn * 2 ** ji < length + 1:
            model.load_state_dict(torch.load(experiment_name + "/models/" + str(sub) + "-reptile.pkl"))
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
            if not j == 0:
                train_data, test_data = metadataset.last_n_data_subj(subj=sub, train=j, rs=42)
                data = DataLoader(train_data, batch_size=j, drop_last=True, shuffle=True)
                _, _ = train(model, optimizer, torch.nn.CrossEntropyLoss(), data,
                             epochs=int(j*params['a']-params['b']), device=device, logging=False)

            else:
                train_data, test_data = metadataset.last_n_data_subj(subj=sub, train=1, rs=42)
            test_data_loader = DataLoader(test_data, batch_size=500, drop_last=False, shuffle=False)
            with torch.no_grad():
                for batch in test_data_loader:
                    inputs, targets = batch
                inputs = inputs.to(device=device, dtype=torch.float)
                output = model(inputs)
                targets = targets.type(torch.LongTensor)
                targets = targets.to(device=device, dtype=torch.float)
                correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
                stat += torch.sum(correct).item()/len(targets)
            j = cn * 2 ** ji
            ji += 1
    return stat/((ji+1) * len(tst_sub))


def aftrain_params(metadataset: MetaDataset, model, tst_subj: list, trials: int, jobs: int,
                   experiment_name='experiment', last_layer=False):
    path = pathlib.Path(pathlib.Path.cwd(), experiment_name)
    os.mkdir(pathlib.Path(path, 'af_params'))
    file_path = "./" + experiment_name + "/af_params/journal.log"
    lock_obj = optuna.storages.JournalFileOpenLock(file_path)
    storage = optuna.storages.JournalStorage(
        optuna.storages.JournalFileStorage(file_path, lock_obj=lock_obj),
    )
    func = lambda trial: p_aftrain(trial, model=model, metadataset=metadataset, tst_sub=tst_subj,
                                   experiment_name=experiment_name, last_layer=last_layer)
    study = optuna.create_study(storage=storage, study_name='optuna_af_params', direction='maximize')
    study.optimize(func, n_trials=trials, n_jobs=jobs)
    af_params = study.best_params
    with open("./" + experiment_name + "/af_params/aftrain_params.txt", 'w') as fp:
        for k, v in af_params.items():
            fp.write(str(k) + ' == ' + str(v) + '\n\n')
    return af_params


def aftrain(target_sub, model, af_params, metadataset: MetaDataset, iterations=1, length=50, logging=False,
            experiment_name='experiment', last_layer=False, groups=False):
    dt = {}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = deepcopy(model)
    model.to(device)
    if last_layer:
        for n, p in model.named_parameters():
            p.requires_grad = False
            if n.startswith('out_features'):
                p.requires_grad = True
    cn = metadataset.data[target_sub[0]]['train']['Y'].nunique(dropna=True)
    for sub in target_sub:
        if logging:
            print('afftrain for ' + str(sub) + ' subject of ' + str(len(target_sub)))
        data_points = []
        stat = []
        rstat = []
        for k in range(iterations):
            if logging:
                print('afftrain iteration ' + str(k+1) + ' of ' + str(iterations) + ' started')
            in_data_points = []
            in_stat = []
            r_in_stat = []
            ji = 0
            j = 0
            while cn * 2 ** ji < length + 1:
                if logging:
                    print('performed ' + str(j) + ' steps')
                in_data_points.append(j)
                if not groups:
                    model.load_state_dict(torch.load(experiment_name + "/models/" + str(sub) + "-reptile.pkl"))
                    optimizer = torch.optim.AdamW(model.parameters(), lr=af_params['lr'])
                if not j == 0:
                    train_data, test_data = metadataset.last_n_data_subj(subj=sub, train=j, rs=42+k)
                    data = DataLoader(train_data, batch_size=j, drop_last=True, shuffle=True)
                    if groups:
                        gm_path = experiment_name + "/models/" + 'groped_models_' + str(sub) + '/'
                        model.load_state_dict(gr_m_load(gm_path, data, device=device))
                        optimizer = torch.optim.AdamW(model.parameters(), lr=af_params['lr'])
                    _, _ = train(model, optimizer, torch.nn.CrossEntropyLoss(), data,
                                 epochs=int(j*af_params['a_ep']-af_params['b_ep']), device=device, logging=False)

                else:
                    train_data, test_data = metadataset.last_n_data_subj(subj=sub, train=1, rs=42+k)
                    if groups:
                        data = DataLoader(test_data, batch_size=10, drop_last=False, shuffle=False)
                        gm_path = experiment_name + "/models/" + 'groped_models_' + str(sub) + '/'
                        model.load_state_dict(gr_m_load(gm_path, data, device=device))
                test_data_loader = DataLoader(test_data, batch_size=500, drop_last=False, shuffle=False)
                model.eval()
                with torch.no_grad():
                    for batch in test_data_loader:
                        inputs, targets = batch
                    inputs = inputs.to(device=device, dtype=torch.float)
                    output = model(inputs)
                    targets = targets.type(torch.LongTensor)
                    targets = targets.to(device=device, dtype=torch.float)
                    correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
                    r_in_stat.append(torch.sum(correct).item()/len(targets))
                model.load_state_dict(torch.load(experiment_name + "/models/" + str(sub) + "-baseline.pkl"))
                optimizer = torch.optim.Adam(model.parameters(), lr=af_params['lr'])
                if not j == 0:
                    train_data, test_data = metadataset.last_n_data_subj(subj=sub, train=j, rs=42+k)
                    data = DataLoader(train_data, batch_size=j, drop_last=True, shuffle=True)
                    _, _ = train(model, optimizer, torch.nn.CrossEntropyLoss(), data,
                                 epochs=int(j*af_params['a_ep']-af_params['b_ep']),
                                 device=device, logging=False)

                else:
                    train_data, test_data = metadataset.last_n_data_subj(subj=sub, train=1, rs=42+k)
                test_data_loader = DataLoader(test_data, batch_size=500, drop_last=False, shuffle=False)
                model.eval()
                with torch.no_grad():
                    for batch in test_data_loader:
                        inputs, targets = batch
                    inputs = inputs.to(device=device, dtype=torch.float)
                    output = model(inputs)
                    targets = targets.type(torch.LongTensor)
                    targets = targets.to(device=device, dtype=torch.float)
                    correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
                    in_stat.append(torch.sum(correct).item()/len(targets))
                j = cn * 2 ** ji
                ji += 1
            rstat.append(r_in_stat)
            stat.append(in_stat)
            data_points.append(in_data_points)
        dt[str(sub) + '_baseline_data'] = deepcopy(stat)
        dt[str(sub) + '_reptile_data'] = deepcopy(rstat)
        dt[str(sub) + '_datapoints'] = deepcopy(data_points)
    if logging:
        print('aftraining complete')
    path = pathlib.Path(pathlib.Path.cwd(), experiment_name)
    os.mkdir(pathlib.Path(path, 'af_run'))
    path = pathlib.Path(path, 'af_run')
    fig, ax = plt.subplots(len(target_sub)+1, sharex=False, sharey=True, figsize=(12, 9*(len(target_sub)+1)))
    i = 0
    a_acc = []
    a_accr = []
    for sub in target_sub:
        acc = np.mean(np.array(dt[str(sub) + '_baseline_data']), axis=0)
        a_acc.append(acc)
        std = np.std(np.array(dt[str(sub) + '_baseline_data']), axis=0)
        accr = np.mean(np.array(dt[str(sub) + '_reptile_data']), axis=0)
        a_accr.append(accr)
        stdr = np.std(np.array(dt[str(sub) + '_reptile_data']), axis=0)
        num = dt[str(sub) + '_datapoints'][0]
        ax[i].plot(num, acc, label='EEGNet')
        ax[i].fill_between(num, acc - std, acc + std, alpha=0.2)
        ax[i].plot(num, accr, label='EEGNet with Reptile')
        ax[i].fill_between(num, accr - stdr, accr + stdr, alpha=0.2)
        ax[i].set_title('Learning curve for subject' + str(sub))
        ax[i].set_xlabel('Train data size')
        ax[i].set_ylabel('ACC')
        ax[i].legend(loc='best')
        i += 1
    sub = target_sub[0]
    acc = np.mean(a_acc, axis=0)
    std = np.std(a_acc, axis=0)
    accr = np.mean(a_accr, axis=0)
    stdr = np.std(a_accr, axis=0)
    num = dt[str(sub) + '_datapoints'][0]
    ax[i].plot(num, acc, label='EEGNet')
    ax[i].fill_between(num, acc - std, acc + std, alpha=0.2)
    ax[i].plot(num, accr, label='EEGNet with Reptile')
    ax[i].fill_between(num, accr - stdr, accr + stdr, alpha=0.2)
    ax[i].set_title('Mean learning curve')
    ax[i].set_xlabel('Train data size')
    ax[i].set_ylabel('ACC')
    ax[i].legend(loc='best')
    joblib.dump(dt, pathlib.Path(path, 'all_data_af_test.sav'))
    plt.savefig(pathlib.Path(path, "af_Learn_ACC_ALL" + ".pdf"), format="pdf", bbox_inches="tight")
