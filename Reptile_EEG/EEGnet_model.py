import copy
import torch.nn as nn
import torch
import torch.nn.functional as F
import multiprocessing


class inEEG_Net(nn.Module):
    """
    References
    ----------
    [EEGNet4] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon,
       S. M., Hung, C. P., & Lance, B. J. (2018).
       EEGNet: A Compact Convolutional Network for EEG-based
       Brain-Computer Interfaces.
       arXiv preprint arXiv:1611.08024.
    """

    def __init__(self, num_classes=4, dropout1=0.52, dropout2=0.36, f1=16, sampling_rate=250, num_channels=62,
                 depth_multiplier=6, time_of_interest=500, time_points=625, lowpass=50, point_reducer=5):
        super(inEEG_Net, self).__init__()
        self.defined_params = {
            'model_type': 'inEEG_Net',
            'num_classes': num_classes,
            'dropout1': dropout1,
            'dropout2': dropout2,
            'f1': f1,
            'sampling_rate': sampling_rate,
            'num_channels': num_channels,
            'depth_multiplier': depth_multiplier,
            'time_of_interest': time_of_interest,
            'time_points': time_points,
            'lowpass': lowpass,
            'point_reducer': point_reducer
        }
        self.conv_features = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, int(sampling_rate/2)), padding=(0, int(sampling_rate/4)),
                      stride=1, bias=False),
            nn.BatchNorm2d(f1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(f1, f1*depth_multiplier, kernel_size=(num_channels, 1), stride=(1, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1*depth_multiplier, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ELU(alpha=1.0),
            # nn.AdaptiveAvgPool2d(output_size=(1, 28)),
            nn.AvgPool2d(kernel_size=(1, int(sampling_rate/lowpass))),
            nn.Dropout(p=dropout1, inplace=False),
            nn.Conv2d(f1*depth_multiplier, f1*depth_multiplier, kernel_size=(1, int(lowpass*(time_of_interest/1000))),
                      stride=(1, 1), padding=(0, int(lowpass*(time_of_interest/2000))),
                      groups=f1*depth_multiplier, bias=False),
            nn.Conv2d(f1*depth_multiplier, f1*depth_multiplier, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ELU(alpha=1.0),
            # nn.AdaptiveAvgPool2d(output_size=(1, 7)),
            nn.AvgPool2d(kernel_size=(1, point_reducer)),
            nn.Dropout(p=dropout2, inplace=False)
        )
        self.out_features = nn.Sequential(
            nn.Flatten(),
            #nn.AdaptiveAvgPool1d(output_size=int(f1*depth_multiplier*time_points/point_reducer/sampling_rate*lowpass)),
            # nn.Linear(in_features=int(time_points/8/32*sampling_rate*f1*depth_multiplier),
            # out_features=num_classes, bias=True)
            #nn.Linear(in_features=int(f1*depth_multiplier*time_points/point_reducer/sampling_rate*lowpass),
            #          out_features=num_classes, bias=True)
            nn.Linear(in_features=self.feature_dim(), out_features=num_classes, bias=True)
        )


    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.defined_params['num_channels'], self.defined_params['time_points'])
            mock_eeg = self.conv_features(mock_eeg)
        return self.defined_params['f1'] * self.defined_params['depth_multiplier'] * mock_eeg.shape[3]

    def forward(self, x):
        x = self.conv_features(x)
        # print(x.shape)
        x = self.out_features(x)
        return x


def model_from_params(params: dict):
    """
    :param params: - params: dict with model params in this lib all models have model.defined_params, from which they can be
    recreated with untrained weights
    :return: - model which has defined params and type
    """
    if params['model_type'] == 'inEEG_net':
        model = inEEG_Net(num_classes=params['num_classes'], dropout1=params['dropout1'], dropout2=params['dropout2'],
                          f1=params['f1'], sampling_rate=params['sampling_rate'], num_channels=params['num_channels'],
                          depth_multiplier=params['depth_multiplier'], time_of_interest=params['time_of_interest'],
                          time_points=params['time_points'], lowpass=params['lowpass'],
                          point_reducer=params['point_reducer'])
    else:
        raise ValueError('model_class is not defined')
    return model


def train(model, optimizer, loss_fn, train_loader, val_loader=None, epochs=2, device="cpu", logging=True):
    best_acc = 0.0
    best_model = model.state_dict()
    for epoch in range(1, epochs+1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device=device, dtype=torch.float)
            targets = targets.type(torch.LongTensor)
            targets = targets.to(device=device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            # run['metrics/train_loss'].log(loss)
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)
        model.eval()
        num_correct = 0
        num_examples = 0
        if val_loader is not None:
            for batch in val_loader:
                inputs, targets = batch
                # inputs = inputs.view(64,1,22,-1)
                inputs = inputs.to(device=device, dtype=torch.float)
                output = model(inputs)
                targets = targets.type(torch.LongTensor)
                targets = targets.to(device=device)
                loss = loss_fn(output, targets)
                valid_loss += loss.data.item() * inputs.size(0)
                correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
            valid_loss /= len(val_loader.dataset)
            # run["training/batch/acc"].log(num_correct / num_examples)
            if (num_correct / num_examples > best_acc):
                best_model = copy.deepcopy(model)
                best_acc = num_correct / num_examples

            if logging:
                print('in process ' + str(multiprocessing.current_process().name) +
                      '\nEpoch: {}/{}, Training Loss: {:.2f}, Validation Loss: {:.2f}, '
                      'accuracy = {:.2f}'.format(epoch, epochs, training_loss,
                                                 valid_loss, num_correct / num_examples))
        else:
            if logging:
                print('in process ' + str(multiprocessing.current_process().name) +
                      '\nEpoch: {}/{}, Training Loss: {:.2f}'.format(epoch, epochs, training_loss))
    return best_model, best_acc


def single_batch_train(model, optimizer, loss_fn, batch, val_loader=None, epochs=2, device="cpu", logging=True):
    best_acc = 0.0
    best_model = model.state_dict()
    for epoch in range(1, epochs+1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        optimizer.zero_grad()
        inputs, targets = batch
        inputs = inputs.to(device=device, dtype=torch.float)
        targets = targets.type(torch.LongTensor)
        targets = targets.to(device=device)
        output = model(inputs)
        loss = loss_fn(output, targets)
        loss.backward()
        # run['metrics/train_loss'].log(loss)
        optimizer.step()
        training_loss += loss.data.item() * inputs.size(0)
        model.eval()
        num_correct = 0
        num_examples = 0
        if val_loader is not None:
            for batch in val_loader:
                inputs, targets = batch
                # inputs = inputs.view(64,1,22,-1)
                inputs = inputs.to(device=device, dtype=torch.float)
                output = model(inputs)
                targets = targets.type(torch.LongTensor)
                targets = targets.to(device=device)
                loss = loss_fn(output, targets)
                valid_loss += loss.data.item() * inputs.size(0)
                correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
            valid_loss /= len(val_loader.dataset)
            # run["training/batch/acc"].log(num_correct / num_examples)
            if (num_correct / num_examples > best_acc):
                best_model = copy.deepcopy(model)
                best_acc = num_correct / num_examples

            if logging:
                print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss, valid_loss, num_correct / num_examples))
        else:
            if logging:
                print('Epoch: {}, Training Loss: {:.2f}'.format(epoch, training_loss))
    return best_model, best_acc