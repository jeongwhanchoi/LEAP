import collections as co
import numpy as np
import os
import pathlib
import sktime.utils.load_data
import torch
import urllib.request
import zipfile

from . import common


here = pathlib.Path(__file__).resolve().parent


def download():
    base_base_loc = here / 'data'
    base_loc = base_base_loc / 'UEA'
    loc = base_loc / 'Multivariate2018_ts.zip'
    if os.path.exists(loc):
        return
    if not os.path.exists(base_base_loc):
        os.mkdir(base_base_loc)
    if not os.path.exists(base_loc):
        os.mkdir(base_loc)
    urllib.request.urlretrieve('http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip',
                               str(loc))

    with zipfile.ZipFile(loc, 'r') as f:
        f.extractall(str(base_loc))


# Is this actually necessary?
def _pad(channel, maxlen):
    channel = torch.tensor(channel)
    out = torch.full((maxlen,), channel[-1])
    out[:channel.size(0)] = channel
    return out


valid_dataset_names = {'ArticularyWordRecognition',
                       'FaceDetection',
                       'NATOPS',
                       'AtrialFibrillation',
                       'FingerMovements',
                       'PEMS-SF',
                       'BasicMotions',
                       'HandMovementDirection',
                       'PenDigits',
                       'CharacterTrajectories',
                       'Handwriting',
                       'PhonemeSpectra',
                       'Cricket',
                       'Heartbeat',
                       'RacketSports',
                       'DuckDuckGeese',
                       'InsectWingbeat',
                       'SelfRegulationSCP1',
                       'EigenWorms',
                       'JapaneseVowels',
                       'SelfRegulationSCP2',
                       'Epilepsy',
                       'Libras',
                       'SpokenArabicDigits',
                       'ERing',
                       'LSST',
                       'StandWalkJump',
                       'EthanolConcentration',
                       'MotorImagery',
                       'UWaveGestureLibrary'}


def _process_data(dataset_name, missing_rate, intensity):
    
    assert dataset_name in valid_dataset_names, "Must specify a valid dataset name."

    base_filename = here / 'data' / 'UEA' / 'Multivariate_ts' / dataset_name / dataset_name
    train_X, train_y = sktime.utils.load_data.load_from_tsfile_to_dataframe(str(base_filename) + '_TRAIN.ts')
    test_X, test_y = sktime.utils.load_data.load_from_tsfile_to_dataframe(str(base_filename) + '_TEST.ts')
    train_X = train_X.to_numpy()
    test_X = test_X.to_numpy()
    X = np.concatenate((train_X, test_X), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)
    
    lengths = torch.tensor([len(Xi[0]) for Xi in X])
    final_index = lengths - 1
    maxlen = lengths.max()
    X = torch.stack([torch.stack([_pad(channel, maxlen) for channel in batch], dim=0) for batch in X], dim=0)
    X = X.transpose(-1, -2)
    times = torch.linspace(0, X.size(1) - 1, X.size(1))

    generator = torch.Generator().manual_seed(56789)
    for Xi in X:
        removed_points = torch.randperm(X.size(1), generator=generator)[:int(X.size(1) * missing_rate)].sort().values
        Xi[removed_points] = float('nan')

    targets = co.OrderedDict()
    counter = 0
    for yi in y:
        if yi not in targets:
            targets[yi] = counter
            counter += 1
    y = torch.tensor([targets[yi] for yi in y])

    (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
     test_final_index, input_channels) = common.preprocess_data(times, X, y, final_index, append_times=True,
                                                                append_intensity=intensity)

    num_classes = counter

    assert num_classes >= 2, "Have only {} classes.".format(num_classes)

    return (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
            test_final_index, num_classes, input_channels)


def get_data(dataset_name, missing_rate, device, intensity, batch_size):
    
    assert dataset_name in valid_dataset_names, "Must specify a valid dataset name."

    base_base_loc = here / 'processed_data'
    base_loc = base_base_loc / 'UEA'
    loc = base_loc / (dataset_name + str(int(missing_rate * 100)) + ('_intensity' if intensity else ''))
    if os.path.exists(loc):
        tensors = common.load_data(loc)
        times = tensors['times']
        train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d']
        val_coeffs = tensors['val_a'], tensors['val_b'], tensors['val_c'], tensors['val_d']
        test_coeffs = tensors['test_a'], tensors['test_b'], tensors['test_c'], tensors['test_d']
        train_y = tensors['train_y']
        val_y = tensors['val_y']
        test_y = tensors['test_y']
        train_final_index = tensors['train_final_index']
        val_final_index = tensors['val_final_index']
        test_final_index = tensors['test_final_index']
        num_classes = int(tensors['num_classes'])
        input_channels = int(tensors['input_channels'])
        
    else:
        download()
        if not os.path.exists(base_base_loc):
            os.mkdir(base_base_loc)
        if not os.path.exists(base_loc):
            os.mkdir(base_loc)
        if not os.path.exists(loc):
            os.mkdir(loc)
        (times, train_coeffs, val_coeffs, test_coeffs, train_y, val_y, test_y, train_final_index, val_final_index,
         test_final_index, num_classes, input_channels) = _process_data(dataset_name, missing_rate, intensity)
        common.save_data(loc,
                         times=times,
                         train_a=train_coeffs[0], train_b=train_coeffs[1], train_c=train_coeffs[2],
                         train_d=train_coeffs[3],
                         val_a=val_coeffs[0], val_b=val_coeffs[1], val_c=val_coeffs[2], val_d=val_coeffs[3],
                         test_a=test_coeffs[0], test_b=test_coeffs[1], test_c=test_coeffs[2], test_d=test_coeffs[3],
                         train_y=train_y, val_y=val_y, test_y=test_y, train_final_index=train_final_index,
                         val_final_index=val_final_index, test_final_index=test_final_index,
                         num_classes=torch.as_tensor(num_classes), input_channels=torch.as_tensor(input_channels))

    times, train_dataloader, val_dataloader, test_dataloader = common.wrap_data(times, train_coeffs, val_coeffs,
                                                                                test_coeffs, train_y, val_y, test_y,
                                                                                train_final_index, val_final_index,
                                                                                test_final_index, device,
                                                                                num_workers=0, batch_size=batch_size)

    return times, train_dataloader, val_dataloader, test_dataloader, num_classes, input_channels
