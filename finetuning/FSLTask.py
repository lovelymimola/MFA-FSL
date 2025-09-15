import os
import pickle
import numpy as np
import torch
# from tqdm import tqdm

# ========================================================
#   Usefull paths
_datasetFeaturesFiles = {"Wi-Fi": "../feature/novel1_features.plk",}
_test1_datasetFeaturesFiles = {"Wi-Fi": "../feature/query1_features.plk",}
_test2_datasetFeaturesFiles = {"Wi-Fi": "../feature/query2_features.plk",}
_cacheDir = "../cache"
_maxRuns = 10000
_min_examples = -1

# ========================================================
#   Module internal functions and variables

_randStates = None
_rsCfg = None


def _load_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
        labels = [np.full(shape=len(data[key]), fill_value=key) for key in data]
        data = [features for key in data for features in data[key]]
        dataset = dict()
        dataset['data'] = torch.FloatTensor(np.stack(data, axis=0))
        dataset['labels'] = torch.LongTensor(np.concatenate(labels))
        return dataset


# =========================================================
#    Callable variables and functions from outside the module

data = None
labels = None
dsName = None


def loadDataSet(dsname):
    if dsname not in _datasetFeaturesFiles:
        raise NameError('Unknwown dataset: {}'.format(dsname))

    global dsName, data, labels, test1_data, test1_labels, test2_data, test2_label, _randStates, _rsCfg, _min_examples, _test_min_examples
    dsName = dsname
    _randStates = None
    _rsCfg = None

    # Loading data from files on computer
    # home = expanduser("~")
    dataset = _load_pickle(_datasetFeaturesFiles[dsname])
    test1_dataset = _load_pickle(_test1_datasetFeaturesFiles[dsname])
    test2_dataset = _load_pickle(_test2_datasetFeaturesFiles[dsname])

    # Computing the number of items per class in the dataset
    _min_examples = dataset["labels"].shape[0]
    for i in range(dataset["labels"].shape[0]):
        if torch.where(dataset["labels"] == dataset["labels"][i])[0].shape[0] > 0:
            _min_examples = min(_min_examples, torch.where(
                dataset["labels"] == dataset["labels"][i])[0].shape[0])
    print("Guaranteed number of items per class in train dataset: {:d}\n".format(_min_examples))

    _test1_min_examples = test1_dataset["labels"].shape[0]
    for i in range(test1_dataset["labels"].shape[0]):
        if torch.where(test1_dataset["labels"] == test1_dataset["labels"][i])[0].shape[0] > 0:
            _test1_min_examples = min(_test1_min_examples, torch.where(
                test1_dataset["labels"] == test1_dataset["labels"][i])[0].shape[0])
    print("Guaranteed number of items per class in test1 dataset: {:d}\n".format(_test1_min_examples))

    _test2_min_examples = test2_dataset["labels"].shape[0]
    for i in range(test2_dataset["labels"].shape[0]):
        if torch.where(test2_dataset["labels"] == test2_dataset["labels"][i])[0].shape[0] > 0:
            _test2_min_examples = min(_test2_min_examples, torch.where(
                test2_dataset["labels"] == test2_dataset["labels"][i])[0].shape[0])
    print("Guaranteed number of items per class in test2 dataset: {:d}\n".format(_test2_min_examples))

    # Generating data tensors of train
    data = torch.zeros((0, _min_examples, dataset["data"].shape[1]))
    for label in range(0, len(np.unique(dataset["labels"]))):
        indices = torch.where(dataset["labels"] == label)[0]
        data = torch.cat([data, dataset["data"][indices, :][:_min_examples].view(1, _min_examples, -1)], dim=0)
    print("Total of {:d} classes, {:d} elements each, with dimension {:d}\n".format(data.shape[0], data.shape[1], data.shape[2]))

    # Generating data tensors of test1
    test1_data = torch.zeros((0, _test1_min_examples, test1_dataset["data"].shape[1]))
    for label in range(0, len(np.unique(test1_dataset["labels"]))):
        indices = torch.where(test1_dataset["labels"] == label)[0]
        test1_data = torch.cat([test1_data, test1_dataset["data"][indices, :][:_test1_min_examples].view(1, _test1_min_examples, -1)], dim=0)
    print("Total of {:d} classes, {:d} elements each, with dimension {:d}\n".format(test1_data.shape[0], test1_data.shape[1], test1_data.shape[2]))

    # Generating data tensors of test2
    test2_data = torch.zeros((0, _test2_min_examples, test2_dataset["data"].shape[1]))
    for label in range(0, len(np.unique(test2_dataset["labels"]))):
        indices = torch.where(test2_dataset["labels"] == label)[0]
        test2_data = torch.cat([test2_data, test2_dataset["data"][indices, :][:_test2_min_examples].view(1, _test2_min_examples, -1)], dim=0)
    print("Total of {:d} classes, {:d} elements each, with dimension {:d}\n".format(test2_data.shape[0], test2_data.shape[1], test2_data.shape[2]))

def GenerateRun(iRun, cfg, regenRState=False, generate=True):
    global _randStates, data, _min_examples
    if not regenRState:
        np.random.set_state(_randStates[iRun])

    shuffle_indices = np.arange(_min_examples)
    dataset = None
    if generate:
        dataset = torch.zeros((cfg['ways'], cfg['shot'], data.shape[2]))

    for i in range(cfg['ways']):
        shuffle_indices = np.random.permutation(shuffle_indices)
        if generate:
            dataset[i] = data[i, shuffle_indices,:][:cfg['shot']]

    return dataset

def GenerateTest(cfg):
    global test1_data, test2_data
    data1 = []
    labels1 = []
    for i in range(cfg['ways']):
        data1.append(test1_data[i])
        labels1.append(np.ones((len(test1_data[i]),))*i)

    data2 = []
    labels2 = []
    for i in range(cfg['ways']):
        data2.append(test2_data[i])
        labels2.append(np.ones((len(test2_data[i]),))*i)

    return np.concatenate(data1), np.concatenate(labels1).astype(int), np.concatenate(data2), np.concatenate(labels2).astype(int)

def ClassesInRun(iRun, cfg):
    global _randStates, data
    np.random.set_state(_randStates[iRun])

    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    return classes


def setRandomStates(cfg):
    global _randStates, _maxRuns, _rsCfg
    if _rsCfg == cfg:
        return

    rsFile = os.path.join(_cacheDir, "RandStates_{}_s{}_w{}".format(
        dsName, cfg['shot'], cfg['ways']))
    if not os.path.exists(rsFile):
        print("{} does not exist, regenerating it...".format(rsFile))
        np.random.seed(0)
        _randStates = []
        for iRun in range(_maxRuns):
            _randStates.append(np.random.get_state())
            GenerateRun(iRun, cfg, regenRState=True, generate=False)
        torch.save(_randStates, rsFile)
    else:
        print("reloading random states from file....")
        _randStates = torch.load(rsFile)
    _rsCfg = cfg


def GenerateRunSet(start=None, end=None, cfg=None):
    global dataset, test_dataset, _maxRuns
    if start is None:
        start = 0
    if end is None:
        end = _maxRuns
    if cfg is None:
        cfg = {"shot": 1, "ways": 5}

    setRandomStates(cfg)
    print("generating task from {} to {}".format(start, end))

    dataset = torch.zeros((end-start, cfg['ways'], cfg['shot'], data.shape[2]))
    for iRun in range(end-start):
        dataset[iRun] = GenerateRun(start+iRun, cfg)

    test1_data, test1_labels, test2_data, test2_labels = GenerateTest(cfg)

    return dataset, test1_data, test1_labels, test2_data, test2_labels


# define a main code to test this module
if __name__ == "__main__":

    print("Testing Task loader for Few Shot Learning")
    loadDataSet('miniimagenet')

    cfg = {"shot": 1, "ways": 5, "queries": 15}
    setRandomStates(cfg)

    run10 = GenerateRun(10, cfg)
    print("First call:", run10[:2, :2, :2])

    run10 = GenerateRun(10, cfg)
    print("Second call:", run10[:2, :2, :2])

    ds = GenerateRunSet(start=2, end=12, cfg=cfg)
    print("Third call:", ds[8, :2, :2, :2])
    print(ds.size())