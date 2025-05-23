import os
from torch.utils import data
from dataset.phaseUnwrapping import PhaseUnwrappingDataSet,PhaseUnwrappingValDataSet,PhaseUnwrappingTestDataSet

def build_dataset_train(dataRootDir,dataset, input_size, batch_size, random_mirror, num_workers):

    if dataset == 'phaseUnwrapping':
        data_dir = os.path.join(dataRootDir, dataset)
        train_data_list = os.path.join(data_dir, 'train.txt')
        val_data_list = os.path.join(data_dir, 'val.txt')

        trainLoader = data.DataLoader(
            PhaseUnwrappingDataSet(data_dir, train_data_list, crop_size=input_size,
                              mirror=random_mirror),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = data.DataLoader(
            PhaseUnwrappingValDataSet(data_dir, val_data_list),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
            drop_last=True)

        datas = None
        return datas, trainLoader, valLoader
    else:
        raise NotImplementedError(
            "not supports the dataset: %s" % dataset)

def build_dataset_test(dataRootDir, dataset, num_workers, none_gt=False, list_path_override=None, input_size=(256,256)):
    if dataset == 'phaseUnwrapping':
        data_dir = os.path.join(dataRootDir, dataset)
        
        # Use list_path_override if provided, otherwise default to test.txt
        data_list_file = list_path_override if list_path_override else os.path.join(data_dir, 'test.txt')

        if not os.path.isfile(data_list_file):
            raise FileNotFoundError(f"Data list file not found: {data_list_file}")

        if none_gt:
            # PhaseUnwrappingTestDataSet might also need input_size if its processing depends on it.
            # For now, assuming it doesn't or is simpler. If issues arise, it would need similar modification.
            print(f"Loading TestDataSet with list: {data_list_file}")
            testLoader = data.DataLoader(
                PhaseUnwrappingTestDataSet(data_dir, data_list_file), # Ensure this class can handle variable input if needed
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        else:
            # Use PhaseUnwrappingDataSet for consistency and input_size handling
            # It expects crop_size and input_size. We'll use the provided input_size for both.
            print(f"Loading PhaseUnwrappingDataSet (as test/val) with list: {data_list_file}, input_size: {input_size}")
            testLoader = data.DataLoader(
                PhaseUnwrappingDataSet(data_dir, data_list_file, crop_size=input_size, input_size=input_size, mirror=False), # Use PhaseUnwrappingDataSet
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        datas = None
        return datas, testLoader

    else:
        raise NotImplementedError(
                "not supports the dataset: %s" % dataset)