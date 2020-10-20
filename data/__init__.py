from data_structures.data_manager import DataManager
from data_structures.data_reader import DataReader

data_set_names = [
    'iris',
    "synthetic",
]

data_set_is_classification = {
    'iris': True,
    "synthetic": True,
}


original_data_readers = {
    'iris':  DataReader(-1),
    'synthetic':  DataReader(-1),
}

data_paths = {x: f'{x}_data/{x}.data' for x in original_data_readers.keys()}

original_data_sets = DataManager(original_data_readers, data_paths, 'original')

preprocessed_data_readers = {
    'iris':  DataReader(),
    'synthetic':  DataReader(),

}

preprocessed_data_sets = DataManager(preprocessed_data_readers, data_paths, 'preprocessed')


k_means_data_sets = DataManager(data_paths=data_paths, data_folder='k_means')

dbscan_data_sets = DataManager(data_paths=data_paths, data_folder='dbscan')





data_managers = {
    'original': original_data_sets,
    'preprocessed': preprocessed_data_sets,
    'k_means': k_means_data_sets,
    'dbscan': dbscan_data_sets,
    

}
