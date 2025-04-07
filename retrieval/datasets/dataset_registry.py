
DATASET_REGISTRY = {}

def register_dataset(dataset_name, dataset_fn):
    DATASET_REGISTRY[dataset_name] = dataset_fn
