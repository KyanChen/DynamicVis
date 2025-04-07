
import os
import torch
import time
import tqdm
import numpy as np
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# Init database
connections.connect("default", host="localhost", port="19530")


def run_experiment(queries, database, data_type='float', length=768):
    if data_type == 'float':
        dtype = DataType.FLOAT_VECTOR
        index = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "L2"}
        search_params = {"metric_type": "L2"}  # "params": {"nprobe": 10},
    elif data_type == 'bool':
        dtype = DataType.BINARY_VECTOR
        index = {"index_type": "BIN_IVF_FLAT", "params": {"nlist": 128}, "metric_type": "HAMMING"}
        search_params = {"metric_type": "HAMMING"}
    else:
        raise NotImplementedError

    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=dtype, dim=length)
    ]
    schema = CollectionSchema(fields, "test speed")
    try:
        collection = Collection("igarss", schema)
    except:
        utility.drop_collection("igarss")
        collection = Collection("igarss", schema)
        pass

    # Register data
    # Iterate over database to avoid hitting the max entries threshold
    step_size = 10000
    for i in tqdm.tqdm(range(0, len(database), step_size), desc='Upload images'):
        m = min(len(database), i + 10000)
        entities = [
            [i for i in range(i, m)],
            database[i:m]
        ]
        insert_result = collection.insert(entities)
    collection.flush()  

    # Create index
    collection.create_index("vector", index)

    collection.load()

    # Run retrieval test

    time_start = time.time()
    for n, query in enumerate(queries):
        result = collection.search([query], "vector", search_params, limit=20, output_fields=["pk"])
        if (n+1) % 1000 == 0:
            print(f'Average retrieval time after {n+1} samples: {(time.time() - time_start) / (n+1):.4f} s/query')

    # Drop database
    utility.drop_collection("igarss")


def main():    
    output_dir = 'output/embeddings/PrithviViT/BigEarthNet'

    print('\nLoad Binary hash codes with length 32')
    # Load hash codes
    queries = torch.load(os.path.join(output_dir, 'val/hash_codes.pt')).numpy()[:, :32]
    database = torch.load(os.path.join(output_dir, 'test/hash_codes.pt')).numpy()[:, :32]
    # Create binary vectors
    queries = [bytes(q) for q in np.packbits(queries, axis=-1)]
    database = [bytes(d) for d in np.packbits(database, axis=-1)]

    print('Experiment with 10K data')
    run_experiment(queries[:1000], database[:10000], data_type='bool', length=32)
    print('Experiment with 50K data')
    run_experiment(queries[:1000], database[:50000], data_type='bool', length=32)
    print('Experiment with 100K data')
    run_experiment(queries[:1000], database[:100000], data_type='bool', length=32)

    # Load embeddings
    print('\nLoad float embeddings with length 768')
    queries = torch.load(os.path.join(output_dir, 'val/embeddings.pt')).numpy()
    database = torch.load(os.path.join(output_dir, 'test/embeddings.pt')).numpy()

    print('Experiment with 10K data')
    run_experiment(queries[:1000], database[:10000], data_type='float', length=768)
    print('Experiment with 50K data')
    run_experiment(queries[:1000], database[:50000], data_type='float', length=768)
    print('Experiment with 100K data')
    run_experiment(queries[:1000], database[:100000], data_type='float', length=768)

    print('\nUse binary embeddings with length 768')
    queries = (queries > 0).to(int)
    database = (database > 0).to(int)
    queries = [bytes(q) for q in np.packbits(queries, axis=-1)]
    database = [bytes(d) for d in np.packbits(database, axis=-1)]

    print('Experiment with 10K data')
    run_experiment(queries[:1000], database[:10000], data_type='bool', length=768)
    print('Experiment with 50K data')
    run_experiment(queries[:1000], database[:50000], data_type='bool', length=768)
    print('Experiment with 100K data')
    run_experiment(queries[:1000], database[:100000], data_type='bool', length=768)


if __name__ == '__main__':
    main()
