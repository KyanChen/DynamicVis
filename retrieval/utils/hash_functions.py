
import torch
import numpy as np
from lshashpy3 import LSHash


def get_hash(embedding, method='trivial', *args, **kwargs):
    """
    Convert embedding or list of embeddings to hash codes using the defined method.
    """
    # Check for type
    if not isinstance(embedding, torch.Tensor):
        if isinstance(embedding, list):
            # Iterate over list of embeddings
            return [get_hash(e, method, *args, **kwargs) for e in embedding]
        else:
            raise TypeError

    # Create hash codes based on method
    if method == 'trivial':
        return trivial_hash(embedding, *args, **kwargs)
    elif method == 'lsh':
        return lshash(embedding, *args, **kwargs)
    elif method == 'none':
        return embedding
    else:
        raise NotImplementedError

def trivial_hash(embedding: torch.Tensor, length: str = 64, threshold: float = 0., seed=None):
    """
    Creates a trivial binary hash by averaging multiple embedding dimensions and binarization with a threshold.

    :param embedding: torch tensor with shape [samples, embedding]
    :param length: hash length (must divide the embedding length without a rest)
    :param threshold: value to binarize float embedding
    :return: binary hash codes as int (0 and 1) with shape [samples, hash]
    """
    assert embedding.size(-1) % length == 0, \
        f"Cannot create hash with length {length} with embedding dim {embedding.size(-1)}"
    resize_factor = int(embedding.size(-1) / length)
    binary_hash = embedding.reshape([-1, resize_factor, length]).mean(dim=1) > threshold
    return binary_hash.int()


def lshash(embedding: torch.Tensor, length: str = 64, store: str = None, seed: int = 42):
    """
    Creates a binary hash by applying LSH from the paper:
    Tang, Y. K., Mao, X. L., Hao, Y. J., Xu, C., & Huang, H. (2017).
    Locality-sensitive hashing for finding nearest neighbors in probability distributions.
    In Social Media Processing: 6th National Conference, SMP 2017, Beijing, China, September 14-17, 2017,
    Proceedings (pp. 3-15). Springer Singapore.

    Using the implementation from https://github.com/loretoparisi/lshash.

    :param embedding: torch tensor with shape [samples, embedding]
    :param length: hash length
    :param store: hash table file name (optional)
    :param seed: seed for reproducibility. Leads to similar hash codes with multiple calls.
    :return: binary hash codes as int (0 and 1) with shape [samples, hash]
    """
    # Init Locality-Sensitive Hashing
    np.random.seed(seed)
    lsh = LSHash(hash_size=length, input_dim=embedding.size(-1), num_hashtables=1,
                 hashtable_filename=store, overwrite=True)

    # Generate hashes for each embedding
    hashes = []
    for e in embedding:
        h = lsh.index(e.tolist(), extra_data=None)
        hashes.append(list(map(int, h[0])))

    return torch.tensor(hashes).int()
