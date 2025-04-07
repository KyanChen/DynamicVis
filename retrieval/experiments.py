
import os
import pandas as pd
import tqdm
import argparse
import glob
import logging
import torch
import numpy as np
import torch.nn.functional as F
from datetime import datetime, timezone
from torchmetrics.retrieval import RetrievalMAP, RetrievalNormalizedDCG, RetrievalPrecision
from utils.hash_functions import get_hash

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_similarity(queries, database, distance='hamming'):
    if distance == 'hamming':
        return -torch.cdist(queries.float(), database.float(), p=1)
    elif distance == 'euclidean':
        return -torch.cdist(queries.float(), database.float(), p=2)
    elif distance == 'cosine':
        return F.cosine_similarity(queries.float().unsqueeze(1), database.float().unsqueeze(0), dim=-1)
    elif distance == 'dotproduct':
        return torch.einsum('ab,cb->ac', queries, database).float()
    else:
        raise NotImplementedError


def run_experiment(val_embeddings, test_embeddings, val_labels, test_labels, distance_function, hash_method,
                   hash_length, k, match, seed=None):
    # Run multiple seeds for lsh hash
    if hash_method == 'lsh' and seed is None:
        mean_average_precision = []
        precision = []
        # Run experiment 5 times with different seeds
        for s in range(1, 6):
            map, p = run_experiment(val_embeddings, test_embeddings, val_labels, test_labels, distance_function,
                                    hash_method, hash_length, k, match, seed=s*42)
            mean_average_precision.append(map)
            precision.append(p)
        # Compute mean and 95% confidence interval
        map_mean = np.mean(mean_average_precision) * 100
        map_ci = np.std(mean_average_precision, ddof=1) / np.sqrt(len(mean_average_precision)) * 1.96  * 100
        precision_mean = np.mean(precision) * 100
        precision_ci = np.std(precision, ddof=1) / np.sqrt(len(precision)) * 1.96 * 100

        return f'{map_mean:.2f} ± {map_ci:.2f}', f'{precision_mean:.2f} ± {precision_ci:.2f}'

    logging.info(f'Top {k} experiment with {hash_method} hash, length {hash_length}, and {distance_function} distance.')
    val_hash, test_hash = get_hash([val_embeddings, test_embeddings], hash_method,
                                     length=hash_length, seed=seed)

    # Init metrics
    map_metric = RetrievalMAP(top_k=k)
    precision_metric = RetrievalPrecision(top_k=k)
    # Passing only the top k results to ndcg results in different values. Skipping this metric.
    # ndcg_metric = RetrievalNormalizedDCG(top_k=k)

    # Iterate over results to avoid OOM errors
    step_size = 100
    for i in tqdm.tqdm(range(0, len(val_hash), step_size), desc='Compute retrival results'):
        # Get similarity values
        similarity = get_similarity(val_hash[i:i+step_size], test_hash, distance=distance_function)
        similarity = similarity.to(device)

        target = val_labels[i:i+step_size].unsqueeze(1).repeat(1, len(test_labels), 1)
        if match == 'any':
            # Count any positive overlap for reported experiments
            target = target.logical_and(target == test_labels).any(dim=-1)
        elif match == 'exact':
            target = (target == test_labels).all(dim=-1)

        # Select top k results to reduce computation time (no influence on mAP metric)
        assert k < similarity.shape[-1]
        ranking = similarity.topk(k, sorted=True, dim=-1)[1]
        similarity_k = torch.gather(similarity, 1, ranking)
        target_k = torch.gather(target, 1, ranking)
        indexes_k = torch.arange(i, i+len(ranking)).unsqueeze(1).repeat(1, k)

        # Add samples to retrieval metrics
        map_metric.update(similarity_k, target_k, indexes_k)
        precision_metric.update(similarity_k, target_k, indexes_k)

    # Compute metrics
    mean_average_precision = map_metric.compute().item()
    precision = precision_metric.compute().item()

    # Log results
    logging.debug(f'Retrival mAP@{k}: {mean_average_precision:.4f}')
    logging.debug(f'Retrival Precision@{k}: {precision:.4f}')

    return mean_average_precision, precision


def main(args):
    output_path = os.getenv('OUTPUT_PATH', os.path.join('work_dirs', 'embeddings'))
    # expects results in the structure <output_path>/<model>/<dataset>/<split>/embeddings.pt
    val_embedding_files = sorted(glob.glob(os.path.join(output_path, args.folder_pattern, 'val', 'embeddings.pt')))

    results = pd.DataFrame([], columns=['Dataset', 'Model', 'Match', 'Distance', 'Hash method', 'Hash length',
                                        'Top K', 'mAP@K', 'Precision@K'])
    results_path = args.results_file or os.path.join('work_dirs', 'results.csv')

    logging.info(f'Found {len(val_embedding_files)} model+dataset combinations.')

    for val_embedding_file in val_embedding_files:
        # Embeddings with shape [sample, embedding]
        val_embeddings = torch.load(val_embedding_file, map_location=device)
        test_embeddings = torch.load(val_embedding_file.replace('val', 'test'), map_location=device)
        # Multi-labels with shape [sample, label]
        val_labels = torch.load(val_embedding_file.replace('embeddings.pt', 'labels.pt'),
                                  map_location=device)
        test_labels = torch.load(val_embedding_file.replace('embeddings.pt', 'labels.pt')
                                 .replace('val', 'test'), map_location=device)
        model, dataset = val_embedding_file.split('/')[-4:-2]
        logging.info(f'Embedding and labels loaded for {model} and {dataset}.')

        for match in args.match.split(','):
            for distance_function in args.distance_function.split(','):
                for hash_method in args.hash_method.split(','):
                    hash_lengths = args.hash_length.split(',') if hash_method != 'none' else [val_embeddings.size(-1)]
                    for hash_length in hash_lengths:
                        hash_length = int(hash_length)
                        for k in args.top_k.split(','):
                            k = int(k)
                            metrics = run_experiment(val_embeddings, test_embeddings, val_labels, test_labels,
                                                     distance_function, hash_method, hash_length, k, match)
                            results.loc[len(results)] = [dataset, model, match, distance_function, hash_method,
                                                         hash_length, k, *metrics]

                            # Save results
                            results.to_csv(results_path)
                            logging.debug(f'Saved metrics in {results_path}')
    logging.info(f'Finished experiments. Results are saved in {results_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Path to output dir for embeddings and labels '
                             '(default: output/embeddings/<dataset>/<model>)')
    parser.add_argument('--folder_pattern', type=str, default='*/*',
                        help='Pattern for output dir, default: assumes <model>/<dataset>/')
    parser.add_argument('--match', type=str, default='any', help='Select match type (any, exact)')
    parser.add_argument('--distance_function', type=str, default='hamming',
                        help='Distance function (hamming, euclidean, cosine, dotproduct)')
    parser.add_argument('--hash_method', type=str, default='trivial,lsh,none',
                        help='Method (trivial, lsh, none)')
    parser.add_argument('--hash_length', type=str, default='32,768', help='Hash length')
    parser.add_argument('--top_k', type=str, default='20', help='Number of retrieved samples')
    parser.add_argument('--results_file', type=str, default=None, help='Results file')
    parser.add_argument('--log_level', type=str, default='INFO',
                        help='Log level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--log_file', type=str, default=None, help='Log file')
    args = parser.parse_args()

    # init logger
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%MZ")
    log_file = args.log_file or f"logs/{current_time}_experiments.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=args.log_level.upper(),
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        main(args)
    except Exception as e:
        # log potential error
        logging.error(f'{type(e)}: {e}')
        raise e
