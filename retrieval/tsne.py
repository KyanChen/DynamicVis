
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from utils.hash_functions import get_hash

os.environ['OUTPUT_PATH'] = '/Users/chenkeyan/Documents/work_dirs/embeddings_bak'

def main(args):
    # Init embeddings with shape [sample, embedding]
    output_path = os.getenv('OUTPUT_PATH', os.path.join('output', 'embeddings'))
    test_embeddings = torch.load(os.path.join(output_path, args.model, args.dataset, 'test', 'embeddings.pt'),
                                 map_location='cpu')
    # Multi-labels with shape [sample, label]
    test_labels = torch.load(os.path.join(output_path, args.model, args.dataset, 'test', 'labels.pt'),
                             map_location='cpu')

    # np.random.seed(42)
    binary_emb = get_hash(test_embeddings, 'trivial', length=768)
    trivial_hash = get_hash(test_embeddings, 'trivial', length=args.length)
    ls_hash = get_hash(test_embeddings, 'lsh', length=args.length)

    # Select a subset to avoid computation costs
    num_samples = 1000
    indices = np.random.choice(len(test_embeddings), min(num_samples, len(test_embeddings)), replace=False)
    sampled_binary_emb = binary_emb[indices]
    sampled_embeddings = test_embeddings[indices]
    sampled_trivial_hash = trivial_hash[indices]
    sampled_ls_hash = ls_hash[indices]
    sampled_labels = test_labels[indices].nonzero()[:, 1]
    assert len(sampled_labels) == len(indices), "t-SNE plot is not working with multi-label datasets."

    for i, (name, vector) in enumerate([('embedding', sampled_embeddings),
                                        ('binary', sampled_binary_emb),
                                        ('lsh', sampled_ls_hash),
                                        ('trivial', sampled_trivial_hash)]):
        # Compute t-SNE embeddings
        tsne = TSNE(n_components=2, random_state=42, max_iter=10000)
        tsne_results = tsne.fit_transform(vector)

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=sampled_labels, cmap='Dark2', alpha=0.9)
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        if os.path.isdir(args.output_dir):
            output_file = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_tsne_{name}_{args.length}.pdf")
        else:
            output_file = args.output_dir

        plt.savefig(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default='/Users/chenkeyan/Documents/work_dirs/embeddings_bak/figures',
                        help='Path to output dir')
    parser.add_argument('-d', '--dataset', type=str, default='ForestNet4RGB512')  #ForestNetRGB512 ForestNet4RGB512 ForestNetRGB224 ForestNet4RGB224
    parser.add_argument('-m', '--model', type=str, default='DynamicVisBackbone_b')  # DynamicVisBackbone_b  RGB_ViT
    parser.add_argument('--length', type=int, default=32)
    args = parser.parse_args()

    main(args)
