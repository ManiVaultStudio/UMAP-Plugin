# UMAP Plugin  [![Actions Status](https://github.com/ManiVaultStudio/UMAP-Plugin/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/ManiVaultStudio/UMAP-Plugin/actions)

UMAP Analysis plugin for the [ManiVault](https://github.com/ManiVaultStudio/core) visual analytics framework based on the [libscran/umappp](https://github.com/libscran/umappp) library.

Clone the repo, all dependencies will be downloaded during CMake configuration:
```
git clone https://github.com/ManiVaultStudio/UMAP-Plugin.git
```

<p align="center">
  <img src="https://github.com/ManiVaultStudio/UMAP-Plugin/assets/58806453/e8541f15-2dbd-44b6-90a6-fe608e19b076" alt="UMAP embedding of MNIST data" align="middle" width="35%">
  <img src="https://github.com/ManiVaultStudio/UMAP-Plugin/assets/58806453/d7527b47-e196-4b62-bb0b-cb22f2fc1132" alt="UMAP embedding of MNIST data" align="middle" width="43%">  </br>
  Left: UMAP embedding of 10k MNIST test data. Right: UMAP embedding of Indian Pines data in 3 dimensions, with (top) showing x and y and (bottom) showing y and z embedding dimensions as well as (right) a re-coloring of the Indian Pines image based on the 3d embedding space interpreted as HSV colorspace.
</p>

## Settings

Main settings:
- `Epochs`:  Number of epochs for the gradient descent, i.e., optimization iterations. Larger values improve accuracy at the cost of computational work. For datasets with no more than 10000 observations, the number of epochs is set to 500. For larger datasets, the number of epochs decreases from 500 according to the number of cells beyond 10000, to a lower limit of 200. This choice aims to reduce computational work for very large datasets. 
- `Initialization`: How should the initial coordinates of the embedding be obtained?
  - `SPECTRAL`: attempts initialization based on spectral decomposition of the graph Laplacian. If that fails, we fall back to random draws from a normal distribution.
  - `RANDOM`: fills the embedding with random draws from a normal distribution.
- `Embedding dimensions`: Number of output dimensions.

Advanced settings:
- `local_connectivity`: The number of nearest neighbors that are assumed to be always connected, with maximum membership confidence. Larger values increase the connectivity of the embedding and reduce the focus on local structure.
- `bandwidth`: Effective bandwidth of the kernel when converting the distance to a neighbor into a fuzzy set membership confidence. Larger values reduce the decay in confidence with respect to distance, increasing connectivity and favoring global structure. 
- `mix_ratio`: This symmetrizes the sets by ensuring that the confidence of $A$ belonging to $B$'s set is the same as the confidence of $B$ belonging to $A$'s set. A mixing ratio of 1 will take the union of confidences, a ratio of 0 will take the intersection, and intermediate values will interpolate between them. Larger values (up to 1) favor connectivity and more global structure.
- `spread`: Scale of the coordinates of the final low-dimensional embedding.
- `min_dist`: Minimum distance between observations in the final low-dimensional embedding. Smaller values will increase local clustering while larger values favors a more even distribution. This is interpreted relative to the spread of points in `spread`.
- `negative_sample_rate`: Rate of sampling negative observations to compute repulsive forces. This is interpreted with respect to the number of neighbors with attractive forces, i.e., for each attractive interaction, `n` negative samples are taken for repulsive interactions. Smaller values can improve the speed of convergence but at the cost of stability.
- `a`: Positive value for the $a$ parameter for the fuzzy set membership strength calculations. Larger values yield a sharper decay in membership strength with increasing distance between observations. If this or $b$ is set to zero, a suitable value for this parameter is automatically determined from the values provided to `spread` and `min_dist`.
- `b`: Value in $(0, 1)$ for the $b$ parameter for the fuzzy set membership strength calculations. Larger values yield an earlier decay in membership strength with increasing distance between observations. If this or $a$ is set to zero, a suitable value for this parameter is automatically determined from the values provided to `spread` and `min_dist`.
- `repulsion_strength`: Modifier for the repulsive force. Larger values increase repulsion and favor local structure.
- `learning_rate`: Initial learning rate used in the gradient descent. Larger values can improve the speed of convergence but at the cost of stability.
- `seed`: Seed to use for the Mersenne Twister when sampling negative observations.

knn settings:
- `Number knn`:  Number of neighbors to use to define the fuzzy sets. Larger values improve connectivity and favor preservation of global structure, at the cost of increased computational work.
- `Multithreading`: Whether to use all available threads for knn computation. This will be faster while using more memory.
- `Algorithm`: Type of approximated knn algorithm/library to be used. Either [Annoy](https://github.com/spotify/annoy) or [HNSW](https://github.com/nmslib/hnswlib/).
  - `Annoy Trees` & `Annoy Checks`: correspond to `n_trees` and `search_k`, see the respective [Annoy docs](https://github.com/spotify/annoy?tab=readme-ov-file#tradeoffs)
  - `HNSW M` & `HNSW ef`: are detailed in the respective [HNSW docs](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md#hnsw-algorithm-parameters)
- `Metric`:
  - Euclidean (L2): `sum((Ai-Bi)^2)`
  - Cosine: `1.0 - sum(Ai*Bi)`
  - Inner: `1.0 - sum(Ai*Bi) / sqrt(sum(Ai*Ai) * sum(Bi*Bi))`
  - Correlation: `1.0 - corr(A, B)`. Based on [Pearson's correlation coefficient](https://en.wikipedia.org/wiki/Correlation) between data vectors. See the correlation distance in [umap-learn](https://github.com/lmcinnes/umap/blob/15e55bb6a1ca23b8d6040d9d6184a7ae98325ace/umap/distances.py#L598). Only available with HNSW. (Technically not a metric.)

## References
- [libscran/umappp](https://github.com/libscran/umappp): Aaron Lun, BSD 2-Clause License
- UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, McInnes L, Healy J, Melville J (2020), [arxiv: 1802.03426](https://arxiv.org/abs/1802.03426)
