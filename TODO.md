# TODO

- [X] Kmean measure
- [X] Trustworthiness measure
- [ ] other datasets than iris
- [ ] new graph building functions, weighting functions and features ??? maybe
- [X] do grid search on all current combinaions of weights, builds, features, etc.
- [ ] maybe we can try running GraphSAGE for longer or more complicated networks.. my experiments use 2-layer networks for time reasons, now it takes about 3 minutes to learn
  - layers and number of weights is adjusted by num_samples and layer_sizes parameters of embed_data (these two lists have to be of same length)
- [ ] investigate GraphSAGE curved result

- [ ] Remake the graph building to stellargraph framework directly?? No no time ...
  - networkx is used for graph construction and node2vec, other algorithms use StellarGraph, so if we don't use node2vec, but other algorithms it would be more efficient to construct the graph as Stellargraph directly and not transform it from networkx

- [ ] Start the poster
- [ ] Clean up the code (Theo)



We get this error : 
The requirements.txt doesn't work with python3.6 it has to be updated

