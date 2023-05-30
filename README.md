# GMLFinalProject

Package to develop new dimensionality reduction techniques using graphs and graph machine learning.

## Installation

To run the `final_notebook_colab.ipynb` on GoogleColab no special installation is necessary.

To run the package and `final_notebook.ipynb` locally it is required to install dependencies by using:

```
pip install --upgrade pip
pip install -r requirements.txt
```

Using virtual environment with python 3.6 is recommended.

Almost all functionality is enabled by default, however, python 3.6 is required for running the GraphSAGE embedding algorithm. This algorithm will not work otherwise.
The StellarGraph library requires python 3.6, so using this specific version is highly recommended.

This code does not use GPU.

## Usage

Example usage of the provided functions and classes is showcased in `final_notebook.ipynb`.

The structure of the project consists of `Builders` and `Embedders`.

- `Builders` construct graphs of different types from input data points.
    - weight and feature functions can be chosen from `./src/utils/weights.py` and `./src/utils/features.py` respectively
    - note that only GraphSAGE uses and requires node features, weights are used by all algorithms
- `Embedders` takes a graph built by a `Builder` and embeds it into a space of specified dimension.

More `Builders` than showcased in the `final_notebook.ipynb` are in `./src/utils/build.py` and `Embedders` are in `./src/utils/ebmedding.py`.

These two instances are then passed into a function `reduce_dimension(data, builder, embedder)` from `./src/utils/dim_reduction.py` which connects them and returns the embedding of the input data.

To visualize graphs and final embeddings use functions provided in `./src/utils/visualization.py`, to see metrics evaluating per

## Example

In this section we show a simple example usage of this package on the Swiss roll dataset.

First we import all necessary packages and modules.

```
from sklearn.datasets import make_swiss_roll
from utils import embedding, build, visualization, weights, features, dim_reduction, evaluation
```

Then, the dataset needs to be generated. It can also be displayed.

```
data, labels = make_swiss_roll(n_samples=1000, noise=0.0, random_state=0)
visualization.show_data(data, labels=labels, square=True)
```

To embed the data a builder and embedder need to be created.

```
builder = build.CheapestBuilder()
embedder = embedding.KamadaKawaiEmbedder(embedding_dim=2)
```

Then, the dimensionality reduction is performed and the trustworthiness metric is computed.

```
embeddings = dim_reduction.reduce_dimension(data, builder, embedder)
evaluation.print_evaluation(data, embeddings)
```

To show unwrapped swiss roll use tools from visualization module.

```
visualization.show_data(embedder.embeddings, labels=labels, square=True)
```
