# Analysing the Importance of LLMs Embeddings' Components on Probing Linguistic Tasks
This repository contains the source code to reproduce experiments from the project for Skoltech Machine Learning course ([Paper](https://github.com/pkseniya/EmbeddingComponents/blob/main/paper/Analysing_the_Importance_of_LLMs_Embeddings__Components_on_Probing_Linguistic_Tasks.pdf), [Slides](https://github.com/pkseniya/EmbeddingComponents/blob/main/slides/slides.pdf)). 
## Setup
This code is written in python. The dependencies are:

* Python 2/3 with [NumPy](http://www.numpy.org/)/[SciPy](http://www.scipy.org/)
* [Pytorch](http://pytorch.org/)>=0.4
* [scikit-learn](http://scikit-learn.org/stable/index.html)>=0.18.0

## Repository structure
### Experiments
- `notebooks/outliers.ipynb` -- calculation of outlier dimensions of embeddings [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pkseniya/EmbeddingComponents/blob/main/notebooks/outliers.ipynb)

- `notebooks/logreg.ipynb` -- getting feature importance of embeddings with logistic regression [![Open In Colab](https://colab.research.google.com/github/pkseniya/EmbeddingComponents/blob/main/notebooks/logreg.ipynb)

- `notebooks/shap.ipynb` -- getting feature importance of embeddings with shap and mlp [![Open In Colab](https://colab.research.google.com/github/pkseniya/EmbeddingComponents/blob/main/notebooks/shap.ipynb)

- `notebooks/fvalue.ipynb` -- getting feature importance of embeddings with ANOVA F-value [![Open In Colab](https://colab.research.google.com/github/pkseniya/EmbeddingComponents/blob/main/notebooks/fvalue.ipynb)
## Results
[TBD]
## References
### Analysing the Importance of LLMs Embeddings' Components on Probing Linguistic Tasks
Contact: [petrushina.ke@phystech.edu](mailto:petrushina.ke@phystech.edu)
### Credits
* [*SentEval: An Evaluation Toolkit for Universal Sentence Representations*](https://arxiv.org/abs/1803.05449)
