# Analysing the Importance of LLMs Embeddings' Components on Probing Linguistic Tasks
This repository contains the source code to reproduce experiments from the project for Skoltech Machine Learning course ([Paper](https://github.com/pkseniya/EmbeddingComponents/blob/main/paper/Analysing_the_Importance_of_LLMs_Embeddings__Components_on_Probing_Linguistic_Tasks.pdf), [Slides](https://github.com/pkseniya/EmbeddingComponents/blob/main/slides/slides.pdf)). 

Large language models (LLMs) have helped researchers to achieve tremendous results in the field of NLP. However, work is still being done on their interpretability, part of which is contextualized embeddings from LLMs. Previous works demonstrated that some dimensions in LLMs' embeddings are important to the representational quality of these embeddings for task specific knowledge. In this study, we analyze components' importance of LLMs by probing on simple tasks. Our results suggest that several embeddings' dimensions are directly responsible for definite linguistic properties.
## Setup
The implementation is on python and GPU-based. Tested with torch 2.2.1 and 1 Tesla T4 on Google Colab.

Local setup:
1. Clone this repository, for example, `git clone https://github.com/pkseniya/EmbeddingComponents.git`
2. `pip install -r ./EmbeddingComponents/requirements.txt` &ndash; installing the required libraries  
3. `pip install -e ./EmbeddingComponents/` &ndash; SentEval installation 
## Repository structure
All the experiments are issued in the form of pretty self-explanatory jupyter notebooks (notebooks/). For convenience, the majority of the evaluation output is preserved. Auxilary source code is moved to .py (feature_importance/).
### Experiments

- `python -m examples.bert` - computation of embeddings of probing tasks. Outputs the result into `datasets` folder.

- `notebooks/outliers.ipynb` &ndash; calculation of outlier dimensions of embeddings [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pkseniya/EmbeddingComponents/blob/main/notebooks/outliers.ipynb)

- `notebooks/outlier_vs_random_vs_all.ipynb` &ndash; comparing accuracues of logistic regression on all, outlier and random features [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pkseniya/EmbeddingComponents/blob/main/notebooks/outlier_vs_random_vs_all.ipynb)

- `notebooks/logreg.ipynb` &ndash; getting feature importance of embedding components with logistic regression [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pkseniya/EmbeddingComponents/blob/main/notebooks/logreg.ipynb)

- `notebooks/shap.ipynb` &ndash; getting feature importance of embedding components with shap and mlp [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pkseniya/EmbeddingComponents/blob/main/notebooks/shap.ipynb)

- `python -m feature_importances.catboost` - getting feature importance of embedding components from gradient boosting.

- `notebooks/fvalue.ipynb` &ndash; getting feature importance of embeddings with ANOVA F-value [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pkseniya/EmbeddingComponents/blob/main/notebooks/fvalue.ipynb)
## Results

### Outlier dimensions perform better than random features on probing tasks
<p align="center"><img src="pics/Test_accuracy.png" width="700" /></p>

### Several outlier dimensions with high feature importance for each task
<p align="center"><img src="pics/BigramShift.png" width="700" /></p>
To plot feature importance vs deviation from the mean (outlierness of the component) - run `./plot_methods.sh`.

### Few distinctive outlier dimensions with syntactic or semantic information
<p align="center"><img src="pics/TaskIntersection.png" width="700" /></p>

## References
### Analysing the Importance of LLMs Embeddings' Components on Probing Linguistic Tasks
Contact: [petrushina.ke@phystech.edu](mailto:petrushina.ke@phystech.edu)
### Credits
* [*SentEval: An Evaluation Toolkit for Universal Sentence Representations*](https://arxiv.org/abs/1803.05449)
* [*A Unified Approach to Interpreting Model Predictions*](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
* [*RoBERTa: A Robustly Optimized BERT Pretraining Approach*](https://arxiv.org/abs/1907.11692)