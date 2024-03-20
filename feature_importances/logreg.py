from datasets import load_from_disk
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

def get_feature_importances(task):
    dataset = load_from_disk(f'datasets/{task}.hf')

    logreg = LogisticRegression()
    logreg.fit(StandardScaler().fit_transform(dataset['train']['X']), dataset['train']['y'])
    
    with open(f'logreg/{task}.npy', 'wb') as f:
        np.save(f, np.abs(logreg.coef_).mean(axis=0))

if __name__ == '__main__':
    tasks = ['Length', 'WordContent', 'Depth', 'TopConstituents', 
         'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber', 
         'OddManOut', 'CoordinationInversion']

    for task in tasks:
        get_feature_importances(task)