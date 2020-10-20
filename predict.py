from simpletransformers.classification import ClassificationModel
from scipy.stats import pearsonr, spearmanr
import pandas as pd

#model
def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]


def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]

#load data

train_df = pd.read_csv('data/train/train.tsv', sep='\t', error_bad_lines=False)
eval_df = pd.read_csv('data/train/dev.tsv', sep='\t', error_bad_lines=False)

train_df = train_df.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'score': 'labels'}).dropna()
eval_df = eval_df.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'score': 'labels'}).dropna()

model = ClassificationModel('roberta', './outputs/best_model', num_labels=1)
# predictions, raw_outputs = model.predict("I am very fine")
a, b= model.predict(
    [
        [
            "Who is she ?",
            "She is a girl",
        ]
    ])

print(a, b)