from simpletransformers.classification import ClassificationModel
from scipy.stats import pearsonr, spearmanr
import pandas as pd


#load data

train_df = pd.read_csv('data/train/train.tsv', sep='\t', error_bad_lines=False)
eval_df = pd.read_csv('data/train/dev.tsv', sep='\t', error_bad_lines=False)

train_df = train_df.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'score': 'labels'}).dropna()
eval_df = eval_df.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'score': 'labels'}).dropna()

#model
def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]


def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]


train_args = {
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'evaluate_during_training': True,
    'max_seq_length': 512,
    'num_train_epochs': 10,
    'evaluate_during_training_steps': 50,
    'wandb_project': 'sts-b-medium',
    'train_batch_size': 16,

    'regression': True,
}

model = ClassificationModel('roberta', 'roberta-base', num_labels=1, args=train_args)

model.train_model(train_df, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr)