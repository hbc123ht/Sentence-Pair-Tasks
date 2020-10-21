#!/usr/bin/python
# -*- coding: utf-8 -*-
from simpletransformers.classification import ClassificationModel
from scipy.stats import pearsonr, spearmanr
import pandas as pd

#model
def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]


def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]

model = ClassificationModel('roberta', './outputs/best_model', num_labels=1)
# predictions, raw_outputs = model.predict("I am very fine")
a, b= model.predict(
    [
        [
            "起業家精神はどれほど難しいか",
            "非常に難しいです",
        ]
    ])

print(a)