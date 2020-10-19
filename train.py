from simpletransformers.classification import ClassificationModel


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
    