import pandas as pd
import json 
import numpy as np
import glob

data = {
    'text_a' : [],
    'text_b' : [],
    'labels' : []
}
#read data
files = glob.glob('./data3/*.json')
for file in files:
    print(file)
    with open(file) as csv_file:
        csv_reader = json.load(csv_file)
        for Q_and_A in csv_reader:
            question = Q_and_A['question']
            answers = Q_and_A['answers']

            # get the max upvote
            mx = max(int(i[1]) for i in answers)
            
            #append data
            if (mx > 3):
                for answer in answers:

                    #process data
                    processed_answer = ''
                    for sentence in answer[0]:
                        processed_answer += sentence    
                    
                    #append data
                    data['text_a'].append(question)
                    data['text_b'].append(processed_answer)
                    data['labels'].append((5 * int(answer[1]) / mx))
df = pd.DataFrame(data=data)

data_train = df.iloc[0:int(len(df)*0.8)]
data_test = df.iloc[int(len(df)*0.8):int(len(df)*0.9)]
data_dev = df.iloc[int(len(df)*0.9):]

def write_df_to_txt(path, the_df): 
    file = open(path, "w", encoding="utf8")

    samples =  []
    samples.append("{0}\t{1}\t{2}\n".format("text_a", "text_b", "labels"))

    for x, y, z in zip(the_df['text_a'], the_df['text_b'], the_df['labels']): 
        samples.append("{0}\t{1}\t{2}\n".format(x, y, z))
    
    file.writelines(samples)

write_df_to_txt('data3/train/train.tsv', data_train)
write_df_to_txt('data3/train/text.tsv', data_test)
write_df_to_txt('data3/train/dev.tsv', data_dev)

# train_df = pd.read_csv('data/STS-B/train.tsv', sep='\t', error_bad_lines=False)
# eval_df = pd.read_csv('data/STS-B/dev.tsv', sep='\t', error_bad_lines=False)

# train_df = train_df.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'score': 'labels'}).dropna()
# eval_df = eval_df.rename(columns={'sentence1': 'text_a', 'sentence2': 'text_b', 'score': 'labels'}).dropna()
