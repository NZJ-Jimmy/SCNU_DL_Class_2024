import csv
import pickle
from qa_dataset import qa_dataset

# read in QA dataset
data_dict = []
with open('train_proto_qa.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data_dict.append(row)
# visualize some data, also you can vis in csv file
# print(len(data_dict))
# print(data_dict[:20])

# for this example, we just use train data
# we save the vocabulary and data into a pickle file, which can be used for decoding in generate.py
train_set = qa_dataset(data_dict)
vocab_size = train_set.get_vocab_size()
# with open('saves/vocab.pkl','wb') as f:
#     pickle.dump(train_set, f)
