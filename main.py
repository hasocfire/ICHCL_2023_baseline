import pandas as pd

from data_utils import get_stopwords_stemmer, get_dataloaders, get_data, clean_tweet
from training_utils import train, eval, predict
from sklearn.preprocessing import LabelEncoder

import torch
from transformers import BertTokenizer, BertForSequenceClassification

import nltk
nltk.download('stopwords')

import argparse
parser = argparse.ArgumentParser(description='ICHCL')
parser.add_argument('--data_directory', type=str, default='data', help='Where the dataset is stopred.')
parser.add_argument('--task', type=str, default='binary', help='Binary or multiclass classification.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--epochs', type=int, default=10, help='Training epochs before pseudo labelling.')
parser.add_argument('--re_epochs', type=int, default=5, help='Training epochs post pseudo labelling.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate of transformer model.')
parser.add_argument('--wd', type=float, default=0, help='Weight decay of transformer model.')
parser.add_argument('--batch_size', type=int, default=64, help='Training batch size.')
parser.add_argument('--num_labels', type=int, default=2, help='Number of classes.')


args = parser.parse_args()

if args.gpu != -1 and torch.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

if __name__ == '__main__':
    print(args)

    stopword, english_stemmer = get_stopwords_stemmer()
    data_label, data_unlabeled = get_data(args.data_directory, args.task)

    df = pd.DataFrame(data_label, columns = data_label[0].keys(), index = None)
    df.loc[df['label']=='NONE']='NOT'
    df_unlabeled = pd.DataFrame(data_unlabeled, columns = data_unlabeled[0].keys(), index = None)
    print("Number of unlabeled:", len(df_unlabeled))
    print("Binary Distribution")
    print(df['label'].value_counts())
    tweets = df.text
    y = df.label
    cleaned_tweets = [clean_tweet(tweet, english_stemmer, stopword) for tweet in tweets]
    cleaned_unlabeled = [clean_tweet(tweet, english_stemmer, stopword) for tweet in df_unlabeled.text]
    le = LabelEncoder()
    labels = le.fit_transform(y)

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased',do_lower_case = True)
    train_dataloader, validation_dataloader  = get_dataloaders(cleaned_tweets, labels, tokenizer, 0.2, args.batch_size)

    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels = args.num_labels, output_attentions = False, output_hidden_states = False).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay  = args.wd )
    train(model, train_dataloader, optimizer, validation_dataloader, args.epochs, args.device)

    predictions = predict(cleaned_unlabeled, model, tokenizer, args.device)
    predictions = le.inverse_transform(predictions.astype('int'))
    df_unlabeled['label'] = predictions
    df = df.append(df_unlabeled)
    print("Binary Distribution")
    print(df['label'].value_counts())
    tweets = df.text
    y = df.label
    cleaned_tweets = [clean_tweet(tweet, english_stemmer, stopword) for tweet in tweets]
    labels = le.fit_transform(y)
    train_dataloader, validation_dataloader  = get_dataloaders(tweets, labels, tokenizer, 0.2, args.batch_size)

    train(model, train_dataloader, optimizer, validation_dataloader, args.re_epochs, args.device)

    eval(model, validation_dataloader, args.device)