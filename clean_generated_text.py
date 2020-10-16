import string
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from nltk import word_tokenize


def preprocess(df):
    # Define characters to remove
    delete_table = dict.fromkeys(map(ord, '<|>'), '')
    df.generated_text = df.generated_text.apply(
        lambda sent: sent.translate(delete_table).replace('startoftext', '').replace('endoftext', ''))
    df['generated_text'].dropna(inplace=True)  # Remove empty rows
    print(f'After removing <|startoftext|> and <|endoftext|>: {len(df)}')

    # Drop singe word sentences
    df['length'] = df.generated_text.apply(lambda sent: len(word_tokenize(sent)))
    df = df[df['length'] > 1]
    df.drop('length', axis=1, inplace=True)
    print(f'After removing single word sentences: {len(df)}')
    return df


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def clean_text(sent):
    # Convert to lowercase
    sent = sent.lower()

    # Remove punctuation
    translation = dict.fromkeys(map(ord, string.punctuation), '')
    sent = sent.translate(translation)

    return sent


def remove_duplicates(df, concat = False):
    # Remove exact duplicates
    if not concat:
        df['cleaned'] = df['generated_text'].apply(lambda x: clean_text(x))
    df.drop_duplicates(['cleaned'], keep='first', inplace=True)

    if not concat:
        print('After dropping in-domain duplicates:', len(df))

    if concat:
        df = df[df.generated_text.notnull()]
        print(f'After dropping out-domain duplicates: {len(df)}')

    return df


def remove_similar_sents(df, percentage=0.9):
    indices = []
    for i, row in df.iterrows():
        for row2 in df.loc[i + 1:].itertuples():
            if similar(row.generated_text, row2[3]) >= percentage:
                indices.append((i, row2[0]))

    similars = [first for first, second in list(set(tuple(sorted(sub)) for sub in indices))]
    df.drop(similars, inplace=True)

    print(f'After dropping similars: {len(df)}')

    return df


def clean(train_df, generated_df, type=''):
    print(type)
    print(f'Original length: {len(generated_df)}')
    generated_df = preprocess(generated_df)

    generated_df['cleaned'] = generated_df.generated_text.apply(lambda sent: clean_text(sent))
    generated_df = generated_df.reset_index()

    generated_df = remove_duplicates(generated_df)
    generated_df = remove_similar_sents(generated_df)

    train_df['cleaned'] = train_df.train_text.apply(lambda sent: clean_text(sent))
    total_df = generated_df.append(train_df)

    total_df = remove_duplicates(total_df, concat=True)

    cleaned_df = total_df['generated_text']
    cleaned_df.to_csv(f'./data/cleaned/withoutsimilars_{type}.csv', index=False)


def run():
    gibert_dir = 'data/gibert'
    for filepath in Path(gibert_dir).glob('**/*.csv'):
        if 'nohate' in str(filepath):
            train_nohate = pd.read_csv(filepath,
                                       sep='\n',
                                       header=None,
                                       error_bad_lines=False,
                                       names=['train_text'])
        elif 'hate' in str(filepath):
            train_hate = pd.read_csv(filepath,
                                     sep='\n',
                                     header=None,
                                     error_bad_lines=False,
                                     names=['train_text'])

    gpt_dir = 'data/gpt2/'
    for filepath in Path(gpt_dir).glob('**/*.txt'):
        if 'nohate' in str(filepath):
            generated_nohate = pd.read_csv(filepath,
                                           sep='\n',
                                           header=None,
                                           error_bad_lines=False,
                                           names=['generated_text'])
        elif 'hate' in str(filepath):
            generated_hate = pd.read_csv(filepath,
                                         sep='\n',
                                         header=None,
                                         error_bad_lines=False,
                                         names=['generated_text'])

    clean(train_hate, generated_hate, type='hate')
    print()
    clean(train_nohate, generated_nohate, type='nohate')


if __name__ == '__main__':
    run()
