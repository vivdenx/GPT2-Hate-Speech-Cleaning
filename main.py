import os
import string

import pandas as pd
from nltk import word_tokenize

generated_dir = 'data/gpt2/'

for file in os.listdir(generated_dir):
    if file.startswith('test.csv'):
        #generated_df = pd.read_csv(filepath_or_buffer=generated_dir + file,
                                   #sep='\n',
                                   #header = None,
                                   #error_bad_lines=False,
                                   #names=['text'])
        generated_df = pd.read_csv(generated_dir + file)

og_dir = 'data/gibert/'

for file in os.listdir(og_dir):
    if file.startswith('hate'):
        original_df = pd.read_csv(filepath_or_buffer=og_dir + file,
                                  sep='\n',
                                  header=None,
                                  error_bad_lines=False,
                                  names=['text'])

print(len(generated_df))


def remove_tags(df):
    """
    Remove '<|startoftext|>' and <|endoftext|> tags from the dataframe.
    These are sometimes left over from the GPT2 generation.

    :param df: GPT2 generated DataFrame.
    :return: updated DataFrame with these tags removed (df)
    """
    # Remove '<|startoftext|>' and '<|endoftext|>
    things_to_remove = ['>', '<', '|']
    delete_table = dict.fromkeys(map(ord, things_to_remove), '')

    df.text = df.text.apply(lambda x: x.translate(delete_table))
    df.text = df.text.apply(lambda x: x.replace('startoftext', '').replace('endoftext', ''))

    df['text'].dropna(inplace=True)
    return df


generated_df = remove_tags(generated_df)
print('After tag removal:', len(generated_df))


# Remove too short text
def remove_one_word_sentences(df):
    """
    Remove all sentences that only contain one word.

    :param df: GPT2 generated DataFrame
    :return: updated DataFrame
    """
    df['length'] = df.text.apply(lambda text: len(word_tokenize(text)))
    df = df[df.length > 1]
    df.drop('length', axis=1, inplace=True)
    return df


generated_df = remove_one_word_sentences(generated_df)

from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def clean_text(sentence):
    # Convert to lowercase
    sentence = sentence.lower()

    # Remove punctuation
    translation_table = dict.fromkeys(map(ord, string.punctuation), '')
    sentence = sentence.translate(translation_table)

    return sentence


def remove_duplicates(df, /, percentage=0.9):
    # Remove exact duplicates.
    df['cleaned'] = df['text'].apply(lambda x: clean_text(x))
    df.drop_duplicates(['cleaned'], keep='first', inplace=True)
    print('After dropping duplicates:', len(df))

    # Remove very similar texts.
    indices = []

    for row in df.itertuples():
        for row2 in df.loc[row[0] + 1:].itertuples():
            if similar(row[2], row2[2]) >= percentage:
                indices.append((row[0], row2[0]))

    similars = [first for first, second in list(set(tuple(sorted(sub)) for sub in indices))]
    df.drop(similars, inplace=True)

    # Dropped a total of 30.
    return df


# generated_df = remove_duplicates(generated_df)
# generated_df.to_csv('./data/gpt2/test.csv', index=False)

def remove_replicates(new_df, og_df, /, percentage=0.9):
    new_df = new_df.append(og_df, ignore_index=True)
    print(new_df)

    df = remove_duplicates(new_df, percentage=percentage)
    df['cleaned'].dropna(inplace=True)
    #df.drop('cleaned', axis=1, inplace=True)
    return df


generated_df = remove_replicates(generated_df, original_df)
print(len(generated_df ))
generated_df.to_csv('./data/gpt2/test2.csv', index=False)

# Remove duplicates that line up with the original dataset.
# for row in generated_df.itertuples():
#    for row2 in original_df.itertuples():
#        if similar(row[1].lower(), row2[1].lower()) >= 0.9:
#            print(row[1], row2[1])
#            generated_df.drop(row[0], inplace=True)

print(len(generated_df))
