import pandas as pd
import string
from difflib import SequenceMatcher
from nltk import word_tokenize
from pathlib import Path
import numpy as np

run = [3, 5, 6]
similars = [0, 1]
tokens = [5]
hate = [0, 1]

combinations = np.array(np.meshgrid(run, similars, tokens, hate)).T.reshape(-1,4)

for combo in combinations:
    RUN, SIMILARS, TOKENS, HATE = combo

    def preprocess(df):
        """
        Preprocesses the text. Consists of 2 steps:
            - Remove <|startoftext|> and <|endoftext|>.
            - Remove sentences containing smaller than n in tokens.

        :param df: original DataFrame with no preprocessing done
        :return: updated DataFrame
        """
        # Define characters to remove
        delete_table = dict.fromkeys(map(ord, '<|>'), '')
        df.generated_text = df.generated_text.apply(
            lambda sent: sent.translate(delete_table).replace('startoftext', '').replace('endoftext', ''))
        df['generated_text'].dropna(inplace=True)  # Remove empty rows
        print(f'After removing <|startoftext|> and <|endoftext|>: {len(df)}')

        # Drop sentences of n words
        df['length'] = df.generated_text.apply(lambda sent: len(word_tokenize(sent)))
        df = df[df['length'] > TOKENS]
        df.drop('length', axis=1, inplace=True)
        print(f'After removing single word sentences: {len(df)}')
        return df


    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()


    def clean_text(sent):
        """
        Converts the text to lowercase and removes punctuation.
        :param sent: sentence
        :return: cleaned sentence (lowercase, no punctuation).
        """
        # Convert to lowercase
        sent = sent.lower()

        # Remove punctuation
        translation = dict.fromkeys(map(ord, string.punctuation), '')
        sent = sent.translate(translation)

        return sent


    def remove_duplicates(df, concat=False):
        """
        Drops duplicates from the DataFrame.

        :param df:
        :param concat:
        :return:
        """
        # Remove exact duplicates
        if not concat:
            df['cleaned'] = df['generated_text'].apply(lambda x: clean_text(x))
        df.drop_duplicates(['cleaned'], keep='first', inplace=True)

        domain = 'in'
        if concat:
            domain = 'out'
            df = df['generated_text']
            df = df.dropna()

        print(f'After dropping {domain}-domain duplicates: {len(df)}.')

        return df


    def remove_similar_sents(df, percentage=0.9):
        """
        Remove sentences that are very similar.
        :param df:
        :param percentage:
        :return:
        """
        indices = []
        for i, row in df.iterrows():
            for row2 in df.loc[i + 1:].itertuples():
                if similar(row.generated_text, row2[3]) >= percentage:
                    indices.append((i, row2[0]))

        similars = [first for first, second in list(set(tuple(sorted(sub)) for sub in indices))]
        df.drop(similars, inplace=True)

        print(f'After dropping similars: {len(df)}')

        return df


    def clean(train_df, generated_df, run=int()):
        print(f'HATE: {HATE}')
        print(f'Original length: {len(generated_df)}')
        generated_df = preprocess(generated_df)

        generated_df['cleaned'] = generated_df['generated_text'].apply(lambda sent: clean_text(sent))
        generated_df = generated_df.reset_index()

        generated_df = remove_duplicates(generated_df)

        if SIMILARS:
            generated_df = remove_similar_sents(generated_df)

        train_df['cleaned'] = train_df.train_text.apply(lambda sent: clean_text(sent))

        total_df = generated_df.append(train_df)
        cleaned_df = remove_duplicates(total_df, concat=True)

        if HATE:
            type = 'hate'
        else:
            type = 'nohate'

        cleaned_df.to_csv(f'./data/cleaned/run{run}/min{TOKENS}tokens_similars{SIMILARS}_{type}.csv', index=False)


    def run():

        gibert_dir = 'data/gibert'
        for filepath in Path(gibert_dir).glob('**/*.csv'):
            if not HATE:
                if 'nohate' in str(filepath):
                    train_nohate = pd.read_csv(filepath,
                                               sep='\n',
                                               header=None,
                                               error_bad_lines=False,
                                               names=['train_text'])
            if HATE:
                if 'hate' in str(filepath):
                    train_hate = pd.read_csv(filepath,
                                             sep='\n',
                                             header=None,
                                             error_bad_lines=False,
                                             names=['train_text'])

        gpt_dir = './data/'
        for filepath in Path(gpt_dir).glob(f'**/run{RUN}/*.txt'):
            if not HATE:
                if 'nohate' in str(filepath):
                    generated_nohate = pd.read_csv(filepath,
                                                   sep='\n',
                                                   header=None,
                                                   error_bad_lines=False,
                                                   names=['generated_text'])
            if HATE:
                if 'hate' in str(filepath):
                    generated_hate = pd.read_csv(filepath,
                                                 sep='\n',
                                                 header=None,
                                                 error_bad_lines=False,
                                                 names=['generated_text'])

        if HATE:
            clean(train_hate, generated_hate, run=RUN)

        else:
            clean(train_nohate, generated_nohate, run=RUN)


    if __name__ == '__main__':
        run()
