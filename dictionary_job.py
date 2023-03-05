from sklearn.metrics import classification_report
import pandas as pd


class DictionaryJob:
    def __init__(self, test_path, train_path, separator, labels_path):
        self.entity_dict = {}
        self.train_path = train_path
        self.test_path = test_path
        self.train_data_df = None
        self.test_data_df = None
        self.labels = None
        self.labels_path = labels_path
        self.separator = separator
        self.evaluate_dict = None

    def load_dataset(self):
        self.labels = self.get_all_labels_without_iob()
        self.train_data_df = self.read_dataset_to_df(self.train_path, self.separator)
        self.test_data_df = self.read_dataset_to_df(self.test_path, self.separator)

        del self.train_data_df['POS']
        del self.train_data_df['CHUNK']
        del self.train_data_df['SENTENCE']
        del self.test_data_df['POS']
        del self.test_data_df['CHUNK']
        del self.test_data_df['SENTENCE']

        self.train_data_df['TOKEN'] = self.train_data_df['TOKEN'].str.replace(r'[^\w\s]', '').str.lower()
        self.train_data_df = self.merge_iob(self.train_data_df)

        self.test_data_df['TOKEN'] = self.test_data_df['TOKEN'].str.replace(r'[^\w\s]', '').str.lower()
        self.test_data_df = self.merge_iob(self.test_data_df, True)
        self.test_data_df['PRED_NE'] = 'O'

    def merge_iob(self, df, include_outside_tokens=False):
        m = df['NE'].eq('O')
        outside_tokens = df[m]
        outside_tokens_without_empty_strings = outside_tokens[outside_tokens['TOKEN'].astype(bool)]
        df = df[~m]
        df['NE'] = df['NE'].str.replace('[IB]-', '')
        df = (df.groupby([m.cumsum(), 'NE'])['TOKEN']
              .agg(' '.join)
              .droplevel(0)
              .reset_index()
              .reindex(df.columns, axis=1))

        if include_outside_tokens:
            return pd.concat([df, outside_tokens_without_empty_strings], ignore_index=True, sort=False)

        return df

    def evaluate_dataset(self):
        for row in self.train_data_df.itertuples(index=False):
            token = row.TOKEN
            entity = row.NE

            if token not in self.test_data_df.values:
                continue

            found_similarities_idxs = self.test_data_df.index[self.test_data_df['TOKEN'].eq(token)].tolist()
            for similarity_idx in found_similarities_idxs:
                self.test_data_df.at[similarity_idx, 'PRED_NE'] = entity

        self.evaluate_dict = classification_report(
            y_pred=self.test_data_df['PRED_NE'].tolist(),
            y_true=self.test_data_df['NE'].tolist(),
            labels=self.labels,
            zero_division=0
        )

    def read_dataset_to_df(self, dataset_path, separator):
        df = pd.read_csv(dataset_path,
                         sep=separator, header=None, keep_default_na=False,
                         names=['TOKEN', 'POS', 'CHUNK', 'NE'],
                         quoting=3, skip_blank_lines=False)
        df['SENTENCE'] = (df.TOKEN == '').cumsum()
        return df[df.TOKEN != '']

    def get_all_labels_without_iob(self):
        all_labels = set()
        with open(self.labels_path, 'r') as labels_file:
            for entity_tag in labels_file:
                entity_tag = entity_tag.rstrip('\n')

                if entity_tag == 'O':
                    continue

                all_labels.add(entity_tag[2:])

        return list(all_labels)


def run_on_ontonotes():
    train_path = "./data/ontonotes5.0/train.conll"
    test_path = "./data/ontonotes5.0/test.conll"
    labels_path = "./data/ontonotes5.0/labels.txt"

    dictionary_job = DictionaryJob(
        train_path=train_path,
        test_path=test_path,
        separator='\t',
        labels_path=labels_path
    )
    dictionary_job.load_dataset()
    dictionary_job.evaluate_dataset()
    print(dictionary_job.evaluate_dict)


def run_on_conll():
    train_path = "./data/conll2003/train.txt"
    test_path = "./data/conll2003/test.txt"
    labels_path = "./data/conll2003/labels.txt"

    dictionary_job = DictionaryJob(
        train_path=train_path,
        test_path=test_path,
        separator=' ',
        labels_path=labels_path
    )
    dictionary_job.load_dataset()
    dictionary_job.evaluate_dataset()
    print(dictionary_job.evaluate_dict)


if __name__ == "__main__":
    run_on_ontonotes()
    #run_on_conll()
