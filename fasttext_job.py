import fasttext
import fasttext.util
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ft_model = fasttext.load_model('/mnt/c/Lamosst/FEL/Paty_semestr/Projekt_bakalarka/research-test/task/classification/token/ner/wiki.simple.bin')
#print(ft_model.get_dimension())
#fasttext.util.reduce_model(ft_model, 100)
#ft_model.save_model(path="/mnt/c/Lamosst/FEL/Paty_semestr/Projekt_bakalarka/research-test/task/classification/token/ner/wiki.simple100.bin")

class FasttextJob:
    def __init__(self, test_path, train_path, separator, labels_path):
        self.entity_dict = {}
        self.train_path = train_path
        self.test_path = test_path
        self.train_data_df = None
        self.test_data_df = None
        self.labels = None
        self.labels_path = labels_path
        self.separator = separator
        self.entity_embeddings = None  # dictionary['entity_label'] = fasttext_embedding will be filled using get_word_vector
        self.evaluate_dict = None

    def load_dataset(self):
        self.labels = self.get_all_labels_without_iob()
        print(self.labels)
        self.train_data_df = self.read_dataset_to_df(self.train_path, self.separator)
        self.test_data_df = self.read_dataset_to_df(self.test_path, self.separator)

        del self.train_data_df['POS']
        del self.train_data_df['CHUNK']
        del self.train_data_df['SENTENCE']
        del self.test_data_df['POS']
        del self.test_data_df['CHUNK']
        del self.test_data_df['SENTENCE']

        self.train_data_df['TOKEN'] = self.train_data_df['TOKEN'].str.replace(r'[^\w\s]', '').str.lower()
        self.train_data_df = self.merge_iob(self.train_data_df, True)

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

    def fill_entity_embeddings(self):
        self.entity_embeddings = np.zeros((300, len(self.labels + ['O'])))

        for entity_count, entity_label in enumerate((self.labels + ['O'])):

            found_entities_idxs = self.train_data_df.index[self.train_data_df['NE'].eq(entity_label)].tolist()
            tmp_arr = np.zeros((300, len(found_entities_idxs)))
            for count, idx in enumerate(found_entities_idxs):
                cur_word_embedding = ft_model.get_word_vector(self.train_data_df.at[idx, 'TOKEN'])
                tmp_arr[:, count] = cur_word_embedding

            self.entity_embeddings[:, entity_count] = np.mean(tmp_arr, axis=1)

    def evaluate_dataset(self):
        labels_with_outside = self.labels + ['O']

        for row in self.test_data_df.itertuples(index=True):
            token = row.TOKEN
            entity = row.NE
            cur_index = row.Index

            cur_token_embedding = ft_model.get_word_vector(token).reshape((300, 1))
            cur_cos_similarities = np.zeros(len(labels_with_outside))

            for i in range(len(labels_with_outside)):
                cur_cos_similarities[i] = cosine_similarity(cur_token_embedding.T, self.entity_embeddings[:, i].reshape((300, 1)).T).flatten()

            entity_index = np.argmax(cur_cos_similarities)
            self.test_data_df.at[cur_index, 'PRED_NE'] = labels_with_outside[entity_index]

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
    train_path = "/mnt/c/Lamosst/FEL/Paty_semestr/Projekt_bakalarka/research-test/data/ner/ontonotes5.0/train.conll"
    test_path = "/mnt/c/Lamosst/FEL/Paty_semestr/Projekt_bakalarka/research-test/data/ner/ontonotes5.0/test.conll"
    labels_path = "/mnt/c/Lamosst/FEL/Paty_semestr/Projekt_bakalarka/research-test/data/ner/ontonotes5.0/labels.txt"

    fasttext_job = FasttextJob(
        train_path=train_path,
        test_path=test_path,
        separator='\t',
        labels_path=labels_path
    )
    fasttext_job.load_dataset()
    fasttext_job.fill_entity_embeddings()
    fasttext_job.evaluate_dataset()
    print(fasttext_job.evaluate_dict)


def run_on_conll():
    train_path = "/mnt/c/Lamosst/FEL/Paty_semestr/Projekt_bakalarka/research-test/data/ner/conll2003/train.txt"
    test_path = "/mnt/c/Lamosst/FEL/Paty_semestr/Projekt_bakalarka/research-test/data/ner/conll2003/test.txt"
    labels_path = "/mnt/c/Lamosst/FEL/Paty_semestr/Projekt_bakalarka/research-test/data/ner/conll2003/labels.txt"

    fasttext_job = FasttextJob(
        train_path=train_path,
        test_path=test_path,
        separator=' ',
        labels_path=labels_path
    )
    fasttext_job.load_dataset()
    fasttext_job.fill_entity_embeddings()
    fasttext_job.evaluate_dataset()
    print(fasttext_job.evaluate_dict)

if __name__ == "__main__":
    run_on_conll()
    #run_on_ontonotes()
