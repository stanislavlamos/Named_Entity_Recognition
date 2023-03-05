import pathlib
import sys
import spacy
from spacy import Language


class SpacyJob:


    def convert_dataset(self, train_path, test_path, output_path,
                        eval_path=None):

        datasets_paths = [train_path, test_path, eval_path]
        for cur_path in datasets_paths:
            if cur_path is None:
                continue

            dataset_file_type = pathlib.Path(cur_path).suffix
            converter_type = "ner"

            if dataset_file_type == ".conll":
                converter_type = "conll"

            spacy.cli.convert(
                input_path=cur_path,
                output_dir=output_path,
                n_sents=10,
                converter=converter_type,
                file_type="spacy"
            )

    def train_new_model(self, config_path, train_path, output_path,
                        eval_path=None):

        custom_overrides = {
            "paths.train": train_path,
            "paths.vectors": "../spacy_job/word_vectors/wikipedia_fasttext"
        }

        if eval_path is not None:
            custom_overrides["paths.dev"] = eval_path

        spacy.cli.train.train(
            config_path=config_path,
            output_path=output_path,
            overrides=custom_overrides
        )

    def evaluate_model(self, model_best_path, output_path, test_path, displacy_path=None):
        spacy.cli.evaluate(
            model=model_best_path,
            data_path=pathlib.Path(test_path),
            output=pathlib.Path(output_path),
            displacy_path=displacy_path,
            displacy_limit=sys.maxsize
        )


def run_new_model_ontonotes():
    spacy_job = SpacyJob()
    config_path = "../spacy_job/default_config.cfg"
    train_onto = "../data/ontonotes5.0/train.conll"
    test_onto = "../data/ontonotes5.0/test.conll"
    eval_onto = "../data/ontonotes5.0/development.conll"

    train_spacy_onto = "../data/ontonotes5.0/train.spacy"
    test_spacy_onto = "../data/ontonotes5.0/test.spacy"
    eval_spacy_onto = "../data/ontonotes5.0/development.spacy"

    spacy_job.convert_dataset(
        train_path=train_onto,
        test_path=test_onto,
        eval_path=eval_onto,
        output_path="../data/ontonotes5.0/"
    )

    spacy_job.train_new_model(
        config_path=config_path,
        train_path=train_spacy_onto,
        output_path="../spacy_job/model_ontonotes",
        eval_path=eval_spacy_onto
    )

    spacy_job.evaluate_model(
        model_best_path="../spacy_job/model_ontonotes/model-best",
        output_path="../spacy_job/model_ontonotes/metrics_ontonotes.json",
        test_path=test_spacy_onto,
        displacy_path="../spacy_job/model_ontonotes"
    )


def run_new_model_conll():
    spacy_job = SpacyJob()
    config_path = "../spacy_job/default_config.cfg"
    train_conll = "../data/conll2003/train.txt"
    test_conll = "../data/conll2003/test.txt"
    eval_conll = "../data/conll2003/valid.txt"

    train_spacy_conll = "../data/conll2003/train.spacy"
    test_spacy_conll = "../data/conll2003/test.spacy"
    eval_spacy_conll = "../data/conll2003/valid.spacy"

    spacy_job.convert_dataset(
        train_path=train_conll,
        test_path=test_conll,
        eval_path=eval_conll,
        output_path="../data/conll2003/"
    )

    spacy_job.train_new_model(
        config_path=config_path,
        train_path=train_spacy_conll,
        output_path="../spacy_job/model_conll",
        eval_path=eval_spacy_conll
    )

    spacy_job.evaluate_model(
        model_best_path="../spacy_job/model_conll/model-best",
        output_path="../spacy_job/model_conll/metrics_conll.json",
        test_path=test_spacy_conll,
        displacy_path="../spacy_job/model_conll"
    )


def run_pretrained_model_ontonotes():
    spacy_job = SpacyJob()
    test_spacy_onto = "../data/ontonotes5.0/test.spacy"

    spacy_job.evaluate_model(
        model_best_path="en_core_web_sm",
        output_path="../spacy_job/model_en_core_web_sm_ontonotes/metrics_pretrained_ontonotes.json",
        test_path=test_spacy_onto,
        displacy_path="../spacy_job/model_en_core_web_sm_ontonotes"
    )


def run_pretrained_model_conll():
    spacy_job = SpacyJob()
    test_spacy_conll_pretrained = "../data/conll2003/test_pretrained.spacy"

    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("misc_incorporation")
    nlp.to_disk(
        path="../spacy_job/model_en_core_web_sm_conll/model")

    spacy_job.evaluate_model(
        model_best_path="../spacy_job/model_en_core_web_sm_conll/model",
        output_path="../spacy_job/model_en_core_web_sm_conll/metrics_pretrained_conll.json",
        test_path=test_spacy_conll_pretrained,
        displacy_path="../spacy_job/model_en_core_web_sm_conll"
    )


@Language.component("misc_incorporation")
def misc_incorporation(doc):
    ents = list(doc.ents)
    non_misc_labels = ["LOC", "PERSON", "ORG", "O"]
    for ent_idx in range(len(ents)):
        if ents[ent_idx].label_ not in non_misc_labels:
            ents[ent_idx].label_ = "MISC"
    ents = tuple(ents)
    doc.ents = ents
    return doc


if __name__ == "__main__":
    run_new_model_ontonotes()
    run_new_model_conll()
    run_pretrained_model_ontonotes()
    run_pretrained_model_conll()
