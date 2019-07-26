import click

from model.evaluation.text import score_files
from model.img2seq_hand import Img2SeqModel
from model.utils.general import Config
from model.utils.word_data_generator import DataGenerator
from model.utils.text import Vocab


@click.command()
@click.option('--results', default="results/hand_word/", help='Dir to results')
def main(results):
    # restore config and model
    dir_output = results

    config_data = Config(dir_output + "data.json")
    config_vocab = Config(dir_output + "vocab.json")
    config_model = Config(dir_output + "model.json")

    vocab = Vocab(config_vocab)
    model = Img2SeqModel(config_model, dir_output, vocab)
    model.build_pred()
    model.restore_session(dir_output + "model.weights/")

    # load dataset
    test_set = DataGenerator(
        index_file=config_data.index_test,
        path_formulas=config_data.path_formulas_test,
        dir_images=config_data.dir_images_test,
        max_iter=config_data.max_iter,
        path_matching=config_data.path_matching_test,
        form_prepro=vocab.form_prepro)

    # use model to write predictions in files
    config_eval = Config({"dir_answers": dir_output + "formulas_test/",
                          "batch_size": 20})
    files, perplexity = model.write_prediction(config_eval, test_set)
    formula_ref, formula_hyp = files[0], files[1]

    # score the ref and prediction files
    scores = score_files(formula_ref, formula_hyp)
    scores["perplexity"] = perplexity
    msg = " - ".join(["{} {:04.2f}".format(k, v) for k, v in scores.items()])
    model.logger.info("- Test Txt: {}".format(msg))


if __name__ == "__main__":
    main()
