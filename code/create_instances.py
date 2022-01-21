# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tqdm import tqdm
import collections
import json
import random
from create_data_corpus import *
# import tokenization
import tensorflow as tf
import math

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", 'data/processed_data/tokens.txt',
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", 'data/processed_data/token_training_instance.txt',
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("token_vocab_file", 'data/processed_data/vocabs.txt',
                    "The token vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", False,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_integer("max_seq_length", 256, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 30,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 1,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")


class TrainingInstance(object):
    """A single training instance (sentence pair).
    `SCP` for Summarization Category Prediction
    `AWP` for Action Words Prediction"""

    def __init__(self, tokens, SCP, AWP,code_size):
        self.tokens = tokens
        self.SCP = SCP
        self.AWP = AWP
        self.code_size = code_size

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [printable_text(x) for x in self.tokens]))
        s += "SCP: %s\n" % printable_text(self.SCP)
        s += "AWP: %s\n" % printable_text(self.AWP)
        s += "code_size: %s\n" % printable_text(self.code_size)
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(instances, word2id, max_seq_length, output_file):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    writers.append(tf.python_io.TFRecordWriter(output_file))
    writer_index = 0
    total_written = 0
    for (inst_index, instance) in tqdm(enumerate(instances)):
        input_ids = file_to_id(word2id, instance.tokens)
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
        summary_category = instance.SCP
        action_word = instance.AWP
        code_size = min(instance.code_size, max_seq_length)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)

        assert len(input_ids) == max_seq_length

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["SCP_labels"] = create_int_feature([summary_category])
        features["AWP_labels"] = create_int_feature([action_word])
        features["code_size"] = create_int_feature([code_size])
        # maybe 少个weights

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_training_instances(input_codes_file, input_comments_file, SCP_file, AWP_file, vocab,
                              dupe_factor, rng):
    """Create `TrainingInstance`s from raw text."""
    all_codes = [[]]
    all_comments = [[]]
    SCP_list = []
    AWP_list = []
    scp = open(SCP_file, 'r', encoding='utf-8').readlines()
    awp = open(AWP_file, 'r', encoding='utf-8').readlines()
    for i in range(len(scp)):
        SCP_list.append(int(scp[i]))
        AWP_list.append(int(awp[i]))

    with open(input_codes_file, 'r', encoding='utf-8') as f:
        tokendata = f.readlines()
    with open(input_comments_file, 'r', encoding='utf-8') as f:
        commentdata = f.readlines()
    assert(len(tokendata) == len(commentdata))
    for i in tqdm(range(len(tokendata))):
        tokenline = tokendata[i].strip()
        commentline = commentdata[i].strip()
        if not tokenline:
            all_codes.append([])
        else:
            tokens = json.loads(tokenline)
            if tokens:
                all_codes[-1].append(tokens)
        if not commentline:
            all_comments.append([])
        else:
            comment = json.loads(commentline)
            if comment:
                all_comments[-1].append(comment)

    # Remove empty documents
    all_codes = [x for x in all_codes if x]
    all_comments = [x for x in all_comments if x]
    rng.seed(FLAGS.random_seed)
    rng.shuffle(all_codes)
    rng.shuffle(all_comments)

    vocab_words = list(vocab.keys())
    instances = []
    for fac in range(dupe_factor):
        print('dupe time: {}'.format(fac+1))
        for document_index in tqdm(range(len(all_codes))):
            instances.extend(
                create_instances_from_document(
                    all_codes, all_comments, SCP_list, AWP_list, document_index))
    rng.shuffle(instances)
    return instances


def create_instances_from_document(
        all_codes, all_comments, category_prediction, action_words_prediction, document_index):
    """Creates `TrainingInstance`s for a single document."""
    SCP = category_prediction[document_index]
    AWP = action_words_prediction[document_index]
    code_size = len(all_codes[document_index][0])
    tokens = ['CLS']
    tokens.extend(all_codes[document_index][0])
    tokens.append('SEP')
    tokens.extend(all_comments[document_index][0])
    tokens.append('EOS')
    instance = TrainingInstance(
        tokens=tokens,
        SCP=SCP,
        AWP=AWP,
        code_size=code_size)

    return [instance]


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    # input_files = FLAGS.input_file # tokens.txt
    input_files = "data/processed_data/tokens.txt"
    input_comment_files = "data/processed_data/comment_tokens.txt"
    SCP_file = 'data/processed_data/SCP.txt'
    AWP_file = 'data/processed_data/AWP.txt'
    # input_type_files = FLAGS.input_type_file
    token_word2id, token_vocab_size = read_vocab(FLAGS.token_vocab_file)

    rng = random.Random(FLAGS.random_seed)
    instances = create_training_instances(
        input_files, input_comment_files, SCP_file, AWP_file, token_word2id, FLAGS.dupe_factor, rng)

    l = len(instances)
    training_instances = instances[:math.floor(0.8*l)]
    test_instances = instances[math.floor(0.8*l):]

    output_file = FLAGS.output_file
    tf.logging.info("*** Writing to training files ***")
    tf.logging.info("  %s", output_file)
    write_instance_to_example_files(
        training_instances, token_word2id, FLAGS.max_seq_length, "data/processed_data/train_instance.txt")
    tf.logging.info("*** Writing to test files ***")
    tf.logging.info("  %s", output_file)
    write_instance_to_example_files(
        test_instances, token_word2id, FLAGS.max_seq_length, 'data/processed_data/test_instance.txt')


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("token_vocab_file")

    tf.app.run()  # 用这个解析命令行后，运行main函数
