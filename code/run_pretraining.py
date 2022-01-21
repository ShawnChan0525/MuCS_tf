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
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tqdm import tqdm
import os
import modeling
import optimization_gpu
from create_data_corpus import *
import tensorflow as tf
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.distribute.cross_device_ops import AllReduceCrossDeviceOps

flags = tf.compat.v1.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "n_gpus", 2,
    "GPU number")

flags.DEFINE_string("gpu", "0,1", "gpu id")

flags.DEFINE_string(
    "bert_config_file", "bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", "data/processed_data/train_instance.txt",
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "eval_input_file", "data/processed_data/test_instance.txt",
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string("token_vocab_file", 'data/processed_data/vocabs.txt',
                    "The token vocabulary file that the BERT model was trained on.")
flags.DEFINE_string(
    "output_dir", "output_dir",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_integer(
    "summary_categories", 6,
    "The number of summary_categories.")
flags.DEFINE_integer(
    "action_words_categories", 40,
    "The number of action words.")
# Other parameters
flags.DEFINE_string(
    "init_checkpoint", "",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 256,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_test", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 16, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 300000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 1000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 50000, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("isdistributed", True,
                  "Whether to do distributed training or evaluation.")

flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
    "LM", True, "Whether to operate unidirectional LM pretraining.")

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu


def get_lm_weights(target_id, word2id, unk_words):
    labels = target_id
    condition_unk = tf.zeros(labels.shape, dtype=tf.bool)
    zero_weights = tf.zeros_like(labels, tf.float32)
    weights = tf.ones(labels.shape, dtype=tf.float32)
    for word in unk_words:
        unk_id = word2id[word]
        unk_tf = tf.constant(value=unk_id, dtype=tf.int32, shape=labels.shape)
        condition_unk = tf.logical_or(condition_unk, tf.equal(labels, unk_tf))
    lm_weights = tf.where(condition_unk, zero_weights, weights)
    return lm_weights


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, word2id):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.compat.v1.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.compat.v1.logging.info(
                "  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        SCP_labels = features["SCP_labels"]
        AWP_labels = features["AWP_labels"]
        code_size = features["code_size"]
        lm_weights = get_lm_weights(input_ids, word2id, [
                                    '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[EOS]'])

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            code_size=code_size,
            use_one_hot_embeddings=use_one_hot_embeddings,
            LM=FLAGS.LM)

        (lm_loss, lm_example_loss, lm_log_probs) = get_lm_output(
            bert_config, model.get_embedding_output(
            ), model.get_embedding_table(), input_ids,
            lm_weights)

        (SCP_loss, SCP_example_loss,
         SCP_log_probs) = get_classify_output(
            bert_config, model.get_pooled_output(), SCP_labels, 'SCP', FLAGS.summary_categories)

        (AWP_loss, AWP_example_loss,
         AWP_log_probs) = get_classify_output(
            bert_config, model.get_pooled_output(), AWP_labels, 'AWP', FLAGS.action_words_categories)

        total_loss = SCP_loss + AWP_loss + lm_loss
        tvars = tf.compat.v1.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.compat.v1.train.init_from_checkpoint(
                        init_checkpoint, assignment_map)
                    return tf.compat.v1.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.compat.v1.train.init_from_checkpoint(
                    init_checkpoint, assignment_map)

        tf.compat.v1.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                      init_string)

        total_parameters = 0
        for variable in tf.compat.v1.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            # print(shape)
            # print(len(shape))
            variable_parameters = 1
            for dim in shape:
                # print(dim)
                variable_parameters *= dim.value
            # print(variable_parameters)
            total_parameters += variable_parameters
        print('total parameters: ', total_parameters)
        with open('parameters.txt', 'w') as f:
            f.write(str(total_parameters))

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization_gpu.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(SCP_example_loss,
                          SCP_log_probs, SCP_labels, AWP_example_loss,
                          AWP_log_probs, AWP_labels, lm_example_loss, lm_log_probs,
                          input_ids, lm_weights):
                # log probs: 预测值
                """Computes the loss and accuracy of the model."""
                SCP_log_probs = tf.reshape(
                    SCP_log_probs, [-1, SCP_log_probs.shape[-1]])
                SCP_predictions = tf.argmax(
                    SCP_log_probs, axis=-1, output_type=tf.int32)
                SCP_labels = tf.reshape(SCP_labels, [-1])
                SCP_accuracy = tf.compat.v1.metrics.accuracy(
                    labels=SCP_labels, predictions=SCP_predictions)
                SCP_mean_loss = tf.compat.v1.metrics.mean(
                    values=SCP_example_loss)

                AWP_log_probs = tf.reshape(
                    AWP_log_probs, [-1, AWP_log_probs.shape[-1]])
                AWP_predictions = tf.argmax(
                    AWP_log_probs, axis=-1, output_type=tf.int32)
                AWP_labels = tf.reshape(AWP_labels, [-1])
                AWP_accuracy = tf.compat.v1.metrics.accuracy(
                    labels=AWP_labels, predictions=AWP_predictions)
                AWP_mean_loss = tf.compat.v1.metrics.mean(
                    values=AWP_example_loss)

                lm_log_probs = tf.reshape(lm_log_probs,
                                          [-1, lm_log_probs.shape[-1]])
                lm_predictions = tf.argmax(
                    lm_log_probs, axis=-1, output_type=tf.int32)
                lm_example_loss = tf.reshape(lm_example_loss, [-1])
                lm_target_ids = tf.reshape(input_ids, [-1])
                lm_weights = tf.reshape(lm_weights, [-1])
                lm_accuracy = tf.compat.v1.metrics.accuracy(
                    labels=lm_target_ids,
                    predictions=lm_predictions,
                    weights=lm_weights)
                lm_mean_loss = tf.compat.v1.metrics.mean(
                    values=lm_example_loss, weights=lm_weights)

                return {
                    "SCP_accuracy": SCP_accuracy,
                    "SCP_loss": SCP_mean_loss,
                    "AWP_accuracy": AWP_accuracy,
                    "AWP_loss": AWP_mean_loss,
                    "lm_accuracy": lm_accuracy,
                    "lm_loss": lm_mean_loss,
                }

            unk_id = word2id['[UNK]']

            wrong_lm_label = tf.constant(
                value=-1, dtype=tf.int32, shape=input_ids.shape)
            unk_tf = tf.constant(value=unk_id, dtype=tf.int32,
                                 shape=input_ids.shape)
            condition_lm_tf = tf.equal(input_ids, unk_tf)
            new_lm_labels = tf.where(
                condition_lm_tf, wrong_lm_label, input_ids)

            eval_metrics = metric_fn(SCP_example_loss,
                                     SCP_log_probs, SCP_labels, AWP_example_loss,
                                     AWP_log_probs, AWP_labels,
                                     lm_example_loss, lm_log_probs, new_lm_labels, lm_weights,
                                     )

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.PREDICT:
            lm_log_probs = tf.reshape(lm_log_probs,
                                      [-1, lm_log_probs.shape[-1]])
            lm_predictions = tf.argmax(
                lm_log_probs, axis=-1, output_type=tf.int32)

            lm_target_ids = tf.reshape(input_ids, [-1])

            predictions = {
                "predictions": lm_predictions,
                "input_ids": lm_target_ids,
            }

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions)

        else:
            raise ValueError(
                "Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def get_lm_output(bert_config, input_tensor, output_weights, label_ids, label_weights):
    """Get loss and log probs for the Unidirectional LM."""
    input_tensor = tf.reshape(input_tensor, [-1, bert_config.hidden_size])
    with tf.variable_scope("lm_predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(tf.cast(label_weights, tf.float32), [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = - \
            tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_classify_output(bert_config, input_tensor, labels, scope, classes):
    """Get loss and log probs for the summary category prediction."""

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    with tf.compat.v1.variable_scope(scope):
        output_weights = tf.compat.v1.get_variable(
            "output_weights",
            shape=[classes, bert_config.hidden_size],
            initializer=modeling.create_initializer(bert_config.initializer_range))
        output_bias = tf.compat.v1.get_variable(
            "output_bias", shape=[classes], initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=classes, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     is_training,
                     num_cpu_threads=4):  # 原来是4
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = FLAGS.train_batch_size

        name_to_features = {
            "input_ids":
                tf.compat.v1.FixedLenFeature([max_seq_length], tf.int64),
            "SCP_labels":
                tf.compat.v1.FixedLenFeature([1], tf.int64),
            "AWP_labels":
                tf.compat.v1.FixedLenFeature([1], tf.int64),
            "code_size":
                tf.compat.v1.FixedLenFeature([1], tf.int64),
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            #d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))
        return d

    return input_fn


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.compat.v1.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.compat.v1.to_int32(t)
        example[name] = t

    return example


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    # if not FLAGS.do_train and not FLAGS.do_eval:
    #    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.compat.v1.gfile.MakeDirs(FLAGS.output_dir)

    token_word2id, token_vocab_size = read_vocab(FLAGS.token_vocab_file)
    #input_files = [FLAGS.input_file]
    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.compat.v1.gfile.Glob(input_pattern))

    tf.compat.v1.logging.info("*** Input Files ***")
    for input_file in input_files:
        tf.compat.v1.logging.info("  %s" % input_file)

    log_every_n_steps = 8

    if FLAGS.isdistributed:
        dist_strategy = tf.contrib.distribute.MirroredStrategy(
            num_gpus=FLAGS.n_gpus,
            cross_device_ops=AllReduceCrossDeviceOps(
                'nccl', num_packs=FLAGS.n_gpus),
            # cross_device_ops=AllReduceCrossDeviceOps('hierarchical_copy'),
        )
        log_every_n_steps = 8
        run_config = RunConfig(
            train_distribute=dist_strategy,
            eval_distribute=dist_strategy,
            log_step_count_steps=log_every_n_steps,
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps)

    else:
        run_config = RunConfig(
            log_step_count_steps=log_every_n_steps,
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps)

    steps = 1000
    while steps < FLAGS.num_train_steps:
        model_fn = model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=steps,
            num_warmup_steps=FLAGS.num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu,
            word2id=token_word2id)

        estimator = Estimator(
            model_fn=model_fn,
            config=run_config, )

        if FLAGS.do_train:
            tf.logging.info("***** Running training *****")
            tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
            train_input_fn = input_fn_builder(
                input_files=input_files,
                max_seq_length=FLAGS.max_seq_length,
                is_training=True)
            estimator.train(input_fn=train_input_fn,
                            max_steps=steps)

        if FLAGS.do_eval:
            tf.logging.info("***** Running evaluation *****")
            tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

            eval_input_files = [FLAGS.eval_input_file]
            eval_input_fn = input_fn_builder(
                input_files=eval_input_files,
                max_seq_length=FLAGS.max_seq_length,
                is_training=False)

            result = estimator.evaluate(
                input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

            output_eval_file = os.path.join(FLAGS.output_dir, "results/eval_results_%s.txt"%steps)
            with tf.io.gfile.GFile(output_eval_file, "w") as writer:
                tf.logging.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    tf.logging.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
        
        steps += 1000

        # 先不考虑test
        if FLAGS.do_test:

            tf.logging.info("***** Running test *****")
            tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
            test_input_files = [FLAGS.eval_input_file]
            test_input_fn = input_fn_builder(
                input_files=test_input_files,
                max_seq_length=FLAGS.max_seq_length,
                is_training=False)

            result = estimator.predict(
                input_fn=test_input_fn)
            tf.logging.info("***** Test results *****")
            output_eval_file = os.path.join(
                FLAGS.output_dir, "id_eval_results.txt")
            with tf.io.gfile.GFile(output_eval_file, "w") as writer:
                for i, p in tqdm(enumerate(result)):
                    writer.write(str(p['masked_pre']) + ' ' +
                                str(p['masked_tar']) + '\n')


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")

    tf.compat.v1.app.run()
