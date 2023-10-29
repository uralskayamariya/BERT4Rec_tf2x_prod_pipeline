import os
import time
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
import mlflow
import src.bert4rec.modeling as modeling
import src.bert4rec.optimization as optimization
from src.utils.vocab import FreqVocab
from src.utils.utils import save_pkl
from dotenv import load_dotenv
load_dotenv()


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


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions, label_ids, label_weights):

    """Get loss and log probs for the masked LM."""
    # [batch_size*label_size, dim]
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.compat.v1.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.compat.v1.variable_scope("transform"):
            dense_input_tensor = tf.keras.layers.Dense(units=bert_config.hidden_size,
                                                       activation=modeling.get_activation(bert_config.hidden_act),
                                                       kernel_initializer=modeling.create_initializer(
                                                           bert_config.initializer_range
                                                       ))

            input_tensor = dense_input_tensor(input_tensor)
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.compat.v1.get_variable(
            "output_bias",
            shape=[output_weights.shape[0]],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        # logits, (bs*label_size, vocab_size)
        log_probs = tf.nn.log_softmax(logits, -1)

        label_ids = tf.reshape(label_ids, [-1])

        # label_ids1 = label_ids.eval(session=tf.compat.v1.Session())   
        # print(f'\nlabel_ids1: {np.array(label_ids1)}\n')

        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=output_weights.shape[0], dtype=tf.float32)
        
        # print(f'one_hot_labels: {np.array(one_hot_labels)}')
        
        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(
            log_probs * one_hot_labels, axis=[-1])        
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, item_size):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        print("*** Features ***")
        for name in sorted(features.keys()):
            print("  name = %s, shape = %s" % (name, features[name].shape))

        info = features["info"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=None,
            use_one_hot_embeddings=use_one_hot_embeddings)

        (masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
            bert_config,
            model.get_sequence_output(),
            model.get_embedding_table(), masked_lm_positions, masked_lm_ids,
            masked_lm_weights)

        total_loss = masked_lm_loss

        tvars = tf.compat.v1.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.compat.v1.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

        print("**** Trainable Variables ****")
        list_var_names = []
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            print("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)
            
            var_layer = '/'.join(var.name.split('/')[:-1])
            list_var_names.append(var_layer)
        emb_var_name = list_var_names[-4]
        print(f'emb_var_name: {emb_var_name}')
        file = open(f"config/emb_var_name.txt", "w")
        file.write(emb_var_name)
        file.close()

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate,
                                                     num_train_steps,
                                                     num_warmup_steps, use_tpu)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold=scaffold_fn)
            
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(masked_lm_example_loss, masked_lm_log_probs,
                          masked_lm_ids, masked_lm_weights):
                """Computes the loss and accuracy of the model."""
                masked_lm_log_probs = tf.reshape(
                    masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]])
                masked_lm_predictions = tf.argmax(
                    masked_lm_log_probs, axis=-1, output_type=tf.int32)
                masked_lm_example_loss = tf.reshape(masked_lm_example_loss,
                                                    [-1])
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions,
                    weights=masked_lm_weights)
                masked_lm_mean_loss = tf.compat.v1.metrics.mean(
                    values=masked_lm_example_loss, weights=masked_lm_weights)
                
                return {
                    "masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_loss": masked_lm_mean_loss,
                }

            tf.compat.v1.add_to_collection('eval_sp', masked_lm_log_probs)
            tf.compat.v1.add_to_collection('eval_sp', input_ids)
            tf.compat.v1.add_to_collection('eval_sp', masked_lm_ids)
            tf.compat.v1.add_to_collection('eval_sp', info)

            eval_metrics = metric_fn(masked_lm_example_loss,
                                     masked_lm_log_probs, masked_lm_ids,
                                     masked_lm_weights)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn)
        else:
            raise ValueError("Only TRAIN and EVAL modes are supported: %s" %
                             (mode))

        return output_spec

    return model_fn


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


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        print(f'\nparams: {params}\n')
        if is_training:
            batch_size = params["batch_size_train"]
        else:
            batch_size = params["batch_size_val"]

        name_to_features = {
            "info":
                tf.compat.v1.FixedLenFeature([1], tf.int64),  # [user]
            "input_ids":
                tf.compat.v1.FixedLenFeature([max_seq_length], tf.int64, default_value=[0] * max_seq_length),
            "input_mask":
                tf.compat.v1.FixedLenFeature([max_seq_length], tf.int64, default_value=[0] * max_seq_length),
            "masked_lm_positions":
                tf.compat.v1.FixedLenFeature([max_predictions_per_seq], tf.int64,
                                                     default_value=[0] * max_predictions_per_seq),
            "masked_lm_ids":
                tf.compat.v1.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
                tf.compat.v1.FixedLenFeature([max_predictions_per_seq], tf.float32,
                                             default_value=[0.0] * max_predictions_per_seq)
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.TFRecordDataset(input_files)
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

            # `cycle_length` is the number of parallel files that get read.
            # cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            # d = d.apply(
            #    tf.contrib.data.parallel_interleave(
            #        tf.data.TFRecordDataset,
            #        sloppy=is_training,
            #        cycle_length=cycle_length))
            # d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)

        d = d.map(
            lambda record: _decode_record(record, name_to_features),
            num_parallel_calls=num_cpu_threads)
        d = d.batch(batch_size=batch_size)
        return d
    
    return input_fn


class TrainHooks(tf.estimator.SessionRunHook):
    def __init__(self, time_limit):
        self.time_limit = time_limit

    def begin(self):
        self.training_start_time = time.time()

    def after_run(self,
                run_context,  # pylint: disable=unused-argument
                run_values):
        
        training_time = time.time() - self.training_start_time
        if self.time_limit is not None and training_time >= self.time_limit:
            tf.compat.v1.logging.info(f"time limit: stopping training after {training_time} seconds")
            raise StopIteration() 


class EvalHooks(tf.compat.v1.train.SessionRunHook):
    def __init__(self, output_file, vocab, is_evaluate, max_predictions_per_seq, fast_eval):
        tf.compat.v1.logging.info('run init')
        self.output_file = output_file
        self.vocab = vocab
        self.is_evaluate = is_evaluate
        self.max_predictions_per_seq = max_predictions_per_seq
        self.fast_eval = fast_eval

    def begin(self):
        self.valid_user = 0.0

        self.ndcg_1 = 0.0
        self.hit_1 = 0.0
        self.ndcg_5 = 0.0
        self.hit_5 = 0.0
        self.ndcg_10 = 0.0
        self.hit_10 = 0.0
        self.ndcg_100 = 0.0
        self.hit_100 = 0.0
        self.ap = 0.0
        self.flag_rank100 = 0

        np.random.seed(12345)

    def end(self, session):
        if self.is_evaluate:
            print(
                "ndcg@1:{}, hit@1:{}， ndcg@5:{}, hit@5:{}, ndcg@10:{}, hit@10:{}, ndcg@100:{}, hit@100:{}, ap:{}, valid_user:{}".
                format(self.ndcg_1 / self.valid_user, self.hit_1 / self.valid_user,
                    self.ndcg_5 / self.valid_user, self.hit_5 / self.valid_user,
                    self.ndcg_10 / self.valid_user, self.hit_10 / self.valid_user,
                    self.ndcg_100 / self.valid_user, self.hit_100 / self.valid_user, 
                    self.ap / self.valid_user,
                    self.valid_user))
            print(f'\nДля {self.flag_rank100} пользователей предсказанный товар попал в 100 первых.')
            print(f'Это {round(self.flag_rank100 / self.valid_user * 100, 2)}% от всех пользователей.\n')
            mlflow.log_metrics({'ndcg1': self.ndcg_1 / self.valid_user})
            mlflow.log_metrics({'ndcg5': self.ndcg_5 / self.valid_user})
            mlflow.log_metrics({'ndcg10': self.ndcg_10 / self.valid_user})
            mlflow.log_metrics({'ndcg100': self.ndcg_100 / self.valid_user})
            mlflow.log_metrics({'hit1': self.hit_1 / self.valid_user})
            mlflow.log_metrics({'hit5': self.hit_5 / self.valid_user})
            mlflow.log_metrics({'hit10': self.hit_10 / self.valid_user})
            mlflow.log_metrics({'hit100': self.hit_100 / self.valid_user})
            mlflow.log_metrics({'rank100': self.flag_rank100})
            mlflow.log_metrics({'rank100_percent': round(self.flag_rank100 / self.valid_user * 100, 2)})


    def before_run(self, run_context):
        variables = tf.compat.v1.get_collection('eval_sp')
        return tf.compat.v1.train.SessionRunArgs(variables)

    
    def after_run(self, run_context, run_values):
        masked_lm_log_probs, input_ids, masked_lm_ids, info = run_values.results
        masked_lm_log_probs = masked_lm_log_probs.reshape(
            (-1, self.max_predictions_per_seq, masked_lm_log_probs.shape[1]))

        if self.fast_eval:
            masked_lm_log_probs = masked_lm_log_probs[:1]
            input_ids = input_ids[:1]
            masked_lm_ids = masked_lm_ids[:1]
            info = info[:1]
        
        
        for idx in range(len(input_ids)):
            input_items = []
            for item_id in input_ids[idx].tolist():
                while item_id != 0:
                    input_items.append(self.vocab.convert_ids_to_tokens([item_id])[0])
                    break
            masked_item = self.vocab.convert_ids_to_tokens([masked_lm_ids[0].tolist()[0]])[0]

            user_id = f"user_{info[idx][0]}"
            scores = masked_lm_log_probs[idx, 0]
            if self.is_evaluate:
                rank = self.write_predictions(user_id, scores, self.output_file, masked_item) # Определим rank среди всех товаров словаря.

                self.valid_user += 1

                # код проверяет, является ли значение переменной "valid_user" кратным 100. Если условие выполняется, то код выводит точку на экран и сбрасывает буфер вывода
                if self.valid_user % 100 == 0:
                    print('.', end='')
                    sys.stdout.flush()

                if rank:
                    if rank < 1:
                        self.ndcg_1 += 1
                        self.hit_1 += 1
                    if rank < 5:
                        self.ndcg_5 += 1 / np.log2(rank + 2)
                        self.hit_5 += 1
                    if rank < 10:
                        self.ndcg_10 += 1 / np.log2(rank + 2)
                        self.hit_10 += 1
                    if rank < 100:
                        self.ndcg_100 += 1 / np.log2(rank + 2)
                        self.hit_100 += 1

                    self.ap += 1.0 / (rank + 1)

                    if 1.0 / (rank + 1) > 0.01:
                        self.flag_rank100 += 1


    def write_predictions(self, user_id, scores, output_file, masked_item):
        if self.output_file:
            output_file.write(str(scores))
        # predicted_items = np.argsort(scores)[-predictions_per_user:][::-1]
        predicted_items = np.argsort(scores)[::-1]
        if self.output_file:
            output_file.write(user_id)
        list_tokens = []
        # flag_find = None
        true_item_number = None
        for i, item_id in enumerate(predicted_items):
            try:
                token = self.vocab.convert_ids_to_tokens([item_id + 1])[0] # добавила +1 !!!!!, смотри some_exp.ipynb, товары в словаре .vocab закодированы с 1
                list_tokens.append(token)
                score = scores[item_id]
                if self.output_file:
                    output_file.write(f";{token}:{score}")
                if token == masked_item:
                    true_item_number = i
                    # print(f"number of True (next, find) item in predictions from 0: {true_item}")
                    # flag_find = 1
                    break # запись в файл только до правильно предсказанного следующего товара, убрать, если нужно записывать всё
            except IndexError:
                continue
        # if flag_find == None:
        #     print(f'Среди {predictions_per_user} первых предсказанных товаров не нашлось правильного.')
        if self.output_file:
            output_file.write("\n")
        # print(f'\npredicted items (first 100): {list_tokens[:100]}\n')
        return true_item_number
    

class EvalHooks_embs_half1(tf.compat.v1.train.SessionRunHook):
    def __init__(self, project_path, model_name, hidden_size, max_predictions_per_seq):
        tf.compat.v1.logging.info('run init')
        self.list_users_all = []
        self.dfs_batches = []
        self.project_path = project_path
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.max_predictions_per_seq = max_predictions_per_seq

    def begin(self):
        file = open("config/emb_var_name.txt", "r")
        emb_var_name = file.read().rstrip().split('\n')[0]
        file.close()
        self.emb_var_name = f'{emb_var_name}/BiasAdd:0'

    def end(self, session):
        df = pd.concat(self.dfs_batches, axis = 0)
        results_path = f"{self.project_path}/results/{self.model_name}"
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        output_emb_file = f'{results_path}/df_next_sids_embs1.pkl'
        save_pkl(output_emb_file, df)
        print(df)    

    def before_run(self, run_context):
        # emb_layer_name = self.find_embeddings_layer_name(run_context)

        # Получение доступа к предпоследнему слою модели
        embedding_layer = run_context.session.graph.get_tensor_by_name(self.emb_var_name) 
        # Получение эмбеддингов
        embeddings = run_context.session.run(embedding_layer)
        # Конвертация эмбеддингов в массив NumPy
        embeddings_np = np.asarray(embeddings)
        list_embeddings_np = []
        for i in range(len(embeddings_np)):
            if (i + 1) % self.max_predictions_per_seq == 0:
                list_embeddings_np.append(embeddings_np[i])
        # Вывод массива эмбеддингов
        df = pd.DataFrame({'next_sid_embs': list_embeddings_np})
        self.dfs_batches.append(df)
    
    def after_run(self, run_context, run_values):
        pass

    def find_embeddings_layer_name(self, run_context):
        # all layers of the model
        ops = run_context.session.graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        # print(f'\nall_tensor_names: {all_tensor_names}\n')
        list_embs = []
        for layer in all_tensor_names:
            if layer.startswith(self.emb_var_name):
                print(layer)
                embedding_layer = run_context.session.graph.get_tensor_by_name(layer)
                try:
                    embeddings = run_context.session.run(embedding_layer)
                    print(embeddings.shape)
                    embeddings_np = np.asarray(embeddings)
                    if embeddings_np.shape == (self.hidden_size,): 
                        emb_layer_name = layer
                        list_embs.append(emb_layer_name)
                        print(f'{layer}: {embeddings_np}')
                except:
                    pass
        print(f'emb_layer_name: {emb_layer_name}')
        print(f'list_embs: {list_embs}')

        return emb_layer_name


class EvalHooks_embs_half2(tf.compat.v1.train.SessionRunHook):
    def __init__(self, project_path, model_name, hidden_size, max_predictions_per_seq):
        tf.compat.v1.logging.info('run init')
        self.list_users_all = []
        self.dfs_batches = []
        self.project_path = project_path
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.max_predictions_per_seq = max_predictions_per_seq

    def end(self, session):
        try:
            df = pd.concat(self.dfs_batches, axis = 0)
        except:
            df = pd.DataFrame()
        print(df)
        save_pkl(f'{self.project_path}/results/{self.model_name}/df_next_sids_embs2.pkl', df)

    def begin(self):
        file = open("config/emb_var_name.txt", "r")
        emb_var_name = file.read().rstrip().split('\n')[0]
        file.close()
        self.emb_var_name = f'{emb_var_name}/BiasAdd:0'
    
    def after_run(self, run_context, run_values):
        # Получение доступа к предпоследнему слою модели
        embedding_layer = run_context.session.graph.get_tensor_by_name(self.emb_var_name) 
        # Получение эмбеддингов
        embeddings = run_context.session.run(embedding_layer)
        # Конвертация эмбеддингов в массив NumPy
        embeddings_np = np.asarray(embeddings)
        list_embeddings_np = []
        for i in range(len(embeddings_np)):
            if (i + 1) % self.max_predictions_per_seq == 0:
                list_embeddings_np.append(embeddings_np[i])
        df = pd.DataFrame({'next_sid_embs': list_embeddings_np})
        self.dfs_batches.append(df)


class EvalHooks_inf(tf.compat.v1.train.SessionRunHook):
    def __init__(self, vocab, max_predictions_per_seq, predictions_per_user, project_path, model_name): 
        tf.compat.v1.logging.info('run init')
        self.list_items_all = []
        self.list_users_all = []
        self.vocab = vocab
        self.max_predictions_per_seq = max_predictions_per_seq
        self.predictions_per_user = predictions_per_user
        self.project_path = project_path
        self.model_name = model_name

    def end(self, session):
        df = pd.DataFrame({'id': self.list_users_all, 'next_sid': self.list_items_all})
        df['id'] = df.id.apply(lambda x: int(x.split('_')[1]))
        save_pkl(f'{self.project_path}/results/{self.model_name}/df_next_sids.pkl', df)
        print(df)

    def before_run(self, run_context):
        variables = tf.compat.v1.get_collection('eval_sp')
        return tf.compat.v1.train.SessionRunArgs(variables)
    
    def after_run(self, run_context, run_values):

        masked_lm_log_probs, input_ids, masked_lm_ids, info = run_values.results
        masked_lm_log_probs = masked_lm_log_probs.reshape(
            (-1, self.max_predictions_per_seq, masked_lm_log_probs.shape[1]))
        
        # for idx in range(len(input_ids)):

        #     input_items = []
        #     for item_id in input_ids[idx].tolist():
        #         while item_id != 0:
        #             input_items.append(self.vocab.convert_ids_to_tokens([item_id])[0])
        #             break
        #     masked_item = self.vocab.convert_ids_to_tokens([masked_lm_ids[0].tolist()[0]])[0]
        
        #     user_id = f"user_{info[idx][0]}"
        #     scores = masked_lm_log_probs[idx, 0]
            
        #     list_items_user = self.write_predictions(user_id, scores, masked_item)
        #     self.list_items_all.append(list_items_user)
        #     self.list_users_all.append(user_id)

        for idx in range(len(input_ids)):

            # input_items = []
            # for item_id in input_ids[idx].tolist():
            #     while item_id != 0:
            #         input_items.append(self.vocab.convert_ids_to_tokens([item_id])[0])
            #         break
            # masked_item = self.vocab.convert_ids_to_tokens([masked_lm_ids[0].tolist()[0]])[0]
        
            user_id = f"user_{info[idx][0]}"
            scores = masked_lm_log_probs[idx, 0]
            
            list_items_user = self.write_predictions(scores)
            self.list_items_all.append(list_items_user)
            self.list_users_all.append(user_id)


    def write_predictions(self, scores):
        predicted_items = np.argsort(scores)[-self.predictions_per_user:][::-1]
        list_tokens = []
        for i, item_id in enumerate(predicted_items):
            try:
                token = self.vocab.convert_ids_to_tokens([item_id + 1])[0] # добавила +1 !!!!!, смотри some_exp.ipynb, товары в словаре .vocab закодированы с 1
                list_tokens.append(token.split('_')[1])
            except IndexError:
                continue
        
        return list_tokens