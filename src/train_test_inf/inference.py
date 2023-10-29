import os
import pickle
import json
import tensorflow as tf
import mlflow
import argparse
from ruamel.yaml import YAML
import pandas as pd
import shutil
pd.set_option('display.max_colwidth', 150)
import src.bert4rec.modeling as modeling
from src.utils.vocab import FreqVocab
from src.bert4rec.main_processes import input_fn_builder, model_fn_builder, EvalHooks_inf, EvalHooks_embs_half1, EvalHooks_embs_half2
from src.utils.utils import load_pkl, save_pkl
from dotenv import load_dotenv
load_dotenv()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, help='Model name.')
    parser.add_argument('--is_embeddings', type=bool, default=True, help='Создать ли файл с эмбеддингами предпоследнего слоя модели. \
                                                                          Эмбеддинги сохраняются по адресу: projects/<имя_проекта>/results/<model_name>/df_next_sids_embs.pkl.')
    parser.add_argument('--is_items_preds', type=bool, default=True, help='Создать ли файл с предсказаниями товаров. \
                                                                           Сохраняется по адресу: projects/<имя_проекта>/results/<model_name>/df_next_sids.pkl.')
    parser.add_argument('--use_tpu', type=bool, default=False, help='Флаг, указывающий, будет ли использоваться TPU для обучения модели.')
    parser.add_argument('--batch_size', type=int, default=64, help='Размер батча.')
    parser.add_argument('--predictions_per_user', type=int, default=100, help='Сколько товаров необходимо записать в предсказания в порядке убывания уверенности модели. \
                                                                               В данном случае мы рассматриваем предсказания только для следующего товара.')
    args = parser.parse_args()

    return args


def _parse_tf_records(element):
    # Parse the input `tf.train.Example` proto using the dictionary schema.
    schema = {
        'info': tf.io.RaggedFeature(tf.int64, row_splits_dtype=tf.int64),
        'input_ids': tf.io.RaggedFeature(tf.int64, row_splits_dtype=tf.int64),
        'input_mask': tf.io.RaggedFeature(tf.int64, row_splits_dtype=tf.int64),
        'masked_lm_positions': tf.io.RaggedFeature(tf.int64, row_splits_dtype=tf.int64),
        'masked_lm_ids': tf.io.RaggedFeature(tf.int64, row_splits_dtype=tf.int64)
    }
    content = tf.io.parse_single_example(element, schema)
    return content


def main(_):
    stage = 'inference'
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_URI'))
    yaml = YAML()

    with open(f'{project_path}/config/config_mlflow.yaml', 'r') as file:
        config = yaml.load(file)

    experiment_name = config['last_experiment_name']

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=stage) as run:
        run_id = run.info.run_id
        config[stage] = {}
        config[stage]['experiment_name'] = experiment_name
        config[stage]['run_id'] = run_id

        with open(f'{project_path}/config/config_mlflow.yaml', 'w') as file:
            yaml.dump(config, file)

        mlflow.log_artifact(f'{project_path}/config/config_mlflow.yaml')
        mlflow.log_params(mlflow_params)
        mlflow.log_artifact(bert_config_file)
        mlflow.log_artifact(inference_config_path)

        bert_config = modeling.BertConfig.from_json_file(bert_config_file)

        tf.compat.v1.gfile.MakeDirs(checkpoint_dir)

        test_input_files = []
        for input_pattern in test_input_file.split(","):
            test_input_files.extend(tf.compat.v1.gfile.Glob(input_pattern))

        if is_embeddings:
            test_dataset = tf.data.TFRecordDataset([test_input_file])
            parsed_tf_records = test_dataset.map(_parse_tf_records)
            print(parsed_tf_records)
            test_dataset = pd.DataFrame(
                                    parsed_tf_records.as_numpy_iterator(),
                                    columns=['info', 'input_ids', 'input_mask', 'masked_lm_positions', 'masked_lm_ids']
                                )
            test_dataset['info'] = test_dataset['info'].apply(lambda x: int(x[0]))
            user_ids = test_dataset['info'].tolist()
            print(f'\nlen(user_ids): {len(user_ids)}\n')

        print("*** test Input Files ***")
        for input_file in test_input_files:
            print("  %s" % input_file)

        run_config = tf.estimator.RunConfig(
            model_dir=checkpoint_dir,
            save_checkpoints_steps=save_checkpoints_steps)

        with open(vocab_filename, 'rb') as input_file:
            vocab = pickle.load(input_file)
        item_size = len(vocab.counter)

        model_fn = model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=init_checkpoint,
            learning_rate=learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=use_tpu,
            use_one_hot_embeddings=use_tpu,
            item_size=item_size)
        
        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={
                    "batch_size_val": batch_size_val
                })
        
        print(f'\nestimator.latest_checkpoint(): {estimator.latest_checkpoint()}\n')

        print("\n***** Running evaluation *****\n")
        print("  Batch size = %d" % batch_size_val)

        eval_input_fn = input_fn_builder(
                                            input_files=test_input_files,
                                            max_seq_length=mlflow_params['max_seq_length'],
                                            max_predictions_per_seq=mlflow_params['max_predictions_per_seq'],
                                            is_training=False
                                            )
        
        if is_embeddings:
            estimator.evaluate(
                input_fn=eval_input_fn,
                steps=None,
                hooks=[EvalHooks_embs_half1(project_path, model_name, mlflow_params['hidden_size'], mlflow_params['max_predictions_per_seq'])]) 
            estimator.evaluate(
                input_fn=eval_input_fn,
                steps=None,
                hooks=[EvalHooks_embs_half2(project_path, model_name, mlflow_params['hidden_size'], mlflow_params['max_predictions_per_seq'])]) 
            
            path_to_emb1 = f'{project_path}/results/{model_name}/df_next_sids_embs1.pkl'
            path_to_emb2 = f'{project_path}/results/{model_name}/df_next_sids_embs2.pkl'
            df1 = load_pkl(path_to_emb1)
            df2 = load_pkl(path_to_emb2)

            df = pd.concat([df1, df2], axis = 0).reset_index(drop=True)
            df['id'] = user_ids
            print(df)
            save_pkl(f'{project_path}/results/{model_name}/df_next_sids_embs.pkl', df)
            os.remove(path_to_emb1)
            os.remove(path_to_emb2)

            file_path = "config/emb_var_name.txt"
            file = open(file_path, "r")
            emb_var_name = file.read().rstrip().split('\n')[0]
            file.close()
            emb_var_name = f'{emb_var_name}/BiasAdd:0'
            file = open(file_path, "w")
            file.write(emb_var_name)
            file.close()
            mlflow.log_artifact(file_path)
            destination_file = f'{project_path}/models/{model_name}/config/embedding_layer_name.txt'
            shutil.copy(file_path, destination_file)
        
        if is_items_preds:
            estimator.evaluate(
                input_fn=eval_input_fn,
                steps=None,
                hooks=[EvalHooks_inf(vocab, mlflow_params['max_predictions_per_seq'], predictions_per_user, project_path, model_name)]) 
        
        mlflow.log_artifact(checkpoint_dir)


def load_params(bert_config_file, train_config_file):
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    bert_config_json = json.loads(bert_config.to_json_string())
    vocab_size = bert_config_json['vocab_size']
    max_seq_length = bert_config_json['max_position_embeddings']
    hidden_size = bert_config_json['hidden_size']

    with open(train_config_file, "r") as f:
        train_config = json.load(f)
    max_predictions_per_seq = train_config['max_predictions_per_seq']
    num_warmup_steps = train_config['num_warmup_steps']
    num_train_steps = train_config['num_train_steps']
    learning_rate = train_config['learning_rate']
    save_checkpoints_steps = train_config['save_checkpoints_steps']
    model_name = train_config['model_name']

    mlflow_params = {
                    'project': project,
                    'model_name': model_name,
                    'max_seq_length': max_seq_length,
                    'max_predictions_per_seq': max_predictions_per_seq,
                    'predictions_per_user':  predictions_per_user,
                    'is_embeddings': is_embeddings,
                    'is_items_preds': is_items_preds,
                    'vocab_size': vocab_size,
                    'hidden_size': hidden_size
    }
    return mlflow_params, num_warmup_steps, num_train_steps, save_checkpoints_steps, learning_rate, model_name



if __name__ == "__main__":

    args = parse_args()

    use_tpu = args.use_tpu
    batch_size_val = args.batch_size
    predictions_per_user = args.predictions_per_user
    is_embeddings = args.is_embeddings
    is_items_preds = args.is_items_preds
    model_name = args.model_name

    is_training = False
    is_evaluate = True
    init_checkpoint = None

    file = open("config/project_run_now.txt", "r")
    project = file.read().rstrip().split('\n')[0]
    file.close()

    project_path = f'projects/{project}'
    if not os.path.exists(project_path):
        os.makedirs(project_path)

    test_input_file = f'{project_path}/data/processed/{project}.inference.tfrecord'
    vocab_filename = f'{project_path}/references/{project}.vocab'

    if model_name is None:
        # загрузка параметров последней обученной модели
        bert_config_file = f'{project_path}/config/bert_config_{project}.json'
        train_config_file = f'{project_path}/config/train_config_{project}.json'
    else:
        bert_config_file = f'{project_path}/models/{model_name}/config/bert_config_{project}.json'
        train_config_file = f'{project_path}/models/{model_name}/config/train_config_{project}.json'

    mlflow_params, num_warmup_steps, num_train_steps, save_checkpoints_steps, learning_rate, model_name = \
        load_params(bert_config_file, train_config_file)

    checkpoint_dir = f'{project_path}/models/{model_name}'

    inference_config_path = f'{project_path}/models/{model_name}/config/inference_config_{project}.json'
    with open(inference_config_path, "w") as f:
        json.dump(mlflow_params, f, indent=4)

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.compat.v1.app.run()