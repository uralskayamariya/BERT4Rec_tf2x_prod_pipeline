import os
import pickle
import random
import collections
import mlflow
import argparse
from ruamel.yaml import YAML
import json
from src.utils.vocab import FreqVocab
from src.bert4rec.data_processing import gen_samples
from src.utils.utils import mlflow_dvc_definition
from src.utils.utils import yaml_load
from src.bert4rec.data_processing import data_partition
import src.bert4rec.modeling as modeling

from dotenv import load_dotenv
load_dotenv()

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_predictions_per_seq', type=int, default=20, help='Длина выходной последовательности, которая будет предсказана моделью.')
    parser.add_argument('--masked_lm_prob', type=float, default=0.2, help='Доля элементов, которые будут замаскированы во входной последовательности. \
                                                                           По умолчанию установлено значение 0.2, что означает, что около 20% элементов будет замаскировано.')
    parser.add_argument('--prop_sliding_window', type=float, default=0.5, help='Определяет размер шага, с которым окно будет скользить по последовательности элементов, \
                                                                                представляет собой долю от максимальной длины, которую должно занимать окно. \
                                                                                Если prop_sliding_window равно -1.0 , то окно будет иметь размер max_seq_length , то есть не будет скользить.')
    parser.add_argument('--mask_prob', type=float, default=1.0, help='Определяет вероятность замаскирования каждого элемента во входной последовательности. \
                                                                      Например, если  mask_prob  равно 0.2, то около 20% элементов будет замаскировано. \
                                                                      В процессе создания обучающих примеров, модель bert4rec использует механизм маскирования, \
                                                                      где некоторые элементы во входной последовательности замаскированы, \
                                                                      чтобы модель могла предсказать их и восстановить исходную последовательность.')
    parser.add_argument('--dupe_factor', type=int, default=10, help='Используется для определения количества раз, \
                                                                    которое каждая последовательность элементов будет использоваться для создания обучающих примеров с разными масками.')
    parser.add_argument('--pool_size', type=int, default=12, help='Используется для определения количества процессов, которые будут использоваться для параллельного выполнения операций.')
    parser.add_argument('--short_seq_prob', type=int, default=0, help='В процессе создания обучающих примеров, модель bert4rec использует двухпредложное представление, \
                                                                       где каждая входная последовательность элементов разделяется на две части.  \
                                                                       short_seq_prob  определяет вероятность того, что входная последовательность будет сокращена до короткого образца, \
                                                                       состоящего только из одной части исходной последовательности. \
                                                                       Например, если  short_seq_prob  равно 0.2, то около 20% входных последовательностей будет сокращено до короткого образца.')
    parser.add_argument('--task', type=str, default='train_val_test', choices=['train_val_test', 'train_val', 'test', 'inference'], help="Название задачи, для которой необходимо подготовить данные \
                                                                                                                                    ('train_val_test', 'train_val', 'test', 'inference').")
    parser.add_argument('--vocab_filename', type=str, default=None, help='Путь к файлу словаря товаров, если он уже существует, или `None`, если требуется создать новый словарь.')

    args = parser.parse_args()

    return args


def save_vocab(vocab):
    vocab_path = f"projects/{project}/references"
    if not os.path.exists(vocab_path):
        os.makedirs(vocab_path)
    vocab_file_name = output_dir_vocab + '/' + project + '.vocab'
    print('vocab pickle file: ' + vocab_file_name)
    with open(vocab_file_name, 'wb') as output_file:
        pickle.dump(vocab, output_file, protocol=2)
    return vocab_file_name


def test_inference_data_prep(dataset, vocab_filename):
    for idx, u in enumerate(dataset):
        if idx < 3:
            print(f'user_{task}: {dataset[u]}')

    user_data = {
                    'user_' + str(k): ['item_' + str(item) for item in v]
                    for k, v in dataset.items() if len(v) > 0
    }

    if vocab_filename:
        try:
            with open(vocab_filename, 'rb') as input_file:
                vocab = pickle.load(input_file)
            mlflow.log_artifact(vocab_filename)
        except:
            print('Словарь по указанному пути не найден, создаем новый словарь.')
            vocab = FreqVocab(user_data)
            vocab_file_name = save_vocab(vocab)
            mlflow.log_artifact(vocab_file_name)
    else:
        vocab = FreqVocab(user_data)
        vocab_file_name = save_vocab(vocab)
        mlflow.log_artifact(vocab_file_name)
    
    # user_data_output = {
    #                     k: [vocab.convert_tokens_to_ids(v)]
    #                     for k, v in user_data.items()
    # }

    print(f'Begin to generate {task} examples...')
    processed_path = f"{output_dir_data}/processed"
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    output_filename = f'{processed_path}/{project}.{task}.tfrecord'
    print(f'{task} output_filename:{output_filename}')

    gen_samples(
        user_data,
        output_filename,
        rng,
        vocab,
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        mask_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        -1.0,
        pool_size,
        force_last=True
    )
    return vocab


def train_val_test_data_prep(dataset, vocab_filename):
    [user_train, user_valid, user_test] = dataset
    for idx, u in enumerate(user_train):
        if idx < 3:
            print(f'user_train: {user_train[u]}')
            print(f'user_valid: {user_valid[u]}')
            print(f'user_test: {user_test[u]}')

    # put validate into train
    for u in user_train:
        if u in user_valid:
            user_train[u].extend(user_valid[u])

    # get the max index of the data
    user_train_data = {
        'user_' + str(k): ['item_' + str(item) for item in v]
        for k, v in user_train.items() if len(v) > 0
    }
    user_test_data = {
        'user_' + str(u):
            ['item_' + str(item) for item in (user_train[u] + user_test[u])]
        for u in user_train if len(user_train[u]) > 0 and len(user_test[u]) > 0
    }

    if vocab_filename:
        with open(vocab_filename, 'rb') as input_file:
            vocab = pickle.load(input_file)
        mlflow.log_artifact(vocab_filename)
    else:
        vocab = FreqVocab(user_test_data)
        vocab_file_name = save_vocab(vocab)
        mlflow.log_artifact(vocab_file_name)

    # user_data_output = {
    #     k: [vocab.convert_tokens_to_ids(v)]
    #     for k, v in user_test_data.items()
    # }

    print('Begin to generate train, val...')
    processed_path = f"{output_dir_data}/processed"
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    output_filename = f'{processed_path}/{project}.train_val_test.train_val.tfrecord'
    gen_samples(
        user_train_data,
        output_filename,
        rng,
        vocab,
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        mask_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        prop_sliding_window,
        pool_size,
        force_last=False)
    print('train:{}'.format(output_filename))

    print('Begin to generate test...')
    processed_path = f"{output_dir_data}/processed"
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    output_filename = f'{processed_path}/{project}.train_val_test.test.tfrecord'
    gen_samples(
        user_test_data,
        output_filename,
        rng,
        vocab,
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        mask_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        -1.0,
        pool_size,
        force_last=True)
    print('test:{}'.format(output_filename))

    return vocab


def train_val_data_prep(dataset, vocab_filename):
    [user_train, user_valid] = dataset
    for idx, u in enumerate(user_train):
        if idx < 3:
            print(f'user_train: {user_train[u]}')
            print(f'user_valid: {user_valid[u]}')

    # put validate into train
    for u in user_train:
        if u in user_valid:
            user_train[u].extend(user_valid[u])

    # get the max index of the data
    user_train_data = {
        'user_' + str(k): ['item_' + str(item) for item in v]
        for k, v in user_train.items() if len(v) > 0
    }
    
    if vocab_filename:
        with open(vocab_filename, 'rb') as input_file:
            vocab = pickle.load(input_file)
        mlflow.log_artifact(vocab_filename)
    else:
        vocab = FreqVocab(user_train_data)
        vocab_file_name = save_vocab(vocab)
        mlflow.log_artifact(vocab_file_name)

    # user_data_output = {
    #     k: [vocab.convert_tokens_to_ids(v)]
    #     for k, v in user_train_data.items()
    # }

    print(f'Begin to generate {task}...')
    processed_path = f"{output_dir_data}/processed"
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    output_filename = f'{processed_path}/{project}.train_val.tfrecord'
    gen_samples(
        user_train_data,
        output_filename,
        rng,
        vocab,
        max_seq_length,
        dupe_factor,
        short_seq_prob,
        mask_prob,
        masked_lm_prob,
        max_predictions_per_seq,
        prop_sliding_window,
        pool_size,
        force_last=False)
    print(f'{task}:{output_filename}')

    return vocab


def main():

    os.makedirs(output_dir_data, exist_ok=True)

    mlflow.set_tracking_uri(os.environ.get('MLFLOW_URI'))

    yaml = YAML()

    config = yaml_load(yaml, 'config_mlflow', project_path)

    experiment_name = config['last_experiment_name']

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=stage) as run:
        mlflow.log_params(mlflow_params)
        mlflow_dvc_definition(run, stage, experiment_name, config, yaml, project_path)

        dataset = data_partition(fname, task)

        if task in ['test', 'inference']:
            vocab = test_inference_data_prep(dataset, vocab_filename)
        elif task == 'train_val_test':
            vocab = train_val_test_data_prep(dataset, vocab_filename)
        else:
            vocab = train_val_data_prep(dataset, vocab_filename)

        vocab_size = vocab.get_vocab_size()
        mlflow.log_params({'vocab_size': vocab_size})
        bert_config_json["vocab_size"] = vocab_size
        bert_config_path = f"{project_path}/config/bert_config_{project}.json"
        with open(bert_config_path, "w") as f:
            json.dump(bert_config_json, f, indent=4)
        mlflow.log_artifact(bert_config_path)

        print('vocab_size:{}, user_size:{}, item_size:{}, item_with_other_size:{}'.
            format(vocab_size,
                    vocab.get_user_count(),
                    vocab.get_item_count(),
                    vocab.get_item_count() + vocab.get_special_token_count()))

        print('done.')


if __name__ == '__main__':
    args = parse_args()

    max_predictions_per_seq = args.max_predictions_per_seq
    masked_lm_prob = args.masked_lm_prob
    mask_prob = args.mask_prob
    dupe_factor = args.dupe_factor
    pool_size = args.pool_size
    short_seq_prob = args.short_seq_prob
    task = args.task
    vocab_filename = args.vocab_filename
    prop_sliding_window = args.prop_sliding_window

    file = open("config/project_run_now.txt", "r")
    project = file.read().rstrip().split('\n')[0]
    file.close()

    project_path = f'projects/{project}'
    if not os.path.exists(project_path):
        os.makedirs(project_path)

    output_dir_data = f'{project_path}/data'
    output_dir_vocab = f'{project_path}/references'

    fname = output_dir_data + '/' + 'interim/' + project + '_' + task + '.txt'

    bert_config_file = 'config/bert_config.json'
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    bert_config_json = json.loads(bert_config.to_json_string())
    max_seq_length = bert_config_json['max_position_embeddings']
    hidden_size = bert_config_json['hidden_size']

    mlflow_params = {
                    'project': project,
                    'max_seq_length': max_seq_length,
                    'masked_lm_prob': masked_lm_prob,
                    'max_predictions_per_seq': max_predictions_per_seq,
                    'mask_prob': mask_prob,
                    'dupe_factor': dupe_factor,
                    'pool_size': pool_size,
                    'short_seq_prob': short_seq_prob,
                    'hidden_size': hidden_size,
                    'prop_sliding_window': prop_sliding_window,
                    'task': task
    }

    data_config_path = f"{project_path}/config/data_config_{project}.json"
    with open(data_config_path, "w") as f:
        json.dump(mlflow_params, f, indent=4)

    random_seed = 12345
    rng = random.Random(random_seed)

    stage = f'data_processing_all_{task}'
    main()
