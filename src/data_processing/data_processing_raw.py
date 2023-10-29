import os
import mlflow
import argparse
import pandas as pd
# Установка ширины столбцов на 100 символов
pd.set_option('display.max_colwidth', 100)
from ruamel.yaml import YAML
import pickle
from sklearn.preprocessing import LabelEncoder
from src.utils.utils import load_pkl
from src.utils.utils import mlflow_dvc_definition
from src.utils.utils import yaml_load
from dotenv import load_dotenv
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='train_val_test', choices=['train_val_test', 'train_val', 'test', 'inference'], 
                        help="Задача, для которой необходимо подготовить данные: 'train_val_test', 'train_val', 'test', 'inference'.")
    parser.add_argument('--drop_duplicates', type=bool, default=True, help='Удаление дубликатов товаров, которые идут по-порядку.')
    parser.add_argument('--vocab_exist', type=int, default=0, choices=[0, 1],help='1, если нужно использовать уже существующий словарь кодировки номеров пользователей. \
                                        Словарь должен находиться по адресу: projects/<имя_проекта>/referenses/dics/dic_encoding_uid_user_id.pkl')   
    args = parser.parse_args()

    return args


def check_len_seq_sids(lst, len_seq_sids):
    return len(lst) >= len_seq_sids


def make_sids_list(x):
    list_no_double = []
    for i in range(len(x)):
        if i > 0:
            if x[i] != x[i-1]:
                list_no_double.append(x[i-1])
            if i == len(x) - 1:
                list_no_double.append(x[i])
    
    return list_no_double


def code_uid(df_raw):
    le = LabelEncoder()
    uid_unique = df_raw.uid.unique()
    le.fit(uid_unique)
    dic_col = {l: i for (i, l) in enumerate(le.classes_)}
    df_raw['uid'] = df_raw['uid'].apply(lambda x: dic_col[x])
    # сохраним словарь кодировки
    file_path = f'{project_path}/references/dics'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open(f'{file_path}/dic_encoding_uid_user_id.pkl', 'wb') as out:
        pickle.dump(dic_col, out)
    mlflow.log_artifact(f'{file_path}/dic_encoding_uid_user_id.pkl')
    df_raw = df_raw.sort_values('uid').reset_index(drop=True)

    return df_raw


def main():
    
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_URI'))

    yaml = YAML()

    config = yaml_load(yaml, 'config_mlflow', project_path)

    experiment_name = config['last_experiment_name']

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=stage) as run:
        mlflow_dvc_definition(run, stage, experiment_name, config, yaml, project_path)

        mlflow.log_params(mlflow_params)

        df_raw = load_pkl(f'{project_path}/data/external/{project}_{task}.pkl')
        print('Первоначальный датасет:')
        print(df_raw)

        if vocab_exist:
            vocab = load_pkl(f'{project_path}/references/dics/dic_encoding_uid_user_id.pkl')
            df_raw['uid'] = df_raw.uid.apply(lambda x: vocab[x])
        else:
            df_raw = code_uid(df_raw)

        if drop_duplicates:
            # исключение дубликатов, если они идут друг за другом
            df_raw['sid_add_to_cart'] = df_raw.sid_add_to_cart.apply(lambda x: make_sids_list(x))

        df_raw = df_raw[df_raw['sid_add_to_cart'].apply(lambda x: check_len_seq_sids(x, min_len_seq_sids))].reset_index(drop=True)

        # для задачи инференса добавим в конец датафрема дубликат последнего товара, 
        # чтобы дальнейшая подготовка данных была идентична тестированию
        if task == 'inference':
            df_raw['sid_add_to_cart'] = df_raw.sid_add_to_cart.apply(lambda x: x + [x[-1]])

        # Создание списка пользователей и идентификаторов товаров
        product_list = []
        for i, row in df_raw.iterrows():
            uid = row['uid']
            for product in row['sid_add_to_cart']:
                product_list.append(f'{uid} {product}\n')
        print(f'Первые 100 позиций преобразованного датасета:\n {product_list[:100]}')

        interim_path = f"projects/{project}/data/interim"
        if not os.path.exists(interim_path):
            os.makedirs(interim_path)
        filename = f"{interim_path}/{project}_{task}.txt"
        with open(filename, "w") as file:
            for item in product_list:
                file.write(item)

        print("Список успешно сохранен в файл", filename)


if __name__ == '__main__':

    args = parse_args()

    task = args.task
    drop_duplicates = args.drop_duplicates
    vocab_exist = bool(args.vocab_exist)

    file = open("config/project_run_now.txt", "r")
    project = file.read().rstrip().split('\n')[0]
    file.close()

    project_path = f'projects/{project}'
    if not os.path.exists(project_path):
        os.makedirs(project_path)

    if task == 'train_val_test':
        min_len_seq_sids = 3
    elif task in ['train_val', 'test']:
        min_len_seq_sids = 2
    else:
        min_len_seq_sids = 1
    
    mlflow_params = {'project': project, 
                  'min_len_seq_sids': min_len_seq_sids,
                  'drop_duplicates': drop_duplicates}

    stage = f'data_processing_raw_{task}'
    main()


