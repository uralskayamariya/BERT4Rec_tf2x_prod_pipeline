
import re
from ruamel.yaml import YAML
import os
import argparse
from dotenv import load_dotenv
load_dotenv() 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='final_model', help='Имя директории, где будут сохраняться чекпоинты модели')
    parser.add_argument('--is_embeddings', type=bool, default=True, help='Создать ли файл с эмбеддингами предпоследнего слоя модели. \
                                                                         Эмбеддинги сохраняются по адресу: projects/<имя_проекта>/results/<model_name>/df_next_sids_embs.pkl.')
    parser.add_argument('--is_items_preds', type=bool, default=True, help='Создать ли файл с предсказаниями товаров. \
                                                                            Сохраняется по адресу: projects/<имя_проекта>/results/<model_name>/df_next_sids.pkl.')
    args = parser.parse_args()

    return args


def main():

    args = parse_args()
    model_name = args.model_name
    is_embeddings = args.is_embeddings
    is_items_preds = args.is_items_preds

    file = open("config/project_run_now.txt", "r")
    project = file.read().rstrip().split('\n')[0]
    file.close()

    yaml = YAML()

    with open('config/dvc_template.yaml', 'r') as file:
        dvc_file = yaml.load(file)

    all_stages = dict(dvc_file['stages'])

    for stage in all_stages:
        for stage_part in all_stages[stage]:
            if stage_part == 'cmd':
                stage_path = all_stages[stage][stage_part]
                new_path = re.sub(r'{project}', project, stage_path)
                new_path = re.sub(r'{model_name}', model_name, new_path)
                dvc_file['stages'][stage][stage_part] = new_path
            else:
                for i, stage_path in enumerate(all_stages[stage][stage_part]): 
                    new_path = re.sub(r'{project}', project, stage_path)
                    new_path = re.sub(r'{model_name}', model_name, new_path)
                    dvc_file['stages'][stage][stage_part][i] = new_path

        if stage == 'inference':
            if is_embeddings or is_items_preds:
                list_values = []
                try: 
                    dvc_file['stages'][stage]['outs']
                except:
                    dvc_file['stages'][stage]['outs'] = {}
                if is_embeddings:
                    list_values.append(f'projects/{project}/results/{model_name}/df_next_sids_embs.pkl')
                if is_items_preds:
                    list_values.append(f'projects/{project}/results/{model_name}/df_next_sids.pkl')
                dvc_file['stages'][stage]['outs'] = list_values

    project_path = f'projects/{project}/config'
    if not os.path.exists(project_path):
        os.makedirs(project_path)

    yaml.indent(mapping=2, sequence=4, offset=2)

    with open(f'{project_path}/dvc.yaml', 'w') as file:
        yaml.dump(dvc_file, file)

    with open('dvc.yaml', 'w') as file:
        yaml.dump(dvc_file, file)
        


if __name__ == '__main__':
    main()