import mlflow
import os
import datetime
from ruamel.yaml import YAML
import subprocess
import os
from dotenv import load_dotenv
load_dotenv()


def make_experiment():
    # создание проекта
    file = open("config/project_run_now.txt", "r")
    project = file.read().rstrip().split('\n')[0]
    file.close()
    # создание эксперимента
    file = open(f"config/new_experiment_stage.txt", "r")
    new_experiment_stage = file.read().rstrip().split('\n')[0]
    file.close()
    # получение имени текущей ветки
    git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf-8')
    experiment_name = f"bert4rec_{new_experiment_stage}_{datetime.datetime.now()}_{git_branch}"
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_URI'))
    mlflow.create_experiment(experiment_name)
    
    return new_experiment_stage, experiment_name, project


def update_config_mlflow(yaml, experiment_name, project):
    config_dir = f'projects/{project}/config'
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    file_path = f'{config_dir}/config_mlflow.yaml'
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            file.write("last_experiment_name:")
        print(f"Файл {file_path} успешно создан.")
    # обновление общего конфигурационного файла
    with open(file_path, 'r') as file:
        config = yaml.load(file)

    config['last_experiment_name'] = experiment_name

    with open(file_path, 'w') as file:
        yaml.dump(config, file)

    return config


def main():
    new_experiment_stage, experiment_name, project = make_experiment()
    yaml = YAML()
    config = update_config_mlflow(yaml, experiment_name, project)



if __name__ == '__main__':
    main()
