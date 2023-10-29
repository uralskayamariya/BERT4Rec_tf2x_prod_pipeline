import pickle
import mlflow
from dotenv import load_dotenv
load_dotenv()


def load_pkl(path):
    with open(path, 'rb') as inp:
        return pickle.load(inp)


def save_pkl(path, df):
    with open(path,'wb') as out:
        pickle.dump(df, out)


def yaml_load(yaml, config_name, project_path):
    with open(f'{project_path}/config/{config_name}.yaml', 'r') as file:
        config = yaml.load(file)
    return config


def get_output_file(filename):
    if filename is None:
        return None
    return open(filename, 'w')


def mlflow_dvc_definition(run, stage, experiment_name, config, yaml, project_path):
    run_id = run.info.run_id
    config[stage] = {}
    config[stage]['experiment_name'] = experiment_name
    config[stage]['run_id'] = run_id

    with open(f'{project_path}/config/config_mlflow.yaml', 'w') as file:
        yaml.dump(config, file)
    mlflow.log_artifact(f'{project_path}/config/config_mlflow.yaml')