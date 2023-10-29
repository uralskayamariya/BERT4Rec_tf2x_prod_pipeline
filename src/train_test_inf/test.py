import os
import pickle
import json
import tensorflow as tf
import mlflow
import argparse
from ruamel.yaml import YAML
import src.bert4rec.modeling as modeling
from src.utils.vocab import FreqVocab
from src.utils.utils import get_output_file
from src.bert4rec.main_processes import input_fn_builder, model_fn_builder, EvalHooks
from dotenv import load_dotenv
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, help='Model name.')
    parser.add_argument('--use_tpu', type=bool, default=False, help='.')
    parser.add_argument('--batch_size', type=int, default=64, help='.')
    parser.add_argument('--save_predictions_file', type=bool, default=False, help='Сохранять ли предсказания модели на валидационном наборе данных. \
                                                                                   Файл может оказаться очень большого размера. \
                                                                                   Если значение True, предсказания модели сохраняются в projects/<имя_проекта>/results/<model_name>/test_results.txt.') 
    parser.add_argument('--fast_eval', type=bool, default=False, help='.')
    args = parser.parse_args()

    return args   


def main(_):
    stage = 'test'
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
        mlflow.log_artifact(test_config_path)

        bert_config = modeling.BertConfig.from_json_file(bert_config_file)

        tf.compat.v1.gfile.MakeDirs(checkpoint_dir)

        test_input_files = []
        for input_pattern in test_input_file.split(","):
            test_input_files.extend(tf.compat.v1.gfile.Glob(input_pattern))

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

        print("\n***** Running evaluation *****\n")
        print("  Batch size = %d" % batch_size_val)

        eval_input_fn = input_fn_builder(
            input_files=test_input_files,
            max_seq_length=mlflow_params['max_seq_length'],
            max_predictions_per_seq=mlflow_params['max_predictions_per_seq'],
            is_training=False)
        
        #tf.logging.info('special eval ops:', special_eval_ops)
        output_file = None
        output_file = get_output_file(save_predictions_file)
        
        result = estimator.evaluate(
            input_fn=eval_input_fn,
            steps=None,
            hooks=[EvalHooks(output_file, vocab, is_evaluate, mlflow_params['max_predictions_per_seq'], fast_eval)])
        
        results_path = f"{project_path}/results/{model_name}"
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        output_eval_file = f'{results_path}/test_results.txt'
        with tf.compat.v1.gfile.GFile(output_eval_file, "w") as writer:
            print("\n***** Eval results *****\n")
            print(bert_config.to_json_string())
            writer.write(bert_config.to_json_string() + '\n')
            for key in sorted(result.keys()):
                print("%s = %s" % (key, str(result[key])))
                writer.write("%s = %s\n" % (key, str(result[key])))

        mlflow.log_artifact(output_eval_file)
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
                    'vocab_size': vocab_size,
                    'fast_eval': fast_eval,
                    'hidden_size': hidden_size
    }
    return mlflow_params, num_warmup_steps, num_train_steps, save_checkpoints_steps, learning_rate, model_name


if __name__ == "__main__":

    args = parse_args()

    use_tpu = args.use_tpu
    batch_size_val = args.batch_size
    save_predictions_file = args.save_predictions_file
    fast_eval = args.fast_eval
    use_tpu = args.use_tpu
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

    with open(f'{project_path}/models/{model_name}/config/data_config_{project}.json', "r") as f:
        data_config = json.load(f)
    task = data_config['task']

    if task == 'train_val_test':
        test_input_file = f'{project_path}/data/processed/{project}.train_val_test.test.tfrecord'
    else:
        test_input_file = f'{project_path}/data/processed/{project}.test.tfrecord'

    vocab_filename = f'{project_path}/references/{project}.vocab'

    if save_predictions_file:
        save_predictions_file = f'{project_path}/results/{model_name}/preds_test.txt'
    else:
        save_predictions_file = None

    test_config_path = f'{project_path}/models/{model_name}/config/test_config_{project}.json'
    with open(test_config_path, "w") as f:
        json.dump(mlflow_params, f, indent=4)

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.compat.v1.app.run()