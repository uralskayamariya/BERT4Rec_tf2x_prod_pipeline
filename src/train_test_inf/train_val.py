import os
import pickle
import tensorflow as tf
import mlflow
import argparse
from ruamel.yaml import YAML
import json
import src.bert4rec.modeling as modeling
from src.utils.vocab import FreqVocab
from src.utils.utils import get_output_file
from src.bert4rec.main_processes import input_fn_builder, model_fn_builder, TrainHooks, EvalHooks
from dotenv import load_dotenv
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, help='Имя директории, где будут сохраняться чекпоинты модели')
    parser.add_argument('--save_checkpoints_steps', type=int, default=1000, help='Частота сохранения контрольных точек во время обучения модели. \
                                                                                  Сохраняются только 5 последних чекпоинтов, более ранние удаляются автоматически.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Скорость обучения модели.')
    parser.add_argument('--num_train_steps', type=int, default=200000, help='Количество шагов обучения модели.')
    parser.add_argument('--num_warmup_steps', type=int, default=100, help='Количество шагов прогрева, в течение которых скорость обучения постепенно увеличивается.')
    parser.add_argument('--use_tpu', type=bool, default=False, help='Флаг, указывающий, будет ли использоваться TPU для обучения модели.')
    parser.add_argument('--batch_size_train', type=int, default=32, help='Размер батча для обучения.') # 32
    parser.add_argument('--batch_size_val', type=int, default=64, help='Размер батча для валидации.') # 128
    parser.add_argument('--is_train', type=bool, default=True, help='Флаг, указывающий, будет ли модель обучаться. Если модель уже существует в указанной директории, то обучение не будет выполняться, только валидация.') 
    parser.add_argument('--is_evaluate', type=bool, default=False, help='Флаг, указывающий, будет ли выполняться валидация модели. Это может занять длительное время.') 
    parser.add_argument('--fast_eval', type=bool, default=True, help='Флаг, который решает проблему длительной валидации. При значении True валидация происходит только для первой строки в батче.') 
    parser.add_argument('--save_predictions_file', type=bool, default=True, help='Флаг, указывающий, сохранять ли предсказания модели на валидационном наборе данных. Файл может оказаться очень большого размера. \
                                                                                   Если значение True, предсказания модели сохраняются в projects/<имя_проекта>/results/<model_name>/eval_results.txt.')
    parser.add_argument('--training_time_limit_seconds', type=int, default=None, help='Ограничение времени обучения модели в секундах.')
    args = parser.parse_args()  

    return args


def main(_):

    stage = 'train_val'

    mlflow.set_tracking_uri(os.environ.get('MLFLOW_URI'))
    yaml = YAML()

    with open(f'{project_path}/config/config_mlflow.yaml', 'r') as file:
        config = yaml.load(file)

    experiment_name = config['last_experiment_name']

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=stage) as run:
        
        # need to check !!!!!!!!!!!!!!!!!!!!
        # mlflow.tensorflow.autolog(every_n_iter=1, 
        #                           log_models=True, 
        #                           log_datasets=False, 
        #                           disable=False, 
        #                           exclusive=False, 
        #                           disable_for_unsupported_versions=False, 
        #                           silent=False, 
        #                           registered_model_name='bert4rec', 
        #                           log_input_examples=False, 
        #                           log_model_signatures=True, 
        #                           saved_model_kwargs=None, 
        #                           keras_model_kwargs=None, 
        #                           extra_tags=None)

        run_id = run.info.run_id
        config[stage] = {}
        config[stage]['experiment_name'] = experiment_name
        config[stage]['run_id'] = run_id

        # запись изменений в файл, оставляя только выбранные стейджи
        with open(f'{project_path}/config/config_mlflow.yaml', 'w') as file:
            yaml.dump(config, file)

        mlflow.log_artifact(f'{project_path}/config/config_mlflow.yaml')
        mlflow.log_params(mlflow_params)
        mlflow.log_artifact(bert_config_file)
    
        tf.compat.v1.gfile.MakeDirs(checkpoint_dir)

        train_input_files = []
        for input_pattern in train_input_file.split(","):
            train_input_files.extend(tf.compat.v1.gfile.Glob(input_pattern))

        print("*** Train, val input files ***")
        for input_file in train_input_files:
            print("  %s" % input_file)

        run_config = tf.estimator.RunConfig(
            model_dir=checkpoint_dir,
            save_checkpoints_steps=save_checkpoints_steps)

        if vocab_filename is not None:
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
        
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={
                "batch_size_train": batch_size_train,
                "batch_size_val": batch_size_val
            })


        if is_train:
            print("\n***** Running training *****\n")
            print("  Batch size = %d" % batch_size_train)

            train_input_fn = input_fn_builder(
                input_files=train_input_files,
                max_seq_length=max_seq_length,
                max_predictions_per_seq=max_predictions_per_seq,
                is_training=True)
            
            estimator.train(
                input_fn=train_input_fn, 
                        max_steps=num_train_steps,
                        hooks=[TrainHooks(training_time_limit_seconds)]) # для остановки по времени обучения

        mlflow.log_artifact(checkpoint_dir)
            
        if is_evaluate:
            print("\n***** Running evaluation *****\n")
            print("  Batch size = %d" % batch_size_val)

            eval_input_fn = input_fn_builder(
                input_files=train_input_files,
                max_seq_length=max_seq_length,
                max_predictions_per_seq=max_predictions_per_seq,
                is_training=False)
            
            output_file = None
            output_file = get_output_file(save_predictions_file)
            
            # tf.logging.info('special eval ops:', special_eval_ops)
            result = estimator.evaluate(
                input_fn=eval_input_fn,
                steps=None,
                hooks=[EvalHooks(output_file, vocab, is_evaluate, max_predictions_per_seq, fast_eval)])
            
            results_path = f"{project_path}/results/{model_name}"
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            output_eval_file = f'{results_path}/eval_results.txt'
            with tf.compat.v1.gfile.GFile(output_eval_file, "w") as writer:
                print("\n***** Eval results *****\n")
                print(bert_config.to_json_string())
                writer.write(bert_config.to_json_string() + '\n')
                for key in sorted(result.keys()):
                    print("%s = %s" % (key, str(result[key])))
                    writer.write("%s = %s\n" % (key, str(result[key])))

            mlflow.log_artifact(output_eval_file)

        # сохраним конфигурационные файлы в папку модели
        config_path = f"{project_path}/models/{model_name}/config"
        if not os.path.exists(config_path):
            os.makedirs(config_path)

        train_config_path = f"{config_path}/train_config_{project}.json"
        with open(train_config_path, "w") as f:
            json.dump(mlflow_params, f, indent=4)
        mlflow.log_artifact(train_config_path)

        bert_config_path = f"{config_path}/bert_config_{project}.json"
        with open(bert_config_path, "w") as f:
            json.dump(bert_config_json, f, indent=4)

        data_config_path = f"{config_path}/data_config_{project}.json"
        with open(data_config_path, "w") as f:
            json.dump(data_config, f, indent=4)


if __name__ == "__main__":

    args = parse_args()

    model_name = args.model_name
    save_checkpoints_steps = args.save_checkpoints_steps
    learning_rate = args.learning_rate
    num_train_steps = args.num_train_steps
    num_warmup_steps = args.num_warmup_steps
    use_tpu = args.use_tpu
    batch_size_train = args.batch_size_train
    batch_size_val = args.batch_size_val
    is_train = args.is_train
    is_evaluate = args.is_evaluate
    fast_eval = args.fast_eval
    training_time_limit_seconds = args.training_time_limit_seconds
    save_predictions_file = args.save_predictions_file
    model_name = args.model_name
    
    init_checkpoint = None

    file = open("config/project_run_now.txt", "r")
    project = file.read().rstrip().split('\n')[0]
    file.close()

    project_path = f'projects/{project}'
    if not os.path.exists(project_path):
        os.makedirs(project_path)

    with open(f'{project_path}/config/data_config_{project}.json', "r") as f:
        data_config = json.load(f)
    max_predictions_per_seq = data_config['max_predictions_per_seq']
    task = data_config['task']

    if task == 'train_val':
        train_input_file = f'{project_path}/data/processed/{project}.train_val.tfrecord'
    else:
        train_input_file = f'{project_path}/data/processed/{project}.train_val_test.train_val.tfrecord'

    vocab_filename = f'{project_path}/references/{project}.vocab'

    bert_config_file = f'{project_path}/config/bert_config_{project}.json'
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    bert_config_json = json.loads(bert_config.to_json_string())
    vocab_size = bert_config_json['vocab_size']
    max_seq_length = bert_config_json['max_position_embeddings']
    hidden_size = bert_config_json['hidden_size']

    if model_name is None:
        model_name = f'hs{hidden_size}_msl{max_seq_length}_mpps{max_predictions_per_seq}_nts{num_train_steps}_lr{learning_rate}'

    if save_predictions_file:
        results_path = f"{project_path}/results/{model_name}"
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        save_predictions_file = f'{project_path}/results/{model_name}/preds_val.txt'
    else:
        save_predictions_file = None

    checkpoint_dir = f'{project_path}/models/{model_name}'

    mlflow_params = {
                    'project': project,
                    'model_name': model_name,
                    'hidden_size': hidden_size,
                    'learning_rate': learning_rate,
                    'num_train_steps': num_train_steps,
                    'batch_size_train': batch_size_train,
                    'batch_size_val': batch_size_val,
                    'max_seq_length': max_seq_length,
                    'max_predictions_per_seq': max_predictions_per_seq,
                    'training_time_limit_seconds': training_time_limit_seconds,
                    'vocab_size': vocab_size,
                    'num_warmup_steps': num_warmup_steps,
                    'save_checkpoints_steps': save_checkpoints_steps,
                    'fast_eval': fast_eval
    }

    train_config_path = f"{project_path}/config/train_config_{project}.json"
    with open(train_config_path, "w") as f:
        json.dump(mlflow_params, f, indent=4)

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.compat.v1.app.run()