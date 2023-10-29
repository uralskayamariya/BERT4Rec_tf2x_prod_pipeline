# BERT4rec: модель предсказания следующих элементов цепочки закодированных элементов (товаров, фильмов и т.п.)

В данном репозитории собран пайплайн обучения, валидации, тестирования и инференса модели BERT4rec для рекомендательных систем.
На вход модели подается любая цепочка закодированных элементов, на выходе получаем заданное количество следующих элементов.
Например, в качестве входной последовательности может быть цепочка купленных пользователем 10 товаров, расположенных в порядке возрастания очередности покупки. Модель на основании этой последовательности предсказывает 11, 12 и т.д. товар, который пользователь с наибольшей вероятностью купит в следующий раз.
Размеры входной и выходной цепочки задаются параметрами.

# Требования к компьютеру
На компьютере необходимо наличие видеокарты. Репозиторий протестирован на:
1. Windows с Nvidia RTX 2080 Super и Cuda Toolkit 11.4 
2. Linux Ubuntu 22.04 с NVIDIA GeForce RTX 3090 Ti и Cuda Toolkit 11.7

# Описание версий, библиотек, структуры репозитория
Репозиторий протестирован на python 3.9, 3.10.
<br>Обучение модели происходит с помощью фреймворка tensorflow.
Проверено на версиях tensorflow-gpu 2.4, 2.6, 2.10.
<br>В репозитории используется работа с dvc и mlflow.
<br>Репозиторий имеет структуру в соответствии с [Cookiecutter Data Science](https://github.com/drivendata/cookiecutter-data-science). Однако, так как репозиторий предназначен для многократного использования, логично разделить его на проекты и в папке каждого проекта иметь структуру папок для хранения данных. 
<br>Поэтому в корне репозитория расположена папка projects, в которую добавляются папки с названиями проектов. Необходимо для каждого нового варианта подготовки данных создавать новый проект, так как это означает, что весь пайплайн будет выполняться сначала.
<br>Папка с каждым проектом содержит следующие подпапки:
- data
- references
- models
- results
- config
<br>В папке models создаются подпапки с названиями моделей, так как для одного проекта может быть обучено несколько моделей. Также автоматически создаются подпапки projects/<имя_проекта>/models/<имя_модели>/config для хранения конфигурационных файлов текущей версии модели.
<br><br>Код находится в корневой папке src:
- Подпапка bert4rec содержит код модели.
- Подпапка data_processing содержит код для кодготовки данных.
- Подпапка train_test_inf содержит код для обучения, валидации, тестирования и инференса модели.
- Подпапка utils содержит вспомогательный код для создания экспериментов mlflow. 
<br><br>Весь пайплайн в репозитории работает через dvc или makefile. 
<br>Результаты экспериментов сохраняются в mlflow.
<br>В папке config содержатся шаблоны конфигурационных файлов, а также временная информация о текущем названии проекта и стейджа dvc.

# Установка 
1. Откройте терминал.
2. Склонируйте проект: git clone <project_address>
3. Перейдите в папку проекта: cd BERT4Rec_tf2x_prod_pipeline
4. Создайте в корне репозитория и заполните файл .env переменными среды. Он должен выглядеть так:
<br>AWS_ACCESS_KEY_ID=<aws_access_key_id>
<br>AWS_SECRET_ACCESS_KEY=<aws_secret_access_key>
<br>MLFLOW_URI=<server_address>
<br>Если файл .env не будет создан, паплайн работать будет, но не будет работать запись в dvc и mlflow.

## В среде poetry
5. Установите poetry: pip install poetry==1.4.1
6. Необходимые для установки библиотеки прописаны в стандартном файле менеджера сред pyproject.toml, для их установки выполните команду: poetry install
7. Запустите среду проекта: poetry shell

## В среде anaconda
5. Создайте новую среду анаконда: conda create -n bert4rec python=3.10
6. Перейдите в установленную среду: conda activate bert4rec
7. Установите все необходимые библиотеки: pip install -r requirements.txt

<br>После установки среды poetry среда анаконды скорее всего не будет работать.

# Работа с DVC
<br>Так как большие файлы не сохраняются в git, придумали специальные хранилища, например, s3, работу с которыми можно осуществлять с помощью DVC.
<br>DVC также удобно, так как позволяет выполнять целые пайплайны, то есть цепочки файлов скриптов в разных комбинациях в зависимости от наличия файлов, созданных на предыдущих этапах пайплайнов или изменениях в зависимых файлах.
<br>То есть, например, если нужно выполнить инференс модели, можно запустить выполнение только последнего этапа в цепочке (инференс), при этом dvc проверит, есть ли файлы и изменения в них на этапах загрузки и подготовки данных, обучения модели, тестирования модели, проверит наличие файлов самой модели, и выполнит все эти предыдущие этапы, если отсутствуют зависимые файлы или есть изменения в коде. Для этого все необходимые зависимости должны быть прописаны в файле конфигурации dvc.yaml.
<br>Большие файлы, созданные в результате выполнения кода проекта, сохраняются как на диск, так и в хранилище dvc. Эти файлы можно удалять с диска, но при запуске кода, соответствующие файлы (из deps в файле dvc.yaml) будут загружаться из хранилища.
<br>Если отсутствует хранилище dvc-файлов, то файл .env можно не создавать. Данные в dvc сохраняться не будут, но стеджи (stages) dvc запускаться будут.
<br>Пайплайны в dvc называются stages, которые прописаны в файле dvc.yaml.
<br>Каждый стейдж состоит из 1 обязательной (cmd) и 2 необязательных частей (deps, outs).
<br>В cmd прописывается исполняемый файл.
<br>В deps прописываются зависимые файлы, наличие и выполнение (для файлов кода) которых необходимо контролировать перед выполнением текущего стейджа.
<br>Файлы, указанные в outs конфигурационного файла dvc (dvc.yaml), создаются в результате выполнения данного стейджа и автоматически коммитятся в хранилище dvc.
<br>Для разных стейджей файлы в outs не должны повторяться.
<br>Перед первым запуском стейджей необходимо сконфигурировать репозиторий в dvc: dvc init
<br>Запуск каждого стейджа выполняется командой: dvc repro <имя_стейджа>.
<br>Например:
<br>dvc repro data_processing_all
<br>При запуске данного стейджа запустится на выполнение файл src/data_processing/data_processing_all.py.
<br>Если необходимо использовать уже созданные датафреймы и модели, загрузите данные из хранилища s3 командой: dvc pull
<br>В процессе использования репозитория может потребоваться записать новый большой файл в хранилище, который не был записан автоматически. Это можно сделать следующими командами: 
- dvc add <file_path>. (DVC Создаст файл с мета информацией <file_path>.dvc)
- dvc commit
- dvc push
<br>Когда код текущего стейджа уже выполнялся, и в нем или выходных данных не было изменений dvc может отказаться запускать стейдж повторно. Это удобно, когда часть пердыдущих стейджей уже выполнялась в общем пайплайне.
Однако, иногда это необходимо сделать принудительно, выполнив все предыдущие стейджи заново:
<br>dvc repro <имя_стейджа> -f

# Работа с mlflow и makefile
Данные запусков экспериментов удобно сохранять в mlflow.
<br>Под экспериментами можно понимать один или несколько стейджей dvc.
<br>Каждый stage записывается в отдельный run в mlflow.
<br>Если отсутствует хранилище s3 и mlflow-сервер, это не приведет к ошибке в случае запуска стейджей dvc, эксперименты автоматически будут сохраняться в текущем репозитории в папке mlruns, ошибка возникнет только после выполнения стейджа при попытке сохранить файлы из outs в хранилище, это не повляет на выполнение кода.
<br>В связке dvc-mlflow есть недостаток: в эксперименте mlflow сохраняется номер коммита git для возможности воспроизведения эксперимента. Чтобы сохранялся номер правильного коммита, нужно всегда выполнять git push перед запуском стейджей dvc, но часто это не происходит.
<br>Поэтому для гарантированной записи номера коммита git для текущей версии кода необходимо запускать стейджи через makefile. В makefile запускаемые пайплайны будут называться целями. Например, запуск цели data_processing_train_val_test в терминале:
<br>make data_processing_train_val_test
<br>Для удобства названия целей совпадают с названиями стейджей в dvc.yaml.
<br>При запуске цели:
- Сначала выполняется запись названия текущего эксперимента (совпадает с названем текущих цели и стейджа) в файл config/new_experiment_stage.txt.
- Затем выполняется запуск файла src/utils/make_exp.py, в котором сначала создается новый эксперимент в mlflow, название нового эксперимента записывается в конфигурационном файле для экспериментов и ранов mlflow: projects/<имя_проекта>/config/config_mlflow.yaml. Эти конфигурационные файлы необходимы, чтобы понимать, в каких экспериментах и ранах хранятся данные и модели, созданные в результате выполнения текущего стейджа. Файл config_mlflow.yaml содержит информацию обо всех стейджах, которые выполнялись в проекте и на предыдущих, и на последующих стейджах. 
-Затем выполняется push в git и запуск соотвествущего стейджа dvc.
<br>Таким образом, при каждом запуске цели из makefile создается новый эксперимент в mlflow и репозиторий пушится в git.
<br>Что делать, если не нужно создавать новый эксперимент, а нужно дозаписать стейджи в текущий эксперимент?
<br>В этом случае стейджи необходимо запускать непосредственно с помощью dvc, например: dvc repro train. Это добавляет гибкости, но есть риск, что нужная версия кода не будет записана в mlflow. Это нужно контролировать самостоятельно, то есть перед запуском стейджа проверить актуальность репозитория в git.
<br>Пример принудительного запуска уже выполнявшихся ранее стейджей в новом эксперименте:
<br>make data_processing_train_val_test -e F='-f'

<br>В ранах mlflow можно сохранять модели, файлы словарей и прочих вспомогательных файлов, метрики и их графики, изображения, параметры моделей и т.п.
<br>Если не нужно использовать mlflow, можно удалить в коде все строчки, которые начинаются с mlflow.

# Подготовка данных
## Конфигурационные файлы
1. После установки всех библиотек создается папка .dvc. В ней необходимо создать файл "config" следующего содержания:
```
[core]
    remote = bert4rec
    autostage = true
['remote "bert4rec"']
    url = s3://DVC/bert4rec
    endpointurl = https://s3.local
    ssl_verify = false
```

<br>Теперь файлы dvc автоматически будут сохраняться по адресу: s3://DVC/bert4rec.

2. Пропишите название текущего проекта в файле config/project_run_now.txt.
3. В папке projects создайте папку с таким же названием проекта, как и в project_run_now.txt.
4. Положите исходный файл датафрейма по следующему пути: projects/<имя_проекта>/data/external/<имя_проекта>_<task_name>.pkl. <task_name> - это задача, которая будет выполняться: train_val_test, train_val, test, inference.
- train_val_test - весь исходный датафрейм автоматически разделяется на обучающие данные, валидационные и тестовые, где для валидации берутся предпоследние товары в цепочке, для тестирования - последние, обучение происходит на всей цепочке входных данных за исключением последних двух товаров в каждой цепочке.
- train_val - весь исходный датафрейм автоматически разделяется на обучающие и валидационные данные, где для валидации берутся последние товары, удобно, если необходимо обучить финальную версию модели на всех имеющихся данных.
- test - только тестирование на всех данных, для теста используются последние элементы во входной последовательности.
- inference - инференс для всей входной цепочки элементов, в результате инференса можно получить как эмбединги следующих товаров, так и предсказания модели, какой следующий товар будет куплен в порядке убывания уверенности модели.
<br>Репозиторий можно дописать для предсказания нескольких следующих элементов, изменив обработку предсказаний модели.
5. В папке config находятся 2 шаблона конфигурационного файла dvc: dvc_simple.yaml и dvc_template.yaml.
<br>В dvc_simple.yaml прописаны все необходимые стейджи в базовом варианте, то есть без файлов зависимостей между стейджами, так как для каждого проекта названия фалов могут быть разными. То есть при таком варианте каждый стейдж будет выполняться независимо от другого. Конфигурационный файл в таком виде будет одинаковым для всех проектов, его можно скопировать в корень репозитория, заменив имя на dvc.yaml.
<br>Шаблон полного конфигурационного файла - dvc_template.yaml. В нем нужно заменить {project}, {model_name} и {task} на названия проекта, модели и задачи, а затем переименовать и заменить файл dvc.yaml в корне репозитория. Для автоматического создания такого файла для текущего проекта необходимо запустить скрипт src/utils/dvc_create.py.
<br>Пример запуска:
python src/utils/dvc_create.py
<br>Скрипт содержит следующие аргументы командной строки:

-  `--model_name` - Имя модели (существующей для теста и инференса и будующей для задач 'train_val_test', 'train_val'). Если имя не задано, то будет присвоено автоматически в процессе обучения. 
-  `--is_embeddings` - True, если в результате инференса необходимо сохранить эмбеддинги. Если значение False, то в конфигурационный файл dvc.yaml не будет прописан датафрейм с эмбеддингами, соотвественно, в dvc автоматически он не запушится.
-  `--is_items_preds` - True, если в результате инференса необходимо сохранить номера предсказанных товаров. Если значение False, то в конфигурационный файл dvc.yaml не будет прописан датафрейм с товарами, соотвественно, в dvc автоматически он не запушится.

6. Шаблон конфигурационного файла модели bert4rec config/bert_config_<имя_проекта>.json содержит информацию с параметрами модели, а также данные необходимые для создания датасета. Необходимо заполнить этот шаблон перед запуском подготовки данных. Потом в процессе выполнения кода этот конфугурационный файл будет скопирован в папку с проектом.
Пример его содержимого:
```
{
   "attention_probs_dropout_prob": 0.2,
   "hidden_act": "gelu",
   "hidden_dropout_prob": 0.2,
   "hidden_size": 64,
   "initializer_range": 0.02,
   "intermediate_size": 256,
   "max_position_embeddings": 20,
   "num_attention_heads": 2,
   "num_hidden_layers": 2,
   "type_vocab_size": 2,
   "vocab_size": 668438
}
```
<br>Расшифровка параметров конфигурационного файла модели bert4rec:
-  `attention_probs_dropout_prob`: вероятность dropout для весов внимания (attention). Значение 0.2 означает, что во время обучения случайно выбирается 20% весов внимания для отключения. 
-  `hidden_act`: функция активации, используемая в скрытом слое (hidden layer).
-  `hidden_dropout_prob`: вероятность dropout для весов скрытого слоя (hidden layer). Значение 0.2 означает, что во время обучения случайно выбирается 20% весов скрытого слоя для отключения. 
-  `hidden_size`: размерность скрытого слоя (hidden layer) модели. 
-  `initializer_range`: диапазон инициализации весов модели. Значение 0.02 означает, что веса инициализируются случайными значениями в диапазоне [-0.02, 0.02]. 
-  `intermediate_size`: размерность промежуточного слоя (intermediate layer) трансформера. 
-  `max_position_embeddings : максимальное количество токенов во входной последовательности модели. 
-  `num_attention_heads`: количество голов внимания (attention heads) в трансформере. 
-  `num_hidden_layers`: количество скрытых слоев (hidden layers) в трансформере. 
-  `type_vocab_size`: размер словаря типов (type vocabulary) в модели. Если в данных только 2 столбца с номером пользователя и идентификатором товара, то значение будет равно 2.
-  `vocab_size`: размер словаря товаров. Изначально это значение можно оставить пустым, оно заполняется количеством товаров для текущего датасета в процессе формирования обучающего набора данных.

7. В файл .gitignore рекомендуется добавить следующие файлы и папки:
<br>.env
<br>projects/*
<br>mlruns/*
<br>\_\_pycache__/*

## Подготовка данных для модели
### src/data_processing/data_processing_raw.py
Скрипт data_processing_raw.py принимает на вход датафрейм data/external/<имя_проекта>_<task_name>.pkl следующего вида:

| | uid | sid_add_to_cart |
| ------- | ------- | --------------------------------------------------- |
| 0 | 0                   | [628125, 9100792, 5051362, 2506297]                 |
| 1 | 1                   | [3979185, 3979171, 5433489]                 |
...

<br>uid - идентификационный номер пользователя.
<br>sid_add_to_cart - списки идентификационных номеров товаров, добавленных пользователем в корзину в порядке очередности.
<br>На выходе мы получаем текстовый файл projects/<имя_проекта>/data/interim/<имя_проекта>_<task_name>.txt, который может быть использован в скриптах data_processing_train_val_test.py, data_processing_train_val.py, data_processing_test.py, data_processing_inference.py для приведения данных в формат *.tfrecord. Вид содержимого выходного файла:
0 628125
0 9100792
0 5051362
0 2506297
1 3979185
1 3979171
1 5433489
...
<br>
<br>Пример запуска стейджа на выполнение:
<br>dvc repro data_processing_raw
<br>Скрипт можно запустить также в терминале напрямую:
<br>python src/data_processing/data_processing_raw.py
<br>Аргументы командной строки:

-  `--task` - задача, для которой необходимо подготовить данные: 'train_val_test', 'train_val', 'test', 'inference'. 
-  `--drop_duplicates` - аргумент для удаления дубликатов товаров, которые идут последовательно непрерывно (удалять, если True).
-  `--vocab_exist` - 1, если нужно использовать уже существующий словарь кодировки номеров пользователей. Словарь должен находиться по адресу: projects/<имя_проекта>/referenses/dics/dic_encoding_uid_user_id.pkl.

<br>В процессе выполнения скрипта номера пользователей uid кодируются по-порядку целыми числами. Словарь кодировки сохраняется по адресу projects/<имя_проекта>/referenses/dics/dic_encoding_uid_user_id.pkl.
<br>Во время тестирования и инференса идентификационные номера пользователей не приводятся к оригинальным. Для их раскодировки нужно воспользоваться словарем dic_encoding_uid_user_id.pkl.

### src/data_processing/data_processing_all.py
С помощью скрипта data_processing_all.py можно подготовить сразу 2 набора данных: 
- для обучения и валидации
- для тестирования
<br>Пример запуска стейджа на выполнение:
<br>dvc repro data_processing_all
<br>Аргументы командной строки:

-  `--max_predictions_per_seq`: Длина выходной последовательности, которая будет предсказана моделью.
-  `--masked_lm_prob`: Доля элементов, которые будут замаскированы во входной последовательности. По умолчанию установлено значение 0.2, что означает, что около 20% элементов будет замаскировано.
-  `--prop_sliding_window`: Определяет размер шага, с которым окно будет скользить по последовательности элементов, представляет собой долю от максимальной длины, которую должно занимать окно. Если prop_sliding_window равно -1.0 , то окно будет иметь размер max_seq_length , то есть не будет скользить.
-  `--mask_prob`: Определяет вероятность замаскирования каждого элемента во входной последовательности. Например, если  mask_prob  равно 0.2, то около 20% элементов будет замаскировано. В процессе создания обучающих примеров, модель bert4rec использует механизм маскирования, где некоторые элементы во входной последовательности замаскированы, чтобы модель могла предсказать их и восстановить исходную последовательность.
-  `--dupe_factor`: Используется для определения количества раз, которое каждая последовательность элементов будет использоваться для создания обучающих примеров с разными масками.
-  `--pool_size`: Используется для определения количества процессов, которые будут использоваться для параллельного выполнения операций.
-  `--short_seq_prob`: В процессе создания обучающих примеров, модель bert4rec использует двухпредложное представление, где каждая входная последовательность элементов разделяется на две части.  short_seq_prob  определяет вероятность того, что входная последовательность будет сокращена до короткого образца, состоящего только из одной части исходной последовательности. Например, если  short_seq_prob  равно 0.2, то около 20% входных последовательностей будет сокращено до короткого образца.
-  `--task`: Название задачи, для которой необходимо подготовить данные ('train_val_test', 'train_val', 'test', 'inference').
-  `--vocab_filename`: Путь к файлу словаря товаров, если он уже существует, или `None`, если требуется создать новый словарь. 

<br>Данных для обучения и валидации будет (dupe_factor + 1) * количество цепочек товаров в первоначальном датасете.
<br>Данных для тестирования будет столько же, сколько цепочек товаров в первоначальном датасете.
<br>В качестве входных данных используется датасет projects/<имя_проекта>/data/interim/<имя_проекта>_<task_name>.txt, сформированный на предыдущем этапе или во внешнем приложении.
<br>На выходе получаем следующие файлы:
- projects/<имя_проекта>/data/processed/<имя_проекта>.train_val.tfrecord: файл данных с масками и аугментацией для обучения и валидации модели (для <task_name>: 'train_val_test', 'train_val').
- projects/<имя_проекта>/data/processed/<имя_проекта>.test.tfrecord: файл данных без масок и аугментации для тестирования модели на последнем элементе каждой последовательности товаров (для <task_name>: 'train_val_test', 'test').
- projects/<имя_проекта>/data/processed/<имя_проекта>.inference.tfrecord: файл данных без масок и аугментации для инференса модели, последний элемент каждой последовательности дублируется, чтобы на вход модели подавалась полная исходная цепочка без исключения последнего элемента как при тестировании, остальная подготовка данных при этом останется как для тестирования (для <task_name>: 'inference').
- projects/<имя_проекта>/references/<имя_проекта>.vocab: словарь товаров, каждой товар в первоначальном датасете кодируется новым числовым значением.

<br>ВАЖНО ПОМНИТЬ, что при подготовке датасета для обучения модели на новых данных словарь создается заново, то есть параметр vocab_filename должен быть `None`, а для тестирования и инференса обязательно указывать путь к словарю, созданному при обучении модели: projects/<имя_проекта>/references/<имя_проекта>.vocab.

## Обучение и валидация модели - train_val.py
Пример запуска стейджа на выполнение:
<br>dvc repro train_val
<br>Аргументы командной строки:

-  `--model_name`: Имя модели. 
-  `--save_checkpoints_steps`: Частота сохранения контрольных точек во время обучения модели.
-  `--learning_rate`: Скорость обучения модели.
-  `--num_train_steps`: Количество шагов обучения модели.
-  `--num_warmup_steps`: Количество шагов прогрева, в течение которых скорость обучения постепенно увеличивается.
-  `--use_tpu`: Флаг, указывающий, будет ли использоваться TPU для обучения модели.
-  `--batch_size_train`: Размер батча для обучения.
-  `--batch_size_val`: Размер батча для валидации.
-  `--is_train`: Флаг, указывающий, будет ли модель обучаться. Если модель уже существует в указанной директории, то обучение не будет выполняться, только валидация.
-  `--is_evaluate`: Флаг, указывающий, будет ли выполняться валидация модели. Это может занять длительное время.
-  `--fast_eval`: Флаг, который решает проблему длительной валидации. При значении True валидация происходит только для первой строки в батче.
-  `--save_predictions_file`: Путь к файлу, в котором будут сохранены предсказания модели на валидационном наборе данных. Файл может оказаться очень большого размера. 
                                Если значение True, предсказания модели сохраняются в projects/<имя_проекта>/results/<model_name>/eval_results.txt.
-  `--training_time_limit_seconds`: Ограничение времени обучения модели в секундах.

Название словаря должно совпадать с именем проекта.
<br>В результате обучения в папку projects/<имя_проекта>/models/<model_name> сохраняются файлы модели. Если --is_evaluate True, результаты валидации сохраняются в файл projects/<имя_проекта>/results/<model_name>/eval_results.txt.
<br>Если модель с указанным именем проекта в папке уже существует, обученная на заданное количество шагов, то сразу запускается валидация. Если существующая модель обучена на меньшее количество шагов обучения, то дообучается до заданного количества.
<br>Если model_name None, то имя модели присваивается автоматически. Пример имени модели:
<br>hs64_msl20_mpps1_nts100000_lr0.0001
<br>где:
<br>hs64: hidden_size = 64 (размерность слоя эмбеддингов),
<br>msl20: max_seq_length = 20 (максимальная длина входной последовательности),
<br>mpps1: max_predictions_per_seq = 1 (количество предсказаний),
<br>nts100000: num_train_steps = 100000 (количество шагов обучения модели),
<br>lr0.0001: learning_rate = 0.0001 (шаг градиентного спуска)
<br>Если выбрана валидация, выводятся на экран и сохраняются в mlflow метрики: ndcg@1, ndcg@5, ndcg@10, ndcg@100, hit@1, hit@5, hit@10, hit@100, rank100, rank100_percent.
<br>Метрика rank100 показывает, для какого количества строк цепочек товаров следующий предсказанный товар попал в 100 первых предсказаний модели, упорядоченных в порядке ее уверенности.
<br>Метрика rank100_percent вычисляет процент rank100 от общего количества входных цепочек товаров.

<br>Конфигурационные файлы модели и обучения копируются в папку: projects/<имя_проекта>/models/<model_name>/config.

## Тестирование модели - test.py
Пример запуска стейджа на выполнение:
<br>dvc repro test
<br>Аргументы командной строки:

-  `--model_name`: Имя модели. 
-  `--use_tpu`: Флаг, указывающий, будет ли использоваться TPU для обучения модели.
-  `--batch_size`: Размер батча.
-  `--save_predictions_file`: Сохранять ли предсказания модели на валидационном наборе данных. Файл может оказаться очень большого размера. 
                                Если значение True, предсказания модели сохраняются в projects/<имя_проекта>/results/<model_name>/test_results.txt.
-  `--fast_eval`: Флаг, который решает проблему длительности тестирования. При значении True тестирование происходит только для первой строки в батче.

<br>Если имя модели None, то конфигурационные файлы модели и обучения загружаются из projects/<имя_проекта>/config. Если имя модели указано, то конфигурационные файлы загружаются из папки модели.
<br>Выводятся на экран и сохраняются в mlflow метрики: ndcg@1, ndcg@5, ndcg@10, ndcg@100, hit@1, hit@5, hit@10, hit@100, rank100, rank100_percent.

## Инференс модели - inference.py
Пример запуска стейджа на выполнение:
<br>dvc repro inference
<br>Аргументы командной строки:

-  `--model_name`: Имя модели. 
-  `--use_tpu`: Флаг, указывающий, будет ли использоваться TPU для обучения модели.
-  `--batch_size`: Размер батча.
-  `--is_embeddings`: Создать ли файл с эмбеддингами предпоследнего слоя модели. Эмбеддинги сохраняются по адресу: projects/<имя_проекта>/results/<model_name>/df_next_sids_embs.pkl.
-  `--is_items_preds`: Создать ли файл с предсказаниями товаров. Сохраняется по адресу: projects/<имя_проекта>/results/<model_name>/df_next_sids.pkl.
-  `--predictions_per_user`: Сколько товаров необходимо записать в предсказания в порядке убывания уверенности модели. В данном случае мы рассматриваем предсказания только для следующего товара.

<br>Если имя модели None, то конфигурационные файлы модели и обучения загружаются из projects/<имя_проекта>/config. Если имя модели указано, то конфигурационные файлы загружаются из папки модели.

<br>Во время тестирования и инференса идентификационные номера пользователей не приводятся к оригинальным. Для их раскодировки нужно воспользоваться словарем projects/<имя_проекта>/referenses/dics/dic_encoding_uid_user_id.pkl.

# Запуск репозитория
Выполните подготовку данных по предыдущему пункту.
В результате подготовки данных для запуска проекта:
- должен быть заполнен файл config в папке .dvc для работы с dvc и mlflow (не обязательно, если dvc и mlflow использоваться не будут)
- должен быть заполнен файл config/project_run_now.py
- должна быть создана папка projects/<имя_проекта>/data/external c файлами <имя_проекта>_<task_name>.pkl для каждой задачи.
- должен быть заполнен конфигурационный файл config/bert_config.json
- должны быть заполнены аргументы командной строки в файле src/utils/dvc_create.py. Особенно обратите внимание на название модели. Если необходимо использовать dvc-пайплайн, то название модели должно быть заполнено обязательно. При необходимости обучить новую модель, нeобходимо заново генерировать dvc.yaml, так в зависимости от имени модели и проекта генерируются пути сохранения данных в dvc.yaml.
- должен быть создан конфигурационный файл dvc.yaml, например, с помощью скрипта src/utils/dvc_create.py.
- должны быть заполнены аргументы командной строки в файлах:
    - src/data_processing/data_processing_raw.py
    - src/data_processing/data_processing_all.py
    - src/train_test_inf/train_val.py
    - src/train_test_inf/test.py
    - src/train_test_inf/inference.py
<br>Первый запуск нового проекта обязательно выполните через makefile в командной строке. Пример:
<br>make inference
<br>При этом в папке с проектом будут созданы необходимые конфигурационные файлы.
<br>Повторные запуски стейджей проекта можно выполнять с помощью dvc repro <stage>.

# Обучение, валидация, тестирование, инференс
Код для обучения, валидации, теста и инференса моделей расположен в папке src/train_test_inf.
- train_val.py - обучение и валидация
- test.py - тестирование модели
- inference.py - инференс модели

# Tensorboad
Когда запускается обучение, в папке модели сохраняются логи tensorboard. Чтобы открыть tensorboard и посмотреть графики изменения loss:
1. Откройте терминал и перейдите в папку с моделью: cd models/<имя проекта>
2. Запустите tensorboard: tensorboard --logdir=./
3. В нижней строке появится адрес, который нужно открыть в браузере, обычно такой: http://localhost:6006/


# Ссылки
Статья: BERT4Rec https://arxiv.org/pdf/1904.06690.pdf
<br>Исходный репозиторий: https://github.com/FeiSun/BERT4Rec
<br>Основные репозитории:
<br>https://github.com/tunghia1890/BERT4Rec_TF2x
<br>https://github.com/asash/bert4rec_repro
<br>Вспомогательный репозиторий: https://github.com/asash/BERT4rec_py3_tf2