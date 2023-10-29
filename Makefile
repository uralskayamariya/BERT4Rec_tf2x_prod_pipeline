commit_and_repro:
	echo "$(MAKECMDGOALS)" > config/new_experiment_stage.txt \
	&& python src/utils/make_exp.py \
	&& git add . \
	&& git commit -m "run stage $(MAKECMDGOALS)" \
	&& git push \
	&& dvc repro $(MAKECMDGOALS) $(F)
	
data_processing_raw_train_val_test: commit_and_repro
data_processing_raw_train_val: commit_and_repro
data_processing_raw_test: commit_and_repro
data_processing_raw_inference: commit_and_repro

data_processing_all_train_val_test: commit_and_repro
data_processing_all_train_val: commit_and_repro
data_processing_all_test: commit_and_repro
data_processing_all_inference: commit_and_repro

train_val: commit_and_repro
test: commit_and_repro
inference: commit_and_repro


# Команда для запуска цели с аргументом '-f':
# make inference -e F='-f'    