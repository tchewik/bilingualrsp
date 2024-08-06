# $ sh train_bimpm.sh {GUM|RST-DT|RuRSTB} {en|ru}

export CORPUS=${1}
export LANG=${2}
export MODEL_PATH=saves/bimpm/${CORPUS}_${LANG}
export DATA_PATH=data/bimpm_clf_${CORPUS}_${LANG}
export TRAIN_FILE_PATH=${DATA_PATH}/train.tsv
export DEV_FILE_PATH=${DATA_PATH}/dev.tsv
export TEST_FILE_PATH=${DATA_PATH}/test.tsv

echo $CORPUS $LANG

mkdir saves/bimpm
rm -r $MODEL_PATH
allennlp train -s ${MODEL_PATH} configs/bimpm.jsonnet \
    --include-package relation_classifier
allennlp predict --use-dataset-reader --silent \
    --output-file ${MODEL_PATH}/predictions_dev.json ${MODEL_PATH}/model.tar.gz ${DEV_FILE_PATH} \
    --include-package relation_classifier \
    --predictor textual-entailment
allennlp predict --use-dataset-reader --silent \
    --output-file ${MODEL_PATH}/predictions_test.json ${MODEL_PATH}/model.tar.gz ${TEST_FILE_PATH} \
    --include-package relation_classifier \
    --predictor textual-entailment
