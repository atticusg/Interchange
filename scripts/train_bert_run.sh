MODEL=$1
DIFFICULTY=$2
python train_bert.py run \
    -d "mqnli_models/bert.db" \
    -m "python dummy_metascript.py" \
    -n $1 \
    -s -1

