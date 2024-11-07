device=$1
path=$2
epoch=$3

CUDA_VISIBLE_DEVICES=${device} python -u eval_degrad.py --path ${path} --epoch ${epoch} --type 'caicl' \
    2>&1 | tee eval_degrad.log
