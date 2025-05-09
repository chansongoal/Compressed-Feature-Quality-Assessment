#!/bin/bash 
cd /ghome/gaocs/FCM-UFC/machines/dinov2
# python dpt.py >/gdata1/gaocs/Data_FQA/accuracy_log/org/eval_dpt_100.txt 2>&1
# python dpt.py >/gdata1/gaocs/Data_FQA/accuracy_log/inverse_transformed/eval_dpt_kmeans10_bitdepth8_80.txt 2>&1

OUTDIR="/gdata1/gaocs/Data_FQA/accuracy_log/${1}/trained_${2}/${3}${4}_bitdepth${5}/dinov2_dpt"
mkdir -p "$OUTDIR"

python dpt.py \
    --arch "$1" \
    --train_task "$2" \
    --transform_type "$3" \
    --samples "$4" \
    --bit_depth "$5" \
    --learning_rate "$6" \
    --epochs "$7" \
    --batch_size "${8}" \
    --patch_size "${9}" \
    --lambda_value_all "${@:10}" \
    > "$OUTDIR/${1}_trained_${2}_eval_dpt.txt"


