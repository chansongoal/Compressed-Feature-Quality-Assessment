#!/bin/bash 
export HOME=/ghome/gaocs

cd /ghome/gaocs/FCM-UFC/coding/hm
python hm.py

# cd /ghome/gaocs/FCM-UFC/coding/CompressAI
# # pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
# # pip install -e . 

# python run_batch.py \
#     --pipeline_config "$1" \
#     --arch "$2" \
#     --train_model_type "$3" \
#     --train_task "$4" \
#     --transform_type "$5" \
#     --samples "$6" \
#     --bit_depth "$7" \
#     --lambda_value "$8" \
#     --epochs "$9" \
#     --save_period "${10}" \
#     --learning_rate "${11}" \
#     --batch_size "${12}" \
#     --patch_size "${13}" \
#     --pretrained_model "${14}" \
