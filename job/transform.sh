#!/bin/bash 

export HOME=/ghome/gaocs
pip install scikit-learn

# cd /ghome/gaocs/FCM-UFC/coding/transform; python nonlinear_transform.py >/gdata1/gaocs/Data_DTUFC/transform_mapping/dinov2_cls/dinov2_cls_kmenas100.txt 2>&1
# cd /ghome/gaocs/FCM-UFC/coding/transform; python nonlinear_transform.py >/gdata1/gaocs/Data_DTUFC/transform_mapping/dinov2_seg/dinov2_seg_kmenas100.txt 2>&1
# cd /ghome/gaocs/FCM-UFC/coding/transform; python nonlinear_transform.py >/gdata1/gaocs/Data_DTUFC/transform_mapping/dinov2_dpt/dinov2_dpt_kmenas_layer20.txt 2>&1
# cd /ghome/gaocs/FCM-UFC/coding/transform; python nonlinear_transform.py >/gdata1/gaocs/Data_DTUFC/transform_mapping/llama3_csr/llama3_csr_kmenas100.txt 2>&1
# cd /ghome/gaocs/FCM-UFC/coding/transform; python nonlinear_transform.py >/gdata1/gaocs/Data_DTUFC/transform_mapping/sd3_tti/sd3_tti_kmenas100.txt 2>&1
cd /ghome/gaocs/FCM-UFC/coding/transform; python generate_data.py >/gdata1/gaocs/FCM_LM_Train_Data/dinov2/cls/generate_cls.txt 2>&1