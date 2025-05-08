import os
import numpy as np
from typing import Union, List
import nonlinear_transform

def perform_nonlinear_transform(org_feat_path, rec_feat_path, transform_mapping_path, model_type, task, samples, trun_flag, trun_high, trun_low, transform_type, bit_depth, data_size, source_file=None, crop_flag=True):
    transform_mapping_name = f'{transform_mapping_path}/transform_mapping_{task}_{transform_type}{samples}_bitdepth{bit_depth}.json'
    if task=='dpt':
        transform_mapping_name = [f'{transform_mapping_path}/transform_mapping_{task}_layer10_{transform_type}{samples}_bitdepth{bit_depth}.json', \
                                  f'{transform_mapping_path}/transform_mapping_{task}_layer20_{transform_type}{samples}_bitdepth{bit_depth}.json', \
                                  f'{transform_mapping_path}/transform_mapping_{task}_layer30_{transform_type}{samples}_bitdepth{bit_depth}.json', \
                                  f'{transform_mapping_path}/transform_mapping_{task}_layer40_{transform_type}{samples}_bitdepth{bit_depth}.json'
        ]
    quantization_points = nonlinear_transform.load_quantization_points(transform_mapping_name)

    if source_file == None: feat_names = sorted(os.listdir(org_feat_path))
    else: 
        with open(source_file, 'r') as f:
            feat_names = [line.strip().split(' ', 1)[0].split('.')[0]+'.npy' for line in f if line.strip()]
            print('number samples: ', len(feat_names))
    # feat_names = [f for f in os.listdir(org_feat_path) if f.startswith("arc_")]
    print(len(feat_names))
    # Load and quantize features
    for feat_name in feat_names:
        org_feat_name = os.path.join(org_feat_path, feat_name)
        org_feat = np.load(org_feat_name)
        N, C, H, W = org_feat.shape
        dtype = org_feat.dtype
        # print(f"{feat_name}: {N}, {C}, {H}, {W}")
        rec_feat_name = os.path.join(rec_feat_path, feat_name)

        if trun_flag:
            trun_feat = nonlinear_transform.truncation(org_feat, trun_low, trun_high)
        else:
            trun_feat = org_feat

        quantized_feat = nonlinear_transform.nonlinear_quantization(trun_feat, quantization_points, bit_depth)

        # # generate transformed features for test data
        # quantized_feat = quantized_feat.astype(dtype)
        # np.save(rec_feat_name, quantized_feat)
    
        # # generate inverse transformed features for test data
        # dequantized_feat = nonlinear_transform.nonlinear_dequantization(quantized_feat, quantization_points, bit_depth)
        # dequantized_feat = dequantized_feat.astype(dtype)
        # np.save(rec_feat_name, dequantized_feat)

        # generate transformed features for training data
        # pack_feat = nonlinear_transform.packing(quantized_feat, model_type)
        # if crop_flag == True:
        #     cropped_feat = nonlinear_transform.random_crop(pack_feat, (256, 256))   # perform crop after packing, for other tasks
        #     cropped_feat = cropped_feat.astype(dtype)
        #     np.save(rec_feat_name, cropped_feat)
            # # for csr
            # for n in range(2):
            #     rec_feat_name_iter = f"{rec_feat_name[:-4]}_iter{n}.npy"
            #     cropped_feat = nonlinear_transform.random_crop(pack_feat, (64, 1024))
            #     # reshape (64, 1024) to (256, 256)
            #     blocks = np.split(cropped_feat, 4, axis=1)
            #     reshaped_feat = np.concatenate(blocks, axis=0)
            #     reshaped_feat = reshaped_feat.astype(dtype)
            #     np.save(rec_feat_name_iter, reshaped_feat)
        if task=='dpt':
            for n in range(4):
                layer = int((n+1)*10)
                feat = nonlinear_transform.random_crop(quantized_feat[0,n,:,:], (256,256))
                rec_feat_name = os.path.join(rec_feat_path, feat_name[:-4]+f'layer{layer}.npy')
                feat = feat.astype(dtype)
                np.save(rec_feat_name, feat)

def perform_nonlinear_transform_dpt(org_feat_path, rec_feat_path, transform_mapping_path, model_type, task, samples, trun_flag, trun_high, trun_low, transform_type, bit_depth, data_size, source_file=None, crop_flag=True):
    layer = 10
    transform_mapping_name = f'{transform_mapping_path}/transform_mapping_{task}_{transform_type}{samples}_bitdepth{bit_depth}.json'
    if task=='dpt':
        transform_mapping_name = f'{transform_mapping_path}/transform_mapping_{task}_layer{layer}_{transform_type}{samples}_bitdepth{bit_depth}.json'
    quantization_points = nonlinear_transform.load_quantization_points(transform_mapping_name)

    if source_file == None: feat_names = sorted(os.listdir(org_feat_path))
    else: 
        with open(source_file, 'r') as f:
            feat_names = [line.strip().split(' ', 1)[0].split('.')[0]+'.npy' for line in f if line.strip()]
            print('number samples: ', len(feat_names))
    # feat_names = [f for f in os.listdir(org_feat_path) if f.startswith("arc_")]
    
    # feat_names = [f for f in feat_names if f"layer{layer}" in f]
    print(layer, len(feat_names))

    # Load and quantize features
    for feat_name in feat_names:
        org_feat_name = os.path.join(org_feat_path, feat_name)
        org_feat = np.load(org_feat_name)
        # org_feat = np.expand_dims(org_feat, axis=0); org_feat = np.expand_dims(org_feat, axis=0)
        N, C, H, W = org_feat.shape
        dtype = org_feat.dtype
        # print(f"{feat_name}: {N}, {C}, {H}, {W}")
        rec_feat_name = os.path.join(rec_feat_path, feat_name)

        if trun_flag:
            trun_feat = nonlinear_transform.truncation(org_feat, trun_low, trun_high)
        else:
            trun_feat = org_feat

        quantized_feat = nonlinear_transform.nonlinear_quantization(trun_feat, quantization_points, bit_depth)

        # generate transformed features for test data
        quantized_feat = quantized_feat[0,0,:,:]
        quantized_feat = quantized_feat.astype(dtype)
        np.save(rec_feat_name, quantized_feat)

def perform_uniform_normalization(org_feat_path, rec_feat_path, model_type, trun_flag, trun_high, trun_low, bit_depth, data_size, crop_flag):
    feat_names = os.listdir(org_feat_path)[:data_size]
    # Load and quantize features
    for feat_name in feat_names:
        org_feat_name = os.path.join(org_feat_path, feat_name)
        org_feat = np.load(org_feat_name)
        N, C, H, W = org_feat.shape
        dtype = org_feat.dtype; print(dtype)
        # print(f"{feat_name}: {N}, {C}, {H}, {W}")
        rec_feat_name = os.path.join(rec_feat_path, feat_name)

        if trun_flag:
            trun_feat = nonlinear_transform.truncation(org_feat, trun_low, trun_high)
        else:
            trun_feat = org_feat

        quantized_feat = nonlinear_transform.uniform_quantization(trun_feat, trun_low, trun_high, bit_depth)

        pack_feat = nonlinear_transform.packing(quantized_feat, model_type)
        if crop_flag == True:
            cropped_feat = nonlinear_transform.random_crop(pack_feat, (256, 256))   # perform crop after packing
        else: cropped_feat = pack_feat

        cropped_feat = cropped_feat.astype(dtype)
        np.save(rec_feat_name, cropped_feat)

        # dequantized_feat = nonlinear_transform.uniform_dequantization(quantized_feat, trun_low, trun_high, bit_depth)
        # np.save(rec_feat_name, dequantized_feat)


if __name__ == "__main__":
    # model_type = 'dinov2'; task = 'cls'; max_v = 94.15; min_v = -542.31; trun_high = 94.15; trun_low = -542.31; source_name = 'imagenet_selected_label100.txt'
    # model_type = 'dinov2'; task = 'seg'; max_v = 105.95; min_v = -506.97; trun_high = 105.95; trun_low = -506.97; source_name = 'seg_val_100.txt'
    model_type = 'dinov2'; task = 'dpt'; max_v = [3.27, 5.03, 25.05, 100.27]; min_v = [-2.39, -26.44, -323.30, -504.44]; trun_high = [3.27, 5.03, 25.05, 100.27]; trun_low = [-2.39, -26.44, -323.30, -504.44]; source_name = 'nyu_test_name100.txt'
    # model_type = 'llama3'; task = 'csr'; max_v = 47.75; min_v = -71.50; trun_high = 47.75; trun_low = -71.50; source_name = 'arc_challenge_test_longest500_shape.txt'
    # model_type = 'sd3'; task = 'tti'; max_v = 4.46; min_v = -5.79; trun_high = 4.46; trun_low = -5.79
    

    train_data_root = f'/gdata1/gaocs/FCM_LM_Train_Data'
    test_data_root = f'/gdata1/gaocs/FCM_LM_Test_Dataset'
    data_root = f'/gdata1/gaocs/Data_FQA'
    transform_mapping_path = f'{data_root}/transform_mapping/{model_type}_{task}'

    # config = 'train'; source_file = None
    config = 'test'; source_file = f'{test_data_root}/{model_type}/{task}/source/{source_name}'
    
    # quant_type = 'uniform'; samples = 0
    quant_type = 'kmeans'; samples = 10; bit_depths = [8]
    trun_flag = False
    if trun_flag == False: trun_high = max_v; trun_low = min_v

    data_size = 2
    for bit_depth in bit_depths:
        print(model_type, task, trun_flag, quant_type, samples, max_v, min_v, trun_high, trun_low, bit_depth)

        # perform kmeans quantization to generate training data
        # org_feat_path = f'{train_data_root}/{model_type}/{task}/org_feat/{config}'
        org_feat_path = '/gdata1/gaocs/FCM_LM_Test_Dataset/dinov2/dpt/feature'
        rec_feat_path = f'{train_data_root}/{model_type}/{task}/{quant_type}{samples}_bitdepth{bit_depth}/crop_hgt256_wdt256/{config}'; os.makedirs(rec_feat_path, exist_ok=True)
        perform_nonlinear_transform(org_feat_path, rec_feat_path, transform_mapping_path, model_type, task, samples, trun_flag, trun_high, trun_low, quant_type, bit_depth, data_size, source_file, True)
        # perform_nonlinear_transform_dpt(org_feat_path, rec_feat_path, transform_mapping_path, model_type, task, samples, trun_flag, trun_high, trun_low, quant_type, bit_depth, data_size, source_file=None, crop_flag=True)

        # # perform linear quantization to generate training data
        # org_feat_path = f'{train_data_root}/{model_type}/{task}/org_feat/{config}'
        # rec_feat_path = f'{train_data_root}/{model_type}/{task}/{quant_type}{samples}_bitdepth{bit_depth}/crop_hgt256_wdt256/test'
        # if not os.path.exists(rec_feat_path): os.makedirs(rec_feat_path)
        # perform_uniform_normalization(org_feat_path, rec_feat_path, model_type, trun_flag, trun_high, trun_low, bit_depth, data_size, True)

        # # perform kmeans quantization to generate test data 
        # org_feat_path = f'{test_data_root}/{model_type}/{task}/feature' 
        # # rec_feat_path = f'{data_root}/transformed/{model_type}_{task}/{quant_type}{samples}_bitdepth{bit_depth}'; os.makedirs(rec_feat_path, exist_ok=True)
        # rec_feat_path = f'{data_root}/inverse_transformed/{model_type}_{task}/{quant_type}{samples}_bitdepth{bit_depth}'; os.makedirs(rec_feat_path, exist_ok=True)
        # perform_nonlinear_transform(org_feat_path, rec_feat_path, transform_mapping_path, model_type, task, samples, trun_flag, trun_high, trun_low, quant_type, bit_depth, data_size, source_file, False)