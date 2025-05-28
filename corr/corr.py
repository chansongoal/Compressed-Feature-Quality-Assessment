import os
import numpy as np
from scipy.stats import pearsonr, spearmanr


#gcs, define a function to read source file
def get_feature_names(source_file):
    with open(source_file, "r", encoding="utf-8") as file:
        return [line.strip().split()[0].split('.')[0] for line in file if line.strip()]

def extract_image_data(text_lines, image_name):
    """
    Extract data for a specific image from the text content.

    Parameters:
    - text_lines: list[str], the content of the text file, where each line is a string
    - image_name: str, the name of the image to extract data for

    Returns:
    - numpy.ndarray, a 2D array of shape (N, 3), where N is the number of matched rows,
      and the columns represent (mIoU, MSE, SSIM)
    """
    data = []  # List to store extracted data

    for line in text_lines:
        parts = line.strip().split()
        if not parts: continue  # Skip empty lines
        name, _ = os.path.splitext(parts[0])
        # print(name, image_name)
        if name == image_name:
            try:
                values = list(map(float, parts[1:]))  # Extract accuracy, MSE, 
                data.append(values)
            except ValueError:
                continue  # Skip the line if conversion fails
    # print(data)
    return np.array(data)  # Convert to a NumPy array


def correlation(source_file, log_file):
    with open(log_file, "r", encoding="utf-8") as file:
        logs = file.readlines()
    
    feat_names = get_feature_names(source_file)

    plcc_all = []; srocc_all = []
    plcc_cosine_all = []; srocc_cosine_all = []
    plcc_cka_all = []; srocc_cka_all = []
    for idx, feat_name in enumerate(feat_names):
        data = extract_image_data(logs, feat_name)
        plcc, _ = pearsonr(data[:,0], data[:,1])    # mse
        srocc, _ = spearmanr(data[:,0], data[:,1])  # mse
        plcc_cosine, _ = pearsonr(data[:,0], data[:,2])    # cosine
        srocc_cosine, _ = spearmanr(data[:,0], data[:,2])  # cosine
        plcc_cka, _ = pearsonr(data[:,0], data[:,3])    # cka
        srocc_cka, _ = spearmanr(data[:,0], data[:,3])  # cka
        plcc_all.append(plcc); srocc_all.append(srocc)
        plcc_cosine_all.append(plcc_cosine); srocc_cosine_all.append(srocc_cosine)
        plcc_cka_all.append(plcc_cka); srocc_cka_all.append(srocc_cka)
        print(f"{feat_name} {plcc:.4f} {srocc:.4f} {plcc_cosine:.4f} {srocc_cosine:.4f} {plcc_cka:.4f} {srocc_cka:.4f}")

    plcc_all = np.asarray(plcc_all); srocc_all = np.asarray(srocc_all)
    plcc_cosine_all = np.asarray(plcc_cosine_all); srocc_cosine_all = np.asarray(srocc_cosine_all)
    plcc_cka_all = np.asarray(plcc_cka_all); srocc_cka_all = np.asarray(srocc_cka_all)
    # print(f'average: {np.mean(plcc_all):.4f} {np.mean(srocc_all):.4f} {np.mean(plcc_cosine_all):.4f} {np.mean(srocc_cosine_all):.4f} {np.mean(plcc_cka_all):.4f} {np.mean(srocc_cka_all):.4f}')    
    print(f'average: {np.nanmean(plcc_all):.4f} {np.nanmean(srocc_all):.4f} {np.nanmean(plcc_cosine_all):.4f} {np.nanmean(srocc_cosine_all):.4f} {np.nanmean(plcc_cka_all):.4f} {np.nanmean(srocc_cka_all):.4f}')    


    # pos_plcc_all = plcc_all[plcc_all > 0]
    # pos_plcc_count = len(pos_plcc_all)
    # pos_plcc_mean = np.mean(pos_plcc_all) if pos_plcc_count > 0 else 0
    # print(f'positive: {np.mean(pos_plcc_mean):.4f} {np.mean(pos_plcc_mean):.4f}')

    # pos_plcc_all = plcc_all[plcc_all < 0]
    # pos_plcc_count = len(pos_plcc_all)
    # pos_plcc_mean = np.mean(pos_plcc_all) if pos_plcc_count > 0 else 0
    # print(f'negative: {np.mean(pos_plcc_mean):.4f} {np.mean(pos_plcc_mean):.4f}')

if __name__ == "__main__":
    arch = 'hm'; train_task = 'hybrid'; quant_type = 'uniform'; samples = 0; bit_depth = 10
    # arch = 'vtm'; train_task = 'hybrid'; quant_type = 'uniform'; samples = 0; bit_depth = 10
    # arch = 'hyperprior'; train_task = 'dpt'; quant_type = 'kmeans'; samples = 10; bit_depth = 8
    
    test_model_type = 'dinov2'; test_task = 'dpt'; 

    dataset_root = '/gdata1/gaocs/FCM_LM_Test_Dataset'

    if test_task=='cls': source_file = f'{dataset_root}/{test_model_type}/{test_task}/source/imagenet_selected_label100.txt'
    elif test_task=='seg': source_file = f'{dataset_root}/{test_model_type}/{test_task}/source/seg_val_100.txt'
    elif test_task=='dpt': source_file = f'{dataset_root}/{test_model_type}/{test_task}/source/nyu_test_name100.txt'

    log_path = f"/gdata1/gaocs/Data_FQA/accuracy_log/{arch}/trained_{train_task}/{quant_type}{samples}_bitdepth{bit_depth}/{test_model_type}_{test_task}"
    log_file = f'{log_path}/{arch}_trained_{train_task}_eval_{test_task}_all.txt'
    print(log_file)

    correlation(source_file, log_file)