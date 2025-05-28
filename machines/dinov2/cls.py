import os
import json
# import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from dinov2.hub.classifiers import dinov2_vitg14_lc

import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Disable xFormers UserWarning
os.environ['USE_XFORMERS'] = '0'    # Disable xFormers to obtain/extract consistent features in multiple runs
import argparse

class DataFolder(datasets.ImageFolder):
    """Custom dataset class that includes the file path in the returned sample."""

    def __init__(self, root: str, transform=None, **kwargs):
        super().__init__(root, transform, **kwargs)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path


def get_label_from_file(filename, file_path):
    with open(file_path, 'r') as f:
        file_lines = f.readlines()
    
    for line in file_lines:
        parts = line.strip().split()
        if parts[0] == filename:
            return int(parts[-1])  # Return the last element (the number)
    
    return None  # Return None if the file name is not found

def get_label_from_dataset(test_dataset):
    err_list = [22, 30, 32, 36, 54, 59, 60, 67, 68, 120, 123, 134, 137, 138, 161, 170, 173, 184, 186, 193, 196, 204, 213, 231, 236, \
                240, 241, 249, 257, 259, 265, 272, 281, 288, 292, 297, 304, 319, 343, 353, 356, 358, 369, 381, 385, 392, 397, 424, 435, \
                444, 445, 460, 463, 470, 479, 480, 482, 484, 493, 501, 508, 519, 526, 527, 531, 534, 541, 544, 550, 561, 580, 582, 601, \
                605, 608, 616, 619, 620, 630, 639, 650, 651, 664, 673, 675, 676, 691, 702, 724, 733, 742, 743, 747, 750, 754, 776, 778, \
                782, 784, 787, 789, 790, 799, 814, 815, 826, 832, 834, 835, 836, 841, 848, 851, 857, 858, 876, 880, 885, 890, 892, 908, \
                911, 925, 928, 947, 952, 961, 966, 967, 970, 972, 983, 987]
    label_for_correct = []
    total_class = 1000
    required_class = 500
    label_idx = 0
    
    image_names = [sample[0] for sample in test_dataset.samples]
    for idx, image_name in enumerate(image_names):
        img_split = image_name.split('/')
        path_name, img_name = img_split[-2], img_split[-1][:-5]
        if not idx in err_list:
            # print(img_name, idx)    # for imagenet_selected_label500.txt
            # print(path_name, img_name)  # for imagenet_selected_pathname500.txt
            # print(img_name) # for cpu cluster
            label_for_correct.append(idx)
        if len(label_for_correct)==required_class:
            break
    return label_for_correct

def build_dataset(source_img_path: str, batch_size: int, transform: str ='test'):
    # Define data transformations
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    # Load test dataset and DataLoader
    dataset = DataFolder(source_img_path, transform=data_transform["test"])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return dataset, dataloader


def extract_features(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, org_feature_path: str):
    """Extract features from backbone"""
    model.eval()
    device = next(model.parameters()).device

    count = 0; max_samples = 10000
    for batch_x, batch_y, img_name in tqdm(data_loader):
        if count >= max_samples:
            break
        batch_x = batch_x.to(device)
        feature_list = model.forward_backbone(batch_x)
        feature = feature_list[0].unsqueeze(0)

        feat_name = img_name[0].split('/')[-1].split('.')[0]    # get the img name without '.JPEG'
        # print(img_name, img_name[0], feat_name)
        np.save(f'{org_feature_path}/{feat_name}.npy', feature.cpu().detach().numpy())

        count += len(batch_x); print(len(batch_x))

def evaluate_cls(model: torch.nn.Module, org_feature_path: str, rec_feature_path: str, source_label_name: str):
    """Evaluate image classification accuracy and feature reconstruction error.

    Args:
        model (torch.nn.Module): Classification model for evaluation.
        org_feature_path (str): Path to the original features.
        rec_feature_path (str): Path to the reconstructed features.
        source_label_name (str): Path to the source label file.

    Returns:
        Accuracy, feature MSE
    """
    model.eval()
    device = next(model.parameters()).device

    eval_acc = 0.0
    eval_mse = 0.0

    # Retrieve reconstructed feature filenames
    rec_feat_names = [f for f in os.listdir(rec_feature_path) if f.endswith('.npy')]

    for idx, rec_feat_name in enumerate(rec_feat_names):
        # Load reconstructed features
        rec_features_numpy = np.load(f"{rec_feature_path}/{rec_feat_name}")
        rec_features_tensor = torch.from_numpy(rec_features_numpy).to(device)

        with torch.no_grad():
            # Decode features and make predictions
            pred = torch.argmax(model.forward_head(rec_features_tensor), dim=1)

            # Compute accuracy using labels
            label = get_label_from_file(rec_feat_name.split('.')[0], source_label_name)
            label_tensor = torch.tensor(label).to(device)
            num_correct = (pred == label_tensor).sum().item()
            eval_acc += num_correct

            # Compute MSE between original and reconstructed features
            org_feat = np.load(f"{org_feature_path}/{rec_feat_name}")
            mse = np.mean(np.square(org_feat - rec_features_numpy))
            eval_mse += mse

    # Calculate and print metrics
    num_samples = len(rec_feat_names)

    return eval_acc*100 / num_samples, eval_mse / num_samples

def compute_token_cosine_similarity(x_orig, x_recon):
    assert x_orig.shape == x_recon.shape, "Shape mismatch"

    # Normalize token-wise
    x_orig_norm = x_orig / (np.linalg.norm(x_orig, axis=1, keepdims=True) + 1e-8)
    x_recon_norm = x_recon / (np.linalg.norm(x_recon, axis=1, keepdims=True) + 1e-8)

    # Element-wise cosine similarity per token
    cosine_sim = np.sum(x_orig_norm * x_recon_norm, axis=1)  # [N]

    return np.mean(cosine_sim)  # scalar

def compute_linear_CKA(x1, x2):
    assert x1.shape == x2.shape, "Feature shape mismatch"

    x1_centered = x1 - x1.mean(axis=0, keepdims=True)
    x2_centered = x2 - x2.mean(axis=0, keepdims=True)

    # HSIC numerator
    hsic = np.linalg.norm(x1_centered @ x2_centered.T, ord='fro') ** 2

    # Normalization denominator
    norm_x1 = np.linalg.norm(x1_centered @ x1_centered.T, ord='fro')
    norm_x2 = np.linalg.norm(x2_centered @ x2_centered.T, ord='fro')

    return hsic / (norm_x1 * norm_x2 + 1e-8)

def evaluate_cls_single_image(model: torch.nn.Module, org_feature_path: str, rec_feature_path: str, source_label_name: str):
    """Evaluate image classification accuracy and feature reconstruction error.

    Args:
        model (torch.nn.Module): Classification model for evaluation.
        org_feature_path (str): Path to the original features.
        rec_feature_path (str): Path to the reconstructed features.
        source_label_name (str): Path to the source label file.

    Returns:
        Accuracy, feature MSE
    """
    model.eval()
    device = next(model.parameters()).device

    eval_acc = 0.0
    eval_mse = 0.0

    # Retrieve reconstructed feature filenames
    rec_feat_names = [f for f in os.listdir(rec_feature_path) if f.endswith('.npy')]

    for idx, rec_feat_name in enumerate(rec_feat_names):
        # Load reconstructed features
        rec_features_numpy = np.load(f"{rec_feature_path}/{rec_feat_name}")
        rec_features_tensor = torch.from_numpy(rec_features_numpy).to(device)

        with torch.no_grad():
            # Decode features and make predictions
            # pred = torch.argmax(model.forward_head(rec_features_tensor), dim=1)
            # logits = model.forward_head(rec_features_tensor)
            logits = model.forward_head(rec_features_tensor).squeeze()

            # Compute accuracy using labels
            label = get_label_from_file(rec_feat_name.split('.')[0], source_label_name)

            #gcs, get rank
            sorted_indices = torch.argsort(logits, descending=True)
            rank = (sorted_indices == label).nonzero(as_tuple=True)[0].item() + 1
            # print(f"label: {label}, rank: {rank}")
            pred = torch.argmax(logits)

            label_tensor = torch.tensor(label).to(device)
            num_correct = (pred == label_tensor).sum().item()
            eval_acc += num_correct

            # Compute MSE between original and reconstructed features
            org_feat = np.load(f"{org_feature_path}/{rec_feat_name}")
            mse = np.mean(np.square(org_feat - rec_features_numpy))
            eval_mse += mse

            # Calculate metrics
            N = org_feat.shape[0]
            C = org_feat.shape[1]
            ssim_total = 0
            cosine_simi_total = 0
            cka_scroe_total = 0
            for n in range(N):
                for c in range(C):
                    # ssim_total += ssim_func(org_feat[n,c,:,:], rec_features_numpy[n,c,:,:])
                    cosine_simi_total += compute_token_cosine_similarity(org_feat[n,c,:,:], rec_features_numpy[n,c,:,:])
                    cka_scroe_total = compute_linear_CKA(org_feat[n,c,:,:], rec_features_numpy[n,c,:,:])  # between [N, C]
            # ssim_mean = ssim_total/N/C
            cosine_simi_mean = cosine_simi_total/N/C
            cka_scroe_mean = cka_scroe_total/N/C
            print(rec_feat_name, f"{rank}", f"{mse:.8f}", f"{cosine_simi_mean:.4f}", f"{cka_scroe_mean:.4f}")

    # Calculate and print metrics
    num_samples = len(rec_feat_names)

    return eval_acc*100 / num_samples, eval_mse / num_samples

def cls_pipeline(backbone_checkpoint_path: str, head_checkpoint_path: str, source_img_path: str, source_label_name: str, org_feature_path: str, rec_feature_path: str):
    """Main function to run the evaluation."""

    batch_size = 1
    test_dataset, test_dataloader = build_dataset(source_img_path, batch_size, 'test')
    
    # labels = get_label_from_dataset(test_dataset)   # comment this, only used in the first time to generate source label file

    # Initialize the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = dinov2_vitg14_lc(layers=1, pretrained=True, weights=[backbone_checkpoint_path, head_checkpoint_path])
    model.to(device)

    # Extract features
    # extract_features(model, test_dataloader, org_feature_path)

    # # Evaluate and print results
    # acc, feat_mse = evaluate_cls(model, org_feature_path, rec_feature_path, source_label_name)
    # print(f"Classification Accuracy: {acc:.4f}")
    # print(f"Feature MSE: {feat_mse:.8f}")

def transform_evaluation(transform_type, samples, bit_depth, source_name):
    # Set up paths
    backbone_checkpoint_path = '/gdata/gaocs/pretrained_models/dinov2/dinov2_vitg14_pretrain.pth'
    head_checkpoint_path = '/gdata/gaocs/pretrained_models/dinov2/dinov2_vitg14_cls_linear_head.pth'
    source_img_path = '/gpub/imagenet_raw/test'
    source_label_name = f'/gdata1/gaocs/FCM_LM_Test_Dataset/dinov2/cls/source/{source_name}'
    org_feature_path = '/gdata1/gaocs/FCM_LM_Test_Dataset/dinov2/cls/feature'
    root_path = f'/gdata1/gaocs/Data_FQA/inverse_transformed'; print('root_path: ', root_path)

    # Initialize the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = dinov2_vitg14_lc(layers=1, pretrained=True, weights=[backbone_checkpoint_path, head_checkpoint_path])
    model.to(device)

    # Evaluate and print results
    print(source_label_name)
    print(transform_type, samples, bit_depth) 

    rec_feature_path = f"{root_path}/dinov2_cls/{transform_type}{samples}_bitdepth{bit_depth}"
    acc, feat_mse = evaluate_cls(model, org_feature_path, rec_feature_path, source_label_name)
    print(f"Classification Accuracy: {acc:.4f}")
    print(f"Feature MSE: {feat_mse:.8f}")

def compressai_evaluation(arch, train_task, transform_type, samples, bit_depth, lambda_value_all, epochs, learning_rate, batch_size, patch_size):
    # Set up paths
    backbone_checkpoint_path = '/gdata/gaocs/pretrained_models/dinov2/dinov2_vitg14_pretrain.pth'
    head_checkpoint_path = '/gdata/gaocs/pretrained_models/dinov2/dinov2_vitg14_cls_linear_head.pth'
    source_img_path = '/gpub/imagenet_raw/test'
    source_label_name = '/gdata1/gaocs/FCM_LM_Test_Dataset/dinov2/cls/source/imagenet_selected_label100.txt'
    org_feature_path = '/gdata1/gaocs/FCM_LM_Test_Dataset/dinov2/cls/feature'
    root_path = f'/gdata1/gaocs/Data_FQA/decoded'; print('root_path: ', root_path)

    # Initialize the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = dinov2_vitg14_lc(layers=1, pretrained=True, weights=[backbone_checkpoint_path, head_checkpoint_path])
    model.to(device)
    
    # Evaluate and print results
    for lambda_value in lambda_value_all:
        print(source_label_name)
        print(arch, train_task, transform_type, samples, bit_depth, lambda_value, epochs, learning_rate, batch_size, patch_size)

        rec_feature_path = f"{root_path}/{arch}/trained_{train_task}/{transform_type}{samples}_bitdepth{bit_depth}/dinov2_cls/" \
                           f"lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}"
        
        acc, feat_mse = evaluate_cls(model, org_feature_path, rec_feature_path, source_label_name)
        print(f"Accuracy: {acc:.4f}")
        print(f"Feature MSE: {feat_mse:.8f}\n")

def compressai_evaluation_multiple(arch, train_task, transform_type, samples, bit_depth, lambda_value_all, epochs, learning_rate, batch_size, patch_size):
    # Set up paths
    backbone_checkpoint_path = '/gdata/gaocs/pretrained_models/dinov2/dinov2_vitg14_pretrain.pth'
    head_checkpoint_path = '/gdata/gaocs/pretrained_models/dinov2/dinov2_vitg14_cls_linear_head.pth'
    source_img_path = '/gpub/imagenet_raw/test'
    source_label_name = '/gdata1/gaocs/FCM_LM_Test_Dataset/dinov2/cls/source/imagenet_selected_label100.txt'
    org_feature_path = '/gdata1/gaocs/FCM_LM_Test_Dataset/dinov2/cls/feature'
    root_path = f'/gdata1/gaocs/Data_FQA/decoded'; print('root_path: ', root_path)

    # Initialize the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = dinov2_vitg14_lc(layers=1, pretrained=True, weights=[backbone_checkpoint_path, head_checkpoint_path])
    model.to(device)
    
    if train_task == 'cls':
        lambda_all = [0.0016, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.008, 0.01, 0.012, 0.015]
        epochs_all = [200, 200, 200, 1000, 200, 200, 200, 200, 200, 200]
        batch_size_all = [180, 180, 180, 180, 180, 180, 180, 180, 500, 500]
    elif train_task == 'hybrid':
        lambda_all = [0.0005, 0.001, 0.0025, 0.003, 0.004, 0.005, 0.006, 0.007, 0.01, 0.015]
        epochs_all = [1000, 1000, 200, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
        batch_size_all = [180, 180, 500, 180, 180, 180, 180, 180, 180, 180]
    
    # Evaluate and print results
    # lambda_all = lambda_all[:1]
    for idx, lambda_value in enumerate(lambda_all):
        epochs = epochs_all[idx]
        batch_size = batch_size_all[idx]

        print(source_label_name)
        print(arch, train_task, transform_type, samples, bit_depth, lambda_value, epochs, learning_rate, batch_size, patch_size)

        rec_feature_path = f"{root_path}/{arch}/trained_{train_task}/{transform_type}{samples}_bitdepth{bit_depth}/dinov2_cls/" \
                           f"lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}"
        
        acc, feat_mse = evaluate_cls_single_image(model, org_feature_path, rec_feature_path, source_label_name)
        print(f"Accuracy: {acc:.4f}")
        print(f"Feature MSE: {feat_mse:.8f}\n\n")

def h26x_evaluation_multiple(arch, transform_type, samples, bit_depth):
    # Set up paths
    backbone_checkpoint_path = '/gdata/gaocs/pretrained_models/dinov2/dinov2_vitg14_pretrain.pth'
    head_checkpoint_path = '/gdata/gaocs/pretrained_models/dinov2/dinov2_vitg14_cls_linear_head.pth'
    source_img_path = '/gpub/imagenet_raw/test'
    source_label_name = '/gdata1/gaocs/FCM_LM_Test_Dataset/dinov2/cls/source/imagenet_selected_label100.txt'
    org_feature_path = '/gdata1/gaocs/FCM_LM_Test_Dataset/dinov2/cls/feature'
    root_path = f'/gdata1/gaocs/Data_FQA/postprocessed'; print('root_path: ', root_path)

    # Initialize the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = dinov2_vitg14_lc(layers=1, pretrained=True, weights=[backbone_checkpoint_path, head_checkpoint_path])
    model.to(device)
    
    QP_all = [2,4,6,8,10,12,14,16,18,20]
    
    # Evaluate and print results
    for idx, QP in enumerate(QP_all):
        print(source_label_name)
        print(arch, transform_type, samples, bit_depth, QP)

        rec_feature_path = f"{root_path}/{arch}/{transform_type}{samples}_bitdepth{bit_depth}/dinov2_cls/QP{QP}"
        
        acc, feat_mse = evaluate_cls_single_image(model, org_feature_path, rec_feature_path, source_label_name)
        print(f"Accuracy: {acc:.4f}")
        print(f"Feature MSE: {feat_mse:.8f}\n\n")

def argument_parsing():
    parser = argparse.ArgumentParser(description="Train Evaluation Pipeline")
    parser.add_argument('--arch', type=str, default='bmshj2018-hyperprior', help='arch')
    parser.add_argument('--train_task', type=str, default='seg', help='train_task')
    parser.add_argument('--transform_type', type=str, default='kmeans', help='transform_type')
    parser.add_argument('--samples', type=int, default=10, help='samples')
    parser.add_argument('--bit_depth', type=int, default=8, help='bit_depth')
    parser.add_argument('--lambda_value_all', nargs='+', type=float, help='lambda_value_all')
    parser.add_argument('--epochs', type=int, default=200, help='epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning_rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--patch_size', type=str, default='256-256', help='patch_size')
    
    args = parser.parse_args()
    
    return args

# run below to evaluate the reconstructed features
if __name__ == "__main__":
    args = argument_parsing()
    arch = args.arch
    transform_type = args.transform_type
    samples = args.samples
    bit_depth = args.bit_depth
    train_task = args.train_task
    lambda_value_all = args.lambda_value_all
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    patch_size = args.patch_size

    # compressai_evaluation(arch, train_task, transform_type, samples, bit_depth, lambda_value_all, epochs, learning_rate, batch_size, patch_size)
    if arch == 'hyperprior':
        compressai_evaluation_multiple(arch, train_task, transform_type, samples, bit_depth, lambda_value_all, epochs, learning_rate, batch_size, patch_size)
    elif arch == 'vtm':
        h26x_evaluation_multiple(arch, transform_type, samples, bit_depth)
    elif arch == 'hm':
        h26x_evaluation_multiple(arch, transform_type, samples, bit_depth)

    # source_name = 'imagenet_selected_label100.txt'
    # transform_type = 'kmeans'; samples = 10; bit_depth = 8
    # transform_evaluation(transform_type, samples, bit_depth, source_name)
    # transform_type = 'kmeans'; samples = 10; bit_depth = 10
    # transform_evaluation(transform_type, samples, bit_depth, source_name)

# # run below to extract original features as the dataset. 
# # You can skip feature extraction if you have download the test dataset from https://drive.google.com/drive/folders/1RZFGlBd6wZr4emuGO4_YJWfKPtAwcMXQ
# if __name__ == "__main__":
#     backbone_checkpoint_path = '/gdata/gaocs/pretrained_models/dinov2/dinov2_vitg14_pretrain.pth'
#     head_checkpoint_path = '/gdata/gaocs/pretrained_models/dinov2/dinov2_vitg14_cls_linear_head.pth'
#     source_img_path = '/gpub/imagenet_raw/train'
#     source_label_name = '/home/gaocs/projects/FCM-LM/Data/dinov2/cls/source/imagenet_selected_label500.txt'
#     org_feature_path = '/gdata1/gaocs/FCM_LM_Train_Data/dinov2/cls/org_feat/train'
#     rec_feature_path = org_feature_path

#     cls_pipeline(backbone_checkpoint_path, head_checkpoint_path, source_img_path, source_label_name, org_feature_path, rec_feature_path)