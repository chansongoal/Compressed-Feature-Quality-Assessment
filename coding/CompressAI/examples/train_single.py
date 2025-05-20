# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

#gcs
import os
from compressai.datasets import FeatureFolder
from compressai.datasets import ConcatFeatureFolder
from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models
import gc


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]

def safe_item(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item()
    else:
        return float(x)

def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    loss_epoch = AverageMeter()
    bpp_loss_epoch = AverageMeter()
    mse_loss_epoch = AverageMeter()
    aux_loss_epoch = AverageMeter()

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        #gcs, update training loss
        aux_loss_epoch.update(model.aux_loss())
        bpp_loss_epoch.update(out_criterion["bpp_loss"])
        loss_epoch.update(out_criterion["loss"])
        mse_loss_epoch.update(out_criterion["mse_loss"])

        # if i % 100 == 0:
        #     print(
        #         f"Train epoch {epoch}: ["
        #         f"{i*len(d)}/{len(train_dataloader.dataset)}"
        #         f" ({100. * i / len(train_dataloader):.0f}%)]"
        #         f'\tLoss: {out_criterion["loss"].item():.7f} |'
        #         f'\tMSE loss: {out_criterion["mse_loss"].item():.7f} |'
        #         f'\tBpp loss: {out_criterion["bpp_loss"].item():.7f} |'
        #         f"\tAux loss: {aux_loss.item():.2f}"
        #     )
    #gcs, print average training loss
    print(
        f"Train epoch {epoch}: Average losses:"
        f"\tLoss: {loss_epoch.avg:.7f} |"
        f"\tBPFP loss: {bpp_loss_epoch.avg:.7f} |"
        f"\tMSE loss: {mse_loss_epoch.avg:.7f} |"
        f"\tAux loss: {aux_loss_epoch.avg:.7f}"
    ) 

    return safe_item(loss_epoch.avg), safe_item(bpp_loss_epoch.avg), safe_item(mse_loss_epoch.avg), safe_item(aux_loss_epoch.avg)

def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.7f} |"
        f"\tBPFP loss: {bpp_loss.avg:.7f} |"
        f"\tMSE loss: {mse_loss.avg:.7f} |"
        f"\tAux loss: {aux_loss.avg:.4f}"
    )

    # return loss.avg
    return safe_item(loss.avg), safe_item(bpp_loss.avg), safe_item(mse_loss.avg), safe_item(aux_loss.avg)


#gcs
def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    
    if is_best:
        #gcs
        best_checkpoint_name = f"{filename[:-8]}_best.pth.tar"
        # print(best_checkpoint_name)
        shutil.copyfile(filename, best_checkpoint_name)
        #gcs, remove duplicated filename
        os.remove(filename)

# Custom type parsing function
def parse_truncation(value):
    try:
        # Try to parse as a float
        return float(value)
    except ValueError:
        # Try to parse as a list of floats (comma-separated)
        try:
            return [float(x) for x in value.strip('[]').split(',')]
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid input for --truncation: {value}. Must be a float or a list of floats."
            )

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    #gcs, model_type="sd3", train_task="tti", trun_flag=False, trun_low=-20, trun_high=20, transform_type="uniform", qsamples=0, bit_depth=1, quant_points
    parser.add_argument(
        "-model_type",
        "--model_type",
        type=str,
        default="hybrid",
        help="Please input the model_type.",
    )
    parser.add_argument(
        "-train_task",
        "--train_task",
        type=str,
        default="hybrid",
        help="Please input the train_task.",
    )
    parser.add_argument(
        "-trun_flag",
        "--trun_flag",
        type=str,
        default='False',
        help="Please input the trun_flag.",
    )
    # lzj
    parser.add_argument(
        "-Prepro_flag",
        "--Prepro_flag",
        type=int,
        default=1,
        help="Please input the Prepro_flag.",
    )
    parser.add_argument(
        "-trun_low",
        "--trun_low",
        type=parse_truncation,
        default=-5,
        help="Please input the truncated upper value (float or list of floats).",
    )
    parser.add_argument(
        "-trun_high",
        "--trun_high",
        type=parse_truncation,
        default=5,
        help="Please input the truncated upper value (float or list of floats).",
    )
    parser.add_argument(
        "-transform_type",
        "--transform_type",
        type=str,
        default="uniform",
        help="Please input the transform_type.",
    )
    parser.add_argument(
        "-qsamples",
        "--qsamples",
        type=int,
        default=0,
        help="Please input the qsamples.",
    )
    parser.add_argument(
        "-bit_depth",
        "--bit_depth",
        type=int,
        default=1,
        help="Please input the bit_depth.",
    )
    parser.add_argument(
        "-transform_mapping_name",
        "--transform_mapping_name",
        type=str,
        default="None",
        help="Please input the transform_mapping_name filename.",
    )
    parser.add_argument(
        "-mp",
        "--savepath",
        type=str,
        help="Path to save trained models.",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=128,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(512, 512), #(hgt, wdt)
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument(
        "--save_period",
        type=int,
        default=20,
        help="Checkpoint save period (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    #gcs 
    # print(args.dataset)
    # train_dataset = FeatureFolder(args.dataset, split="train")
    # test_dataset = FeatureFolder(args.dataset, split="test")

    all_datasets = args.dataset.split(","); print(all_datasets)
    train_dataset = ConcatFeatureFolder(all_datasets, split="train", suffix='.npy')
    if args.train_task == 'cls': cls_dataset = FeatureFolder(all_datasets[0], split="test", suffix='.npy')
    elif args.train_task == 'seg': seg_dataset = FeatureFolder(all_datasets[0], split="test", suffix='.npy')
    elif args.train_task == 'dpt': 
        dpt_layer10_dataset = FeatureFolder(all_datasets[0], split="test", suffix='layer10.npy')
        dpt_layer20_dataset = FeatureFolder(all_datasets[0], split="test", suffix='layer20.npy')
        dpt_layer30_dataset = FeatureFolder(all_datasets[0], split="test", suffix='layer30.npy')
        dpt_layer40_dataset = FeatureFolder(all_datasets[0], split="test", suffix='layer40.npy')
    
    print(f"model_type={args.model_type}, train_task={args.train_task}, trun_flag={args.trun_flag}, trun_low={args.trun_low}, trun_high={args.trun_high}, transform_type={args.transform_type}, qsamples={args.qsamples}, bit_depth={args.bit_depth}, transform_mapping_name={args.transform_mapping_name}, patch_size={args.patch_size}, Prepro_flag={args.Prepro_flag}")
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )
    print('num_workers: ', args.num_workers)

    if args.train_task == 'cls': cls_loader = DataLoader(cls_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))
    elif args.train_task == 'seg': seg_loader = DataLoader(seg_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))
    elif args.train_task == 'dpt': 
        dpt_layer10_loader = DataLoader(dpt_layer10_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))
        dpt_layer20_loader = DataLoader(dpt_layer20_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))
        dpt_layer30_loader = DataLoader(dpt_layer30_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))
        dpt_layer40_loader = DataLoader(dpt_layer40_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device == "cuda"))

    net = image_models[args.model](quality=1)   # set default quality level to 1
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint and args.checkpoint != 'None':  # load from previous checkpoint
        print(f"Loading: {args.checkpoint} for pretraining")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # last_epoch = checkpoint["epoch"] + 1; 
        print(f'Continue pretrain on epoch {checkpoint["epoch"]}')

        state_dict = checkpoint["state_dict"]
        try:
            net.load_state_dict(state_dict)
            print('Load state dict (direct)')
        except RuntimeError as e1:
            print('Direct load failed, try removing or adding "module." prefix')

            try:
                new_state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
                net.load_state_dict(new_state_dict)
                print('Load state dict (after removing "module.")')
            except RuntimeError as e2:
                try:
                    new_state_dict = {f"module.{k}": v for k, v in state_dict.items()}
                    net.load_state_dict(new_state_dict)
                    print('Load state dict (after adding "module.")')
                except RuntimeError as e3:
                    print("Failed to load state_dict after trying all options.")
                    raise e3

        # net.load_state_dict(checkpoint["state_dict"]); 
        print('Load state dict')
        # optimizer.load_state_dict(checkpoint["optimizer"]); print('Load optimizer')
        # aux_optimizer.load_state_dict(checkpoint["aux_optimizer"]); print('Load aux_optimizer')
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"]); print('Load lr_scheduler')
        # gcs, init learning rate
        # optimizer.param_groups[0]['lr'] = args.learning_rate; print('Use the re-initilized learning rate')

    lr_end_threshold = 2e-8
    lr_patience = 20
    lr_patience_counter = 0
    lr_below_threshold = False

    best_loss = float("inf")
    epoch = last_epoch

    # for epoch in range(last_epoch, args.epochs):
    while True:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch}, Learning rate: {current_lr}")
        train_loss, train_bpfp_loss, train_mse_loss, train_aux_loss = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        # loss = test_epoch(epoch, test_dataloader, net, criterion)
        if args.train_task == 'cls': val_loss, val_bpfp_loss, val_mse_loss, val_aux_loss = test_epoch(epoch, cls_loader, net, criterion)
        elif args.train_task == 'seg': val_loss, val_bpfp_loss, val_mse_loss, val_aux_loss = test_epoch(epoch, seg_loader, net, criterion)
        elif args.train_task == 'dpt':
            val_loss_dpt_layer10, val_bpfp_loss_dpt_layer10, val_mse_loss_dpt_layer10, val_aux_loss_dpt_layer10 = test_epoch(epoch, dpt_layer10_loader, net, criterion)
            val_loss_dpt_layer20, val_bpfp_loss_dpt_layer20, val_mse_loss_dpt_layer20, val_aux_loss_dpt_layer20 = test_epoch(epoch, dpt_layer20_loader, net, criterion)
            val_loss_dpt_layer30, val_bpfp_loss_dpt_layer30, val_mse_loss_dpt_layer30, val_aux_loss_dpt_layer30 = test_epoch(epoch, dpt_layer30_loader, net, criterion)
            val_loss_dpt_layer40, val_bpfp_loss_dpt_layer40, val_mse_loss_dpt_layer40, val_aux_loss_dpt_layer40 = test_epoch(epoch, dpt_layer40_loader, net, criterion)

            val_loss = (val_loss_dpt_layer10 + val_loss_dpt_layer20 + val_loss_dpt_layer30 + val_loss_dpt_layer40) / 4
            val_bpfp_loss = (val_bpfp_loss_dpt_layer10 + val_bpfp_loss_dpt_layer20 + val_bpfp_loss_dpt_layer30 + val_bpfp_loss_dpt_layer40) / 4
            val_mse_loss = (val_mse_loss_dpt_layer10 + val_mse_loss_dpt_layer20 + val_mse_loss_dpt_layer30 + val_mse_loss_dpt_layer40) / 4
            val_aux_loss = (val_aux_loss_dpt_layer10 + val_aux_loss_dpt_layer20 + val_aux_loss_dpt_layer30 + val_aux_loss_dpt_layer40) / 4

        print(
            f"Test epoch {epoch}: Average losses:"
            f"\tLoss: {val_loss:.7f} |"
            f"\tBPFP loss: {val_bpfp_loss:.7f} |"
            f"\tMSE loss: {val_mse_loss:.7f} |"
            f"\tAux loss: {val_aux_loss:.7f}"
        )

        lr_scheduler.step(val_loss)

        is_best = val_loss < best_loss  # is this reasonable? what if MSE cannot measure semantic distortion?!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        best_loss = min(val_loss, best_loss)

        #gcs
        gc.collect()  # Run Python garbage collection
        torch.cuda.empty_cache()  # Then clear cached memory on GPU

        
        #gcs, save checkpoint
        if args.save:
            if is_best:
                print('Current best epoch: ', epoch)
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": val_loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    #gcs
                    args.savepath,
                )
            if (epoch % args.save_period == (args.save_period-1) or epoch == args.epochs - 1):
                checkpoint_name = f"{args.savepath[:-8]}_epoch{epoch}.pth.tar"
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": val_loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    #gcs
                    checkpoint_name,
                )
            #gcs, judge and count epochs
            if current_lr <= lr_end_threshold:
                if not lr_below_threshold:
                    print(f"LR dropped below {lr_end_threshold}, will train {lr_patience} more epochs.")
                    lr_below_threshold = True
                    lr_patience_counter = 0
                else:
                    lr_patience_counter += 1
                    print(f"LR below threshold for {lr_patience_counter}/{lr_patience} epochs.")
                    if lr_patience_counter >= lr_patience:
                        # save checkpoint
                        print(f"Save the last epoch {epoch} checkpoint")
                        checkpoint_name = f"{args.savepath[:-8]}_epoch{epoch}.pth.tar"
                        save_checkpoint(
                            {
                                "epoch": epoch,
                                "state_dict": net.state_dict(),
                                "loss": val_loss,
                                "optimizer": optimizer.state_dict(),
                                "aux_optimizer": aux_optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
                            },
                            is_best,
                            #gcs
                            checkpoint_name,
                        )
                        print("Stopping training due to low learning rate.")
                        break
        epoch += 1
if __name__ == "__main__":
    main(sys.argv[1:])
