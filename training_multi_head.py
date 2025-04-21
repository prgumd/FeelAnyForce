import argparse
import json
import logging
import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import utils
import wandb
from datetime import datetime
from pathlib import Path

from args import get_parser
from composed_model import ComposedModel
from dataset import TacForceDataset
from torch import nn
from torchvision import transforms as pth_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(0)
np.random.seed(0)
g = torch.Generator()
g.manual_seed(0)

dir_name = Path(os.path.abspath(os.path.dirname(__file__)))
os.environ["WANDB_SILENT"] = "true"


def train_regressor(args):
    """
    Main function for training the regressor and decoder.

    Arguments:
        args: Parsed arguments containing training configurations.
    """

    cudnn.benchmark = True

    # Load the model
    if args.tactile_backbone_training == 'calibration':
        ckpt = torch.load(os.path.join(args.ckpt_dir, "best_val.pth.tar"))
        if "config" in ckpt.keys():
            # Save new training values
            ckpt["config"]["data_basedir"] = args.data_basedir
            config = argparse.Namespace(**ckpt["config"])
            labels_train = args.labels_train
            tactile_backbone_train = 'calibration'
            labels_val = args.labels_val
            training_name = args.training_name + '_calibrated'
            layers_calibration = args.layers_calibration
            lr_calibration = args.lr_calibration
            epochs = args.epochs
            scheduler = args.scheduler

            # Update the config
            args.__dict__.update(vars(config))
            args.labels_val = labels_val
            args.training_name = training_name
            args.tactile_backbone_training = tactile_backbone_train
            args.layers_calibration = layers_calibration
            args.epochs = epochs
            args.lr_calibration = lr_calibration
            args.scheduler = scheduler

    model = ComposedModel(args)
    formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.tactile_backbone_training == 'calibration':
        model.load_state_dict(ckpt["state_dict"])
        logger.info('Loaded model from checkpoint')
    else:
        logger.info('New model created')

    # Handle mean and std
    with open(args.dataset_stats) as convert_file:
        dataset_mean_std = json.load(convert_file)

    if args.labels_train in dataset_mean_std and args.tactile_mode in dataset_mean_std[args.labels_train]:
        mean_rgb, std_rgb = dataset_mean_std[args.labels_train][args.tactile_mode]
        mean_depth, std_depth = dataset_mean_std[args.labels_train]['depth']
    else:
        logger.info('Calculating normalization parameters..')
        mean_rgb, std_rgb, mean_depth, std_depth = utils.compute_mean_std(
            torch.utils.data.DataLoader(TacForceDataset(args=args,
                                                        split="train", mode='compute_stats'),
                                        batch_size=args.batch_size_per_gpu))
        dataset_mean_std[args.labels_train] = {args.tactile_mode: [mean_rgb, std_rgb], 'depth': [mean_depth, std_depth]}

        with open(args.dataset_stats, 'w') as json_file:
            json_file.write(json.dumps(dataset_mean_std))

    logger.info(
        f"Dataset with {mean_rgb} mean and {std_rgb} std for rgb and {mean_depth} mean and {std_depth} std for depth.")
    if args.tactile_backbone_training == 'calibration':
        args.labels_train = labels_train

    transform_rgb = pth_transforms.Compose([
        pth_transforms.Pad([0, 40, 0, 40]),
        pth_transforms.Resize(args.crop_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean_rgb, std_rgb),
    ])

    transform_depth = pth_transforms.Compose([
        pth_transforms.Pad([0, 40, 0, 40]),
        pth_transforms.Resize(args.crop_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean_depth, std_depth),
    ])

    dataset_train = TacForceDataset(args=args,
                                    split="train",
                                    transform=transform_rgb,
                                    transform_depth=transform_depth)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    dataset_val = TacForceDataset(args=args,
                                  split="val",
                                  transform=transform_rgb,
                                  transform_depth=transform_depth)

    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if os.path.isfile(os.path.join(args.data_basedir, args.labels_test_seen)):
        dataset_test_seen = TacForceDataset(args=args,
                                            split="test_seen",
                                            transform=transform_rgb,
                                            transform_depth=transform_depth)

        test_loader_seen = torch.utils.data.DataLoader(
            dataset_test_seen,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    logger.info(f"Data loaded with {len(dataset_train)} train, {len(dataset_val)} val.")

    param_groups = model.get_param_groups(lr_backbone=args.lr_backbone,
                                          lr_architecture=args.lr_architecture,
                                          lr_calibration=args.lr_calibration)

    # Set optimizer
    optimizer = torch.optim.Adam(param_groups, args.lr_backbone, weight_decay=args.weight_decay)

    # Set name of the training
    exp_name = args.training_name
    if args.tactile_backbone_training == 'calibration':
        exp_name += '_calibration'
    exp_dir = os.path.join(args.ckpt_dir, exp_name)

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        logger.info(f"Directory created: {exp_dir}")
    elif not args.resume:
        logger.warning(f"Directory '{exp_dir}' already exists! Use --resume to continue.")

        exp_dir = os.path.join(args.ckpt_dir, f"{exp_name}_{formatted_time}")
        os.makedirs(exp_dir)
        logger.info(f"New directory created: {exp_dir}")

    args.exp_dir = exp_dir

    logger.info('Computing mean and std of the dataset')
    if args.wandb:
        wandb.login()

        wandb.init(
            project="TacForce_experiments",
            group=args.input_modality,
            name=exp_name,
            resume=args.resume,
            config=vars(args)
        )

    last_epoch = 0
    best_val_loss = np.inf
    if args.resume:
        ckpt = torch.load(exp_dir / "last.pth.tar")
        last_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])

        best_val_loss = ckpt["best_val_loss"]
        if args.load_backbone_pretrained_weights:
            ckpt = torch.load(args.tactile_weights / "best.pth.tar")
            logger.info(model.tactile_backbone.load_state_dict(ckpt["state_dict"]).keys())

    for epoch in range(last_epoch, args.epochs):
        train_stats = train(model, optimizer, train_loader, epoch, args)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        logger.info('Starting validation')

        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": vars(args)}

        torch.save(save_dict, os.path.join(exp_dir, "last.pth.tar"))
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:

            val_stats = validate_network(val_loader, model, args)

            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()}}

            if val_stats['loss'] < best_val_loss:
                logger.info(f"New optimum find at epoch {epoch}")
                best_val_loss = val_stats["loss"]
                torch.save(save_dict, os.path.join(exp_dir, "best_val.pth.tar"))
                test_stats = validate_network(test_loader_seen, model, args)
                log_stats = {**{k: v for k, v in log_stats.items()},
                             **{f'test_seen_{k}': v for k, v in test_stats.items()}}

        if args.wandb:
            wandb.log(log_stats, step=epoch)

        with open(os.path.join(exp_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

        if args.scheduler:
            scheduler.step()


def train(model, optimizer, loader, epoch, args):
    """
    Trains the model for one epoch.

    Arguments:
        model (nn.Module): The composed model containing backbone, regressor, and decoder.
        optimizer (Optimizer): Optimizer for training.
        loader (DataLoader): Training data loader.
        epoch (int): Current epoch number.
        args: Parsed arguments containing training configurations.

    Returns:
        dict: Dictionary of averaged metrics (e.g., loss, regressor_loss, decoder_loss).
    """

    model.train()
    regressor = model.regressor
    if args.tactile_backbone_training != "calibration":
        decoder = model.decoder

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for (tactile, target, depth) in metric_logger.log_every(loader, args.batch_size_per_gpu, header):
        # move to gpu
        tactile = tactile.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        depth = depth.cuda(non_blocking=True)
        intermediate_output = model.get_encoding(tactile)

        loss = 0
        regressor_output = regressor(intermediate_output)
        regressor_loss = model_loss(target, depth, regressor_output, 'regressor')
        loss += args.alpha_regressor * regressor_loss
        metric_logger.update(regressor_loss=regressor_loss.item())

        if args.tactile_backbone_training != "calibration":
            decoder_output = decoder(intermediate_output)
            decoder_loss = model_loss(target, depth, decoder_output, 'decoder')
            loss += args.alpha_decoder * decoder_loss
            metric_logger.update(decoder_loss=decoder_loss.item())

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    logger.info("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, args):
    """
    Evaluates the model on the validation dataset.

    Arguments:
        val_loader (DataLoader): Validation data loader.
        model (nn.Module): The composed model containing backbone, regressor, and decoder.
        args: Parsed arguments containing validation configurations.

    Returns:
        dict: Dictionary of averaged validation metrics.
    """
    model.eval()
    regressor = model.regressor
    if args.tactile_backbone_training != 'calibration':
        decoder = model.decoder

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for tactile, target, depth in metric_logger.log_every(val_loader, 20, header):

        # move to gpu
        tactile = tactile.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        depth = depth.cuda(non_blocking=True)

        with torch.no_grad():
            intermediate_output = model.get_encoding(tactile)

            loss = 0

            regressor_output = regressor(intermediate_output)
            regressor_loss = model_loss(target, depth, regressor_output, 'regressor')
            loss += args.alpha_regressor * regressor_loss
            metric_logger.update(regressor_loss=regressor_loss.item())

            if args.tactile_backbone_training != "calibration":
                decoder_output = decoder(intermediate_output)
                decoder_loss = model_loss(target, depth, decoder_output, 'decoder')
                loss += args.alpha_decoder * decoder_loss
                metric_logger.update(decoder_loss=decoder_loss.item())

        metric_logger.update(loss=loss.item())

    logger.info("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def model_loss(target, depth, output, head,):
    """
    Computes the loss for the model.
    
    Arguments:
        target (Tensor): Ground truth labels.
        depth (Tensor): Ground truth depth images.
        output (Tensor): Model output.
        head (str): Head name for loss computation.
    Returns:
        Tensor: Loss value.
    """
    if head == 'regressor':
        return nn.L1Loss()(output, target)

    elif head == 'decoder':
        return nn.MSELoss()(output, depth)

    else:
        logger.info("please select a valid architecture")
        exit(1)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    train_regressor(args)
