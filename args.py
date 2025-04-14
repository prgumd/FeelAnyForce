import argparse
import utils
import os
dir_name = os.path.abspath(os.path.dirname(__file__))
def get_parser():
    parser = argparse.ArgumentParser('FeelAnyForce Arguments')
    parser.add_argument('--alpha_decoder', type=float, default=10, help="""Coefficient additive loss for decoder""")
    parser.add_argument('--alpha_regressor', type=float, default=1, help="""Coefficient additive loss for regressor""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
                        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token. We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--ckpt_dir',
                        default=dir_name + "/checkpoint",
                        help='Path to save logs and checkpoints')
    parser.add_argument('--checkpoint',
                        default="best_val.pth.tar",
                        help='checkpoint path to be loaded')
    parser.add_argument('--bias', default=5, type=int, help="bias for depth normalization")
    parser.add_argument('--crop_size', default=224, type=int, help='image_size')
    parser.add_argument('--data_basedir', default=dir_name + '/dataset', help='Path to the dataset')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--forward', default=True, type=utils.bool_flag,
                        help="""Whether to do a simple forward or to concatenate intermediate layers.""")
    parser.add_argument('--input_modality', default='rgbd_concat_early', type=str,
                        help="Either rgb/depth/rgbd/rgbd_concat_early")
    parser.add_argument("--dataset_stats", type=str, default="mean_and_std.json",
                        help="json file for mean and std of datasets")
    parser.add_argument("--labels_test_seen", default="TacForce_test_set.csv", type=str,
                        help="path to validation labels")
    parser.add_argument("--labels_train", default="TacForce_train_set.csv", type=str,
                        help="path to train labels")
    parser.add_argument("--layers_calibration", default=1, type=int,
                        help="Number of layers of the regressor to be finetuned")
    parser.add_argument("--labels_val", default="TacForce_val_set.csv", type=str, help="path to validation labels")
    parser.add_argument('--load_backbone_pretrained_weights', action='store_true',
                        help='loads the weights specified in tactile_weights')
    parser.add_argument('--lr_architecture', default=0.00005, type=float, help="""Learning rate of the heads.""")
    parser.add_argument('--lr_backbone', default=0.00001, type=float, help="""Learning rate of the backbone.""")
    parser.add_argument('--lr_calibration', default=0.000001, type=float,
                        help="""Learning rate of the final layers for calibration.""")
    parser.add_argument('--n_last_blocks', default=1, type=int, help='Number of layers to be concatenated')
    parser.add_argument('--num_labels', default=3, type=int, help='Number of labels for regressor')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    parser.add_argument('--scheduler', type=utils.bool_flag, default=False, help="""Scheduler bool""")
    parser.add_argument('--tactile_backbone_training', default='finetune', type=str, help='finetune/calibration')
    parser.add_argument('--tactile_model', default='dinov2_vitb14', type=str, help='tactile backbone model')
    parser.add_argument('--tactile_mode', default='nobg', type=str, help="the tactile image post-processing")
    parser.add_argument('--tactile_repo', default='facebookresearch/dinov2', type=str, help='repository')
    parser.add_argument('--tactile_weights', default='', type=str, help='Tactile backbone model')
    parser.add_argument('--training_name', default='', type=str, help='exp name for training')
    parser.add_argument('--val_freq', default=1, type=int, help="validation epoch freq.")
    parser.add_argument('--weight_decay', default=0.0, type=float, help='L2 regularization')
    parser.add_argument('--wandb', default=False, type=utils.bool_flag, help="""Whether to use wandb or not.""")

    return parser