import modelSplit_v2
import consts
import logging
import os
import re
import numpy as np
import argparse
import sys
import random
import datetime
import torch
from utils import *
from torchvision.datasets.folder import pil_loader
import gc
import torch

gc.collect()

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

assert sys.version_info >= (3, 6),\
    "This script requires Python >= 3.6"  # TODO 3.7?
assert tuple(int(ver_num) for ver_num in torch.__version__.split('.')) >= (0, 4, 0),\
    "This script requires PyTorch >= 0.4.0"  # TODO 0.4.1?


def str_to_gender(s):
    s = str(s).lower()
    if s in ('m', 'man', '0'):
        return 0
    elif s in ('f', 'female', '1'):
        return 1
    else:
        raise KeyError("No gender found")


def str_to_bool(s):
    s = s.lower()
    if s in ('true', 't', 'yes', 'y', '1'):
        return True
    elif s in ('false', 'f', 'no', 'n', 'o'):
        return False
    else:
        raise KeyError("Invalid boolean")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AgeProgression on PyTorch.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--mode', choices=['train', 'test'], default='train')

    # train params
    parser.add_argument('--epochs', '-e', default=1, type=int)
    parser.add_argument(
        '--models-saving',
        '--ms',
        dest='models_saving',
        choices=('always', 'last', 'tail', 'never'),
        default='always',
        type=str,
        help='Model saving preference.{br}'
             '\talways: Save trained model at the end of every epoch (default){br}'
             '\tUse this option if you have a lot of free memory and you wish to experiment with the progress of your results.{br}'
             '\tlast: Save trained model only at the end of the last epoch{br}'
             '\tUse this option if you don\'t have a lot of free memory and removing large binary files is a costly operation.{br}'
             '\ttail: "Safe-last". Save trained model at the end of every epoch and remove the saved model of the previous epoch{br}'
             '\tUse this option if you don\'t have a lot of free memory and removing large binary files is a cheap operation.{br}'
             '\tnever: Don\'t save trained model{br}'
             '\tUse this option if you only wish to collect statistics and validation results.{br}'
             'All options except \'never\' will also save when interrupted by the user.'.format(br=os.linesep)
    )
    parser.add_argument('--batch-size', '--bs', dest='batch_size', default=64, type=int)
    parser.add_argument('--weight-decay', '--wd', dest='weight_decay', default=1e-5, type=float)
    parser.add_argument('--learning-rate', '--lr', dest='learning_rate', default=2e-4, type=float)
    parser.add_argument('--b1', '-b', dest='b1', default=0.9, type=float)
    parser.add_argument('--b2', '-B', dest='b2', default=0.999, type=float)
    parser.add_argument('--shouldplot', '--sp', dest='sp', default=False, type=bool)

    # test params
    parser.add_argument('--age', '-a', required=False, type=int)
    parser.add_argument('--gender', '-g', required=False, type=str_to_gender)
    parser.add_argument('--watermark', '-w', action='store_true')

    # shared params
    parser.add_argument('--cpu', '-c', action='store_true', help='Run on CPU even if CUDA is available.')
    parser.add_argument('--load', '-l', required=False, default=None, help='Trained models path for pre-training or for testing')
    parser.add_argument('--input', '-i', default=None, help='Training dataset path (default is {}) or testing image path'.format(default_train_results_dir()))
    parser.add_argument('--output', '-o', default='')
    parser.add_argument('-z', dest='z_channels', default=100, type=int, help='Length of Z vector')
    args = parser.parse_args()

    consts.NUM_Z_CHANNELS = args.z_channels
    net = modelSplit_v2.Net()

    if not args.cpu and torch.cuda.is_available():
        net.cuda()

    if args.mode == 'train':

        betas = (args.b1, args.b2) if args.load is None else None
        weight_decay = args.weight_decay if args.load is None else None
        lr = args.learning_rate if args.load is None else None

        if args.load is not None:
            net.load(args.load)
            print("Loading pre-trained models from {}".format(args.load))

        data_src = args.input or consts.UTKFACE_DEFAULT_PATH
        print("Data folder is {}".format(data_src))
        results_dest = args.output or default_train_results_dir()
        os.makedirs(results_dest, exist_ok=True)
        print("Results folder is {}".format(results_dest))

        with open(os.path.join(results_dest, 'session_arguments.txt'), 'w') as info_file:
            info_file.write(' '.join(sys.argv))

        log_path = os.path.join(results_dest, 'log_results.log')
        if os.path.exists(log_path):
            os.remove(log_path)
        logging.basicConfig(filename=log_path, level=logging.DEBUG)

        net.teachSplit(
            utkface_path=data_src,
            batch_size=args.batch_size,
            betas=betas,
            epochs=args.epochs,
            weight_decay=weight_decay,
            lr=lr,
            should_plot=args.sp,
            where_to_save=results_dest,
            models_saving=args.models_saving,
            explainable=True, # Keep True for enabling the xAI-System - Change to False for the Original CAAE !
            explanation_type='saliency'# If explainable = True, choose the explanatory method by typing 'saliency' or 'shap' !
        )

    elif args.mode == 'test':

        if args.load is None:
            raise RuntimeError("Must provide path of trained models")

        net.load(path=args.load, slim=True)

        results_dest = args.output or default_test_results_dir()
        results_dest = os.path.join(results_dest, str(args.age) + '.' + str(args.gender))
        if not os.path.isdir(results_dest):
            os.makedirs(results_dest)

        args.input = os.path.join(args.input, str(args.age) + '.' + str(args.gender))

        for x in range(0, consts.NUM_AGES):
            if not os.path.exists(os.path.join(results_dest, str(args.age) + '.' + str(args.gender) + '_to_' + str(x) + '.' + str(args.gender))):
                os.makedirs(os.path.join(results_dest, str(args.age) + '.' + str(args.gender) + '_to_' + str(x) + '.' + str(args.gender)))
        if not os.path.exists(os.path.join(results_dest, str(args.age) + '.' + str(args.gender) + '_to_all')):
            os.makedirs(os.path.join(results_dest,str(args.age) + '.' + str(args.gender) + '_to_all'))

        images = os.listdir(args.input)
        for image_name in images:
            image_tensor = pil_to_model_tensor_transform(pil_loader(os.path.join(args.input, image_name))).to(net.device)
            net.my_test_single(
                image_tensor=image_tensor,
                image_name=image_name,
                age=args.age,
                gender=args.gender,
                target=results_dest,
                watermark=args.watermark
            )
