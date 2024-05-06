# -*- coding: utf-8 -*-
"""
Created on Oct 20 2023

@author: Omar Al-maqtari
"""

import argparse
import os
from Solver import Solver
from Data_loader import get_loader
from torch.backends import cudnn
import random
import torch


def main(config):
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if config.model_type not in ['EMT','CT_CrackSeg','TransUNet','SwinTransformerSeg','DeepLabv3plus','Efficientnet','Mobilenetv3','Shufflenetv2']:
        print('ERROR!! model_type should be selected in EMT/CT_CrackSeg/TransUNet/SwinTransformerSeg/DeepLabv3plus/Efficientnet/Mobilenetv3/Shufflenetv2')
        print('Your input for model_type was %s'%config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
        config.result_path = os.path.join(config.result_path,config.model_type)
    
    print(config)

    train_loader = get_loader(image_path=config.train_path,
                              image_height=config.image_height,
                              image_width=config.image_width,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_path=config.valid_path,
                              image_height=config.image_height,
                              image_width=config.image_width,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=0)
    test_loader = get_loader(image_path=config.test_path,
                             image_height=config.image_height,
                             image_width=config.image_width,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0)
    
    solver = Solver(config, train_loader, valid_loader, test_loader)
    

    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()
    
        
if __name__ == '__main__':
    for dataset in ['Crack11k','Crack500','DeepCrack','EugenMuller','CFD','GAPs384','RissbilderFlorian','SylvieChambon','Volker']:#'Crack11k','Crack500','DeepCrack','EugenMuller','CFD','GAPs384','RissbilderFlorian','SylvieChambon','Volker'
        for model_type in ['EMT','CT_CrackSeg','TransUNet','SwinTransformerSeg','DeepLabv3plus','Efficientnet','Mobilenetv3','Shufflenetv2']:#'EMT','CT_CrackSeg','TransUNet','SwinTransformerSeg','DeepLabv3plus','Efficientnet','Mobilenetv3','Shufflenetv2'
            parser = argparse.ArgumentParser()
            if dataset == 'EugenMuller':
                batch_size = 6
            else:
                batch_size = 8
            
            # model hyper-parameters
            parser.add_argument('--img_ch', type=int, default=3)
            parser.add_argument('--output_ch', type=int, default=1)
            parser.add_argument('--image_height', type=int, default=256)
            parser.add_argument('--image_width', type=int, default=256)
            parser.add_argument('--num_workers', type=int, default=0)
            
            # training hyper-parameters
            parser.add_argument('--lr', type=float, default=0.001)
            parser.add_argument('--num_epochs', type=int, default=200)
            parser.add_argument('--num_epochs_decay', type=int, default=5)
            parser.add_argument('--batch_size', type=int, default=batch_size)
            parser.add_argument('--loss_threshold', type=float, default=0.5)
            parser.add_argument('--beta1', type=float, default=0.9)       # momentum1 in Adam or SGD
            parser.add_argument('--beta2', type=float, default=0.999)     # momentum2 in Adam
            parser.add_argument('--loss_weight', type=float, default=2.61)# all=2.61
            parser.add_argument('--augmentation_prob', type=float, default=0.15)
            
            # misc
            parser.add_argument('--mode', type=str, default='train')  # add (+ '_Fine_Tune') to the name if you want to train or test with the Fine-tuning Model (FM)'
            parser.add_argument('--report_name', type=str, default='Segmentation Training ' + model_type)
            parser.add_argument('--dataset', type=str, default=dataset, help='Crack11k/Crack500/CrackTree/DeepCrack/EugenMuller/CFD/GAPs384/RissbilderFlorian/SylvieChambon/Volker')
            parser.add_argument('--model_type', type=str, default=model_type, help='EMT/CT_CrackSeg/TransUNet/SwinTransformerSeg/DeepLabv3plus/Efficientnet/Mobilenetv3/Shufflenetv2')
            parser.add_argument('--model_path', type=str, default='Enter the path where models will be saved .../models/')
            parser.add_argument('--result_path', type=str, default='Enter the path where results will be saved .../results/')
            parser.add_argument('--SR_path', type=str, default='Enter the path where segmentation results (Masks) will be saved .../'+dataset+'/SR/')
            parser.add_argument('--train_path', type=str, default='Enter the path where train images will be read from .../'+dataset+'/train/')
            parser.add_argument('--valid_path', type=str, default='Enter the path where valid images will be read from .../'+dataset+'/valid/')
            parser.add_argument('--test_path', type=str, default='Enter the path where test images will be read from .../'+dataset+'/test/')
            
            config = parser.parse_args()
            main(config)
            