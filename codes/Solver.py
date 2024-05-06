# -*- coding: utf-8 -*-
"""
Created on Oct 20 2023

@author: Omar Al-maqtari
"""

import os
import time
from datetime import datetime

import numpy as np

import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F

from EMT import *
from CT_CrackSeg import CT_CrackSeg
from vit_seg_modeling import CONFIGS, TransUNet
from SwinTransformer import SwinTransformerSeg
from DeepLabv3 import DeepLabv3plus
from Efficientnet import EfficientNetSeg, efficientnets
from Mobilenetv3 import MobileNetV3Seg
from Shufflenetv2 import ShuffleNetV2Seg, shufflenets

import csv
from Evaluation import *
from Loss import *
import matplotlib.pyplot as plt
    

class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):
        # Config
        self.cfg = config
        
		# Data loader
        self.mode = config.mode
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

		# Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.num_epochs_decay = config.num_epochs_decay
        self.augmentation_prob = config.augmentation_prob
        self.loss_weight = torch.Tensor([config.loss_weight])

		# Training settings
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.img_size = config.image_height
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        
		# Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.SR_path = config.SR_path
        
        # Report file
        self.report_name = config.report_name
        self.report = open(self.result_path+self.report_name+'.txt','a+')
        self.report.write('\n'+str(datetime.now()))
        self.report.write('\n'+str(config))
        
        # Models
        self.model = None
        #self.model1 = None    # model1 is for the Fine-tuning Model (FM), if you want to train or test with it, you must uncomment all the related lines
        self.optimizer = None
        self.optimizer1 = None
        self.model_type = config.model_type
        self.dataset = config.dataset
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss(pos_weight=self.loss_weight).to(self.device)
        self.DiceLoss = DiceLoss(threshold=config.loss_threshold).to(self.device)
        self.mIoULoss = mIoULoss(threshold=config.loss_threshold).to(self.device)
        
        self.net_path = os.path.join(self.model_path, self.report_name+'.pkl')
        #self.net_path = os.path.join(self.model_path, self.report_name.replace('_Fine_Tune','.pkl'))
        #self.net_path1 = os.path.join(self.model_path, self.report_name+'.pkl')
        
    def build_model(self):
        print("initialize model...")
        
        # EMT
        if self.model_type =='EMT':
            self.model = EMT([[96,96], [96,128,128], [128,192,192,192], [128,128,128]], self.output_ch)
            #self.model1 = FFM_Net(mapping_size=48)
        
        # CT_CrackSeg
        elif self.model_type =='CT_CrackSeg':
            self.model = CT_CrackSeg()
            
        # TransUNet
        elif self.model_type =='TransUNet':
            self.model = TransUNet(CONFIGS['R50-ViT-B_16'], img_size=self.img_size, num_classes=self.output_ch)
        
        # Swin Transformer Seg
        elif self.model_type =='SwinTransformerSeg':
            self.model = SwinTransformerSeg()
            
        # DeepLabv3plus
        elif self.model_type =='DeepLabv3plus':
            self.model = DeepLabv3plus(self.output_ch)
            
        # Efficientnet
        elif self.model_type =='Efficientnet':
            self.model = EfficientNetSeg(efficientnets[7], 32, 160, 640, self.output_ch)
        
        # Mobilenetv3
        elif self.model_type =='Mobilenetv3':
            self.model = MobileNetV3Seg(40, 112, 960, self.output_ch, 'large')
        
        # Shufflenetv2
        elif self.model_type =='Shufflenetv2':
            self.model = ShuffleNetV2Seg(shufflenets[3], 244, 488, 976, self.output_ch)
        
        self.optimizer = optim.Adam(self.model.parameters(), self.lr, [self.beta1, self.beta2], weight_decay=2e-4)
        #self.optimizer1 = optim.Adam(self.model1.parameters(), self.lr, [self.beta1, self.beta2], weight_decay=2e-4)
        self.lr_sch = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.8, patience=self.num_epochs_decay)
        
        self.model.to(self.device)
        #self.model1.to(self.device)
        
        if self.mode == 'train':
            self.print_network(self.model, self.model_type)
            #self.print_network(self.model1, self.model_type)
        
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        
        print(name)
        self.report.write('\n'+str(name))
        print("The number of parameters: {}".format(num_params))
        self.report.write("\n The number of parameters: {}".format(num_params))
        #print(model)
        self.report.write('\n'+str(model))
        
        
    def train(self):
        """Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#
        elapsed = 0.# Time of inference
        t = time.time()
        
        self.build_model()
        
		# Model Train
        if os.path.isfile(self.net_path):
            self.model = torch.load(self.net_path)
            
            Train_results = open(self.result_path+self.report_name+'_Train_result.csv', 'a', encoding='utf-8', newline='')
            twr = csv.writer(Train_results)
            
            Valid_results = open(self.result_path+self.report_name+'_Valid_result.csv', 'a', encoding='utf-8', newline='')
            vwr = csv.writer(Valid_results)
            
        else:
            Train_results = open(self.result_path+self.report_name+'_Train_result.csv', 'a', encoding='utf-8', newline='')
            twr = csv.writer(Train_results)
            
            Valid_results = open(self.result_path+self.report_name+'_Valid_result.csv', 'a', encoding='utf-8', newline='')
            vwr = csv.writer(Valid_results)
            
            twr.writerow(['Train_model','Model_type','Dataset','LR','Epochs','Augmentation_prob'])
            twr.writerow([self.report_name,self.model_type,self.dataset,self.lr,self.num_epochs,self.augmentation_prob])
            twr.writerow(['Epoch','Acc','RC','PC','F1','IoU','mIoU','OIS','AIU','DC'])
            
            vwr.writerow(['Train_model','Model_type','Dataset','LR','Epochs','Augmentation_prob'])
            vwr.writerow([self.report_name,self.model_type,self.dataset,self.lr,self.num_epochs,self.augmentation_prob])
            vwr.writerow(['Epoch','Acc','RC','PC','F1','IoU','mIoU','OIS','AIU','DC'])
            
        # Training
        best_model_score = 0.
        results = [["Loss",[],[]],["Acc",[],[]],["RC",[],[]],["PC",[],[]],["F1",[],[]],["IoU",[],[]],["mIoU",[],[]],["OIS",[],[]],["AIU",[],[]],["DC",[],[]]]
            
        factor = 1
        for epoch in range(self.num_epochs):
            
            self.model.train(True)
            #self.model.train(True)
            train_loss = 0.
            Acc = 0.	# Accuracy
            RC = 0.		# Recall (Sensitivity)
            PC = 0. 	# Precision
            F1 = 0.		# F1 Score
            IoU = 0.    # Intersection over Union (Jaccard Index)
            mIoU = 0.	# mean of Intersection over Union (mIoU)
            OIS = 0.    # 
            AIU = 0.    #
            DC = 0.		# Dice Coefficient
            length = 0
                    
            if (epoch+1)%70 == 0:
                factor -= 0.2
                
            for i, (image, GT, name) in enumerate(self.train_loader):
                
                # SR : Segmentation Result
                # GT : Ground Truth
                image = image.to(self.device)
                GT = GT.to(self.device)
                
                for p in self.model.parameters(): p.require_grad=True
                #for p in self.model1.parameters(): p.require_grad=True
                #with torch.no_grad():
                SR = self.model(image)
                #SR = self.model1(SR)
                
                SR_f = SR.view(-1)
                GT_f = GT.view(-1)
                
                loss1 = self.BCEWithLogitsLoss(SR_f,GT_f)
                loss2 = self.DiceLoss(SR_f,GT_f)
                loss3 = self.mIoULoss(SR_f,GT_f)
                total_loss = loss1 + (factor*(loss2+loss3))
                
                self.model.zero_grad()
                #self.model1.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                #self.optimizer1.step()
                
                # Detach from GPU memory
                SR_f = SR_f.detach()
                GT_f = GT_f.detach()
                
                train_loss += total_loss.detach().item()
                Acc += get_Accuracy(SR_f,GT_f)[1]
                RC += get_F1(SR_f,GT_f)[3]
                PC += get_F1(SR_f,GT_f)[7]
                OIS += get_F1(SR_f,GT_f)[0]
                IoU += get_mIoU(SR_f,GT_f)[1]
                mIoU += get_mIoU(SR_f,GT_f)[5]
                AIU += get_mIoU(SR_f,GT_f)[0]
                DC += get_DC(SR_f,GT_f)[1]
                length += 1
                
            train_loss = train_loss/length
            Acc = Acc/length
            RC = RC/length
            PC = PC/length
            F1 = (2*RC*PC)/(RC+PC+1e-12)
            IoU = IoU/length
            mIoU = mIoU/length
            OIS = OIS/length
            AIU = AIU/length
            DC = DC/length
            
            results[0][1].append((train_loss))
            results[1][1].append((Acc*100))
            results[2][1].append((RC*100))
            results[3][1].append((PC*100))
            results[4][1].append((F1*100))
            results[5][1].append((IoU*100))
            results[6][1].append((mIoU*100))
            results[7][1].append((OIS*100))
            results[8][1].append((AIU*100))
            results[9][1].append((DC*100))
            
            # Print the log info
            print('\nEpoch [%d/%d] \nTrain Loss: %.4f \n[Training] Acc: %.4f, RC: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, OIS: %.4f, AIU: %.4f, DC: %.4f' % (
                epoch+1, self.num_epochs, train_loss, Acc, RC, PC, F1, IoU, mIoU, OIS, AIU, DC))
            self.report.write('\nEpoch [%d/%d] \nTrain Loss: %.4f \n[Training] Acc: %.4f, RC: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, OIS: %.4f, AIU: %.4f, DC: %.4f' % (
                epoch+1, self.num_epochs, train_loss, Acc, RC, PC, F1, IoU, mIoU, OIS, AIU, DC))
            twr.writerow([epoch+1, Acc, RC, PC, F1, IoU, mIoU, OIS, AIU, DC])
            
    		# Clear unoccupied GPU memory after each epoch
            torch.cuda.empty_cache()
            
            #========================== Validation ====================================#
            
            self.model.train(False)
            #self.model1.train(False)
            valid_loss = 0.
            Acc = 0.	# Accuracy
            RC = 0.		# Recall (Sensitivity)
            PC = 0. 	# Precision
            F1 = 0.		# F1 Score
            IoU = 0     # Intersection over Union (Jaccard Index)
            mIoU = 0.	# mean of Intersection over Union (mIoU)
            OIS = 0.    # 
            AIU = 0.    #
            DC = 0.		# Dice Coefficient
            length = 0
            
            for i, (image, GT, name) in enumerate(self.valid_loader):
                
                # SR : Segmentation Result
                # GT : Ground Truth
                image = image.to(self.device)
                GT = GT.to(self.device)
                
                with torch.no_grad():
                    SR = self.model(image)
                    #SR = self.model1(SR)
                
                SR_f = SR.view(-1)
                GT_f = GT.view(-1)
        
                loss1 = self.BCEWithLogitsLoss(SR_f,GT_f)
                loss2 = self.DiceLoss(SR_f,GT_f)
                loss3 = self.mIoULoss(SR_f,GT_f)
                total_loss = loss1 + (factor*(loss2+loss3))
               
                # Detach from GPU memory
                SR_f = SR_f.detach()
                GT_f = GT_f.detach()
        
                # Get metrices results
                valid_loss += total_loss.detach().item()
                Acc += get_Accuracy(SR_f,GT_f)[1]
                RC += get_F1(SR_f,GT_f)[3]
                PC += get_F1(SR_f,GT_f)[7]
                OIS += get_F1(SR_f,GT_f)[0]
                IoU += get_mIoU(SR_f,GT_f)[1]
                mIoU += get_mIoU(SR_f,GT_f)[5]
                AIU += get_mIoU(SR_f,GT_f)[0]
                DC += get_DC(SR_f,GT_f)[1]
                length += 1
                
            valid_loss = valid_loss/length
            Acc = Acc/length
            RC = RC/length
            PC = PC/length
            F1 = (2*RC*PC)/(RC+PC+1e-12)
            IoU = IoU/length
            mIoU = mIoU/length
            OIS = OIS/length
            AIU = AIU/length
            DC = DC/length
            model_score = F1
            
            results[0][2].append((valid_loss))
            results[1][2].append((Acc*100))
            results[2][2].append((RC*100))
            results[3][2].append((PC*100))
            results[4][2].append((F1*100))
            results[5][2].append((IoU*100))
            results[6][2].append((mIoU*100))
            results[7][2].append((OIS*100))
            results[8][2].append((AIU*100))
            results[9][2].append((DC*100))
            
            print('\nVal Loss: %.4f \n[Validation] Acc: %.4f, RC: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, OIS: %.4f, AIU: %.4f, DC: %.4f'%(
                valid_loss, Acc, RC, PC, F1, IoU, mIoU, OIS, AIU, DC))
            self.report.write('\nVal Loss: %.4f \n[Validation] Acc: %.4f, RC: %.4f, PC: %.4f, F1: %.4f, IoU: %.4f, mIoU: %.4f, OIS: %.4f, AIU: %.4f, DC: %.4f'%(
                valid_loss, Acc, RC, PC, F1, IoU, mIoU, OIS, AIU, DC))
            vwr.writerow([epoch+1, Acc, RC, PC, F1, IoU, mIoU, OIS, AIU, DC])
            
            # Decay learning rate
            self.lr_sch.step(valid_loss)
            print(self.lr_sch.get_last_lr())
            
            # Save Best Model
            if model_score > best_model_score:
                best_model_score = model_score
                print('\nBest %s model score : %.4f'%(self.model_type,best_model_score))
                self.report.write('\nBest %s model score : %.4f'%(self.model_type,best_model_score))
                torch.save(self.model,self.net_path)
                #torch.save(self.model1,self.net_path1)
                    
            # Clear unoccupied GPU memory after each epoch
            torch.cuda.empty_cache()
            
        displayfigures(results, self.result_path, self.report_name)
        
        Train_results.close()
        Valid_results.close()
        elapsed = time.time() - t
        print("\nElapsed time: %f seconds.\n\n" %elapsed)
        self.report.write("\nElapsed time: %f seconds.\n\n" %elapsed)
        self.report.close()
        
                    
    def test(self):		
		#===================================== Test ====================================#
        
        # Load Trained Model
        if os.path.isfile(self.net_path):
            self.build_model()
			# Load the pretrained Encoder
            self.model = torch.load(self.net_path,map_location='cpu').to(self.device)
            #self.model1 = torch.load(self.net_path1,map_location='cpu').to(self.device)
            print('%s is Successfully Loaded from %s'%(self.model_type,self.net_path))
            self.report.write('\n%s is Successfully Loaded from %s'%(self.model_type,self.net_path))
        else: 
            print("Trained model NOT found, Please train a model first")
            self.report.write("\nTrained model NOT found, Please train a model first")
            return
        
        self.model.train(False)
        #self.model1.train(False)
        Acc = 0.	# Accuracy
        RC = 0.		# Recall (Sensitivity)
        PC = 0. 	# Precision
        F1 = 0.		# F1 Score
        IoU = 0     # Intersection over Union (Jaccard Index)
        mIoU = 0.	# mean of Intersection over Union (mIoU)
        OIS = 0.    # 
        AIU = 0.    #
        DC = 0.		# Dice Coefficient
        length = 0
        elapsed = 0.# Time of inference
        threshold = 0
        RC_curve = 0.
        PC_curve = 0.
        RC_all = []
        PC_all = []
        
        
        for i, (image, GT, name) in enumerate(self.test_loader):
            
            # SR : Segmentation Result
            # GT : Ground Truth
            image = image.to(self.device)
            GT = GT.to(self.device)
            
            #Time of inference
            t = time.time()
            
            with torch.no_grad():
                SR = self.model(image)
                #SR = self.model1(SR)
            
            elapsed = (time.time() - t)
            
            # Detach from GPU memory
            SR_f = SR.view(-1)
            GT_f = GT.view(-1)
            SR_f = SR_f.detach()
            GT_f = GT_f.detach()
            
            Acc += get_Accuracy(SR_f,GT_f)[1]
            RC += get_F1(SR_f,GT_f)[3]
            RC_all.append(get_F1(SR_f,GT_f)[5])
            PC += get_F1(SR_f,GT_f)[7]
            PC_all.append(get_F1(SR_f,GT_f)[9])
            OIS += get_F1(SR_f,GT_f)[0]
            IoU += get_mIoU(SR_f,GT_f)[1]
            mIoU += get_mIoU(SR_f,GT_f)[5]
            AIU += get_mIoU(SR_f,GT_f)[0]
            DC += get_DC(SR_f,GT_f)[1]
            length += 1

        Acc = Acc/length
        RC = RC/length
        PC = PC/length
        F1 = (2*RC*PC)/(RC+PC+1e-12)
        IoU = IoU/length
        mIoU = mIoU/length
        OIS = OIS/length
        AIU = AIU/length
        DC = DC/length
        elapsed = elapsed/(SR.size(0))
        model_score = F1
        RC_curve, PC_curve = PRC(PC_all, RC_all, self.result_path, self.report_name)
        PRC_report = open(self.result_path+self.report_name+'_PRC.txt','a+')
        PRC_report.write('\n\n Recall = '+str(RC_curve))
        PRC_report.write('\n Precision = '+str(PC_curve))
        PRC_report.close()
        
        f = open(os.path.join(self.result_path,'Test_result.csv'), 'a', encoding='utf-8', newline='')
        wr = csv.writer(f)
        wr.writerow(['Report_file','Model_type','Dataset','Acc','RC','PC','F1','IoU','mIoU','OIS','AIU','DC','Net_score','Time of inference','Threshold','LR','Epochs','Augmentation_prob'])
        wr.writerow([self.report_name,self.model_type,self.dataset,Acc,RC,PC,F1,IoU,mIoU,OIS,AIU,DC,model_score,elapsed,threshold,self.lr,self.num_epochs,self.augmentation_prob])
        f.close()
        
        print('Results have been Saved')
        self.report.write('\nResults have been Saved\n\n')
        
        self.report.close()
        
        