#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 09:45:35 2019

@author: seykia
"""

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def compute_conv_out_size(d_in, h_in, w_in, padding, dilation, kernel_size, stride, UPorDW):
    if UPorDW == 'down':
        d_out = np.floor((d_in + 2 * padding[0] - dilation * (kernel_size - 1) - 1) / stride + 1)
        h_out = np.floor((h_in + 2 * padding[1] - dilation * (kernel_size - 1) - 1) / stride + 1)
        w_out = np.floor((w_in + 2 * padding[2] - dilation * (kernel_size - 1) - 1) / stride + 1)
    elif UPorDW == 'up':
        d_out = (d_in-1) * stride - 2 * padding[0] + dilation * (kernel_size - 1) + 1
        h_out = (h_in-1) * stride - 2 * padding[1] + dilation * (kernel_size - 1) + 1
        w_out = (w_in-1) * stride - 2 * padding[2] + dilation * (kernel_size - 1) + 1
    return d_out, h_out, w_out

################################ ARCHITECTURES ################################

class Encoder(nn.Module):
    def __init__(self, x, y, args):
        super(Encoder, self).__init__()
        self.r_dim = 25
        self.r_conv_dim = 100
        self.lrlu_neg_slope = 0.01
        self.dp_level = 0.1
        
        self.factor=args.m
        self.x_dim = x.shape[2]
        
        # Conv 1
        self.encoder_y_layer_1_conv = nn.Conv3d(in_channels = self.factor, out_channels=self.factor, 
                                                kernel_size=5, stride=2, padding=0, 
                                                dilation=1, groups=self.factor, bias=True) # in:(90,108,90) out:(43,52,43)
        self.encoder_y_layer_1_bn = nn.BatchNorm3d(self.factor)
        d_out_1, h_out_1, w_out_1 = compute_conv_out_size(y.shape[2], y.shape[3], 
                                                          y.shape[4], padding=[0,0,0], 
                                                          dilation=1, kernel_size=5, 
                                                          stride=2, UPorDW='down')
        
        # Conv 2
        self.encoder_y_layer_2_conv = nn.Conv3d(in_channels=self.factor, out_channels=self.factor, 
                                                kernel_size=3, stride=2, padding=0, 
                                                dilation=1, groups=self.factor, bias=True) # out: (21,25,21)
        self.encoder_y_layer_2_bn = nn.BatchNorm3d(self.factor)
        d_out_2, h_out_2, w_out_2 = compute_conv_out_size(d_out_1, h_out_1, 
                                                          w_out_1, padding=[0,0,0], 
                                                          dilation=1, kernel_size=3, 
                                                          stride=2, UPorDW='down')
        
        # Conv 3
        self.encoder_y_layer_3_conv = nn.Conv3d(in_channels=self.factor, out_channels=self.factor, 
                                                kernel_size=3, stride=2, padding=0, 
                                                dilation=1, groups=self.factor, bias=True) # out: (10,12,10)
        self.encoder_y_layer_3_bn = nn.BatchNorm3d(self.factor)
        d_out_3, h_out_3, w_out_3 = compute_conv_out_size(d_out_2, h_out_2, 
                                                          w_out_2, padding=[0,0,0], 
                                                          dilation=1, kernel_size=3, 
                                                          stride=2, UPorDW='down')
        
        # Conv 4
        self.encoder_y_layer_4_conv = nn.Conv3d(in_channels=self.factor, out_channels=1, 
                                                kernel_size=3, stride=2, padding=0, 
                                                dilation=1, groups=1, bias=True) # out: (4,5,4)
        self.encoder_y_layer_4_bn = nn.BatchNorm3d(1)
        d_out_4, h_out_4, w_out_4 = compute_conv_out_size(d_out_3, h_out_3, 
                                                          w_out_3, padding=[0,0,0], 
                                                          dilation=1, kernel_size=3, 
                                                          stride=2, UPorDW='down')
        self.cnn_feature_num = [1, int(d_out_4), int(h_out_4), int(w_out_4)]
        
        # FC 5
        self.encoder_y_layer_5_dp = nn.Dropout(p = self.dp_level)
        self.encoder_y_layer_5_linear = nn.Linear(int(np.prod(self.cnn_feature_num)), self.r_conv_dim)
        
        # FC 6
        self.encoder_xy_layer_6_dp = nn.Dropout(p = self.dp_level)
        self.encoder_xy_layer_6_linear = nn.Linear(self.r_conv_dim + self.x_dim, 50)
        
        # FC 7 
        self.encoder_xy_layer_7_dp = nn.Dropout(p = self.dp_level)
        self.encoder_xy_layer_7_linear = nn.Linear(50, self.r_dim)

    def forward(self, x, y):
        y = F.leaky_relu(self.encoder_y_layer_1_bn(
                self.encoder_y_layer_1_conv(y)), self.lrlu_neg_slope)
        y = F.leaky_relu(self.encoder_y_layer_2_bn(
                self.encoder_y_layer_2_conv(y)),self.lrlu_neg_slope)
        y = F.leaky_relu(self.encoder_y_layer_3_bn(
                self.encoder_y_layer_3_conv(y)),self.lrlu_neg_slope)
        y = F.leaky_relu(self.encoder_y_layer_4_bn(
                self.encoder_y_layer_4_conv(y)),self.lrlu_neg_slope)
        y = F.leaky_relu(self.encoder_y_layer_5_linear(self.encoder_y_layer_5_dp(
                y.view(y.shape[0], np.prod(self.cnn_feature_num)))), self.lrlu_neg_slope)
        x_y = torch.cat((y, torch.mean(x, dim=1)), 1)
        x_y = F.leaky_relu(self.encoder_xy_layer_6_linear(
                self.encoder_xy_layer_6_dp(x_y)),self.lrlu_neg_slope)
        x_y = F.leaky_relu(self.encoder_xy_layer_7_linear(
                self.encoder_xy_layer_7_dp(x_y)),self.lrlu_neg_slope)
        return x_y
    
    
class Decoder(nn.Module):
    def __init__(self, x, y, args):
        super(Decoder, self).__init__()
        self.r_dim = 25
        self.r_conv_dim = 100
        self.lrlu_neg_slope = 0.01
        self.dp_level = 0.1
        self.z_dim = 10
        self.x_dim = x.shape[2]
        self.cnn_feature_num = args.cnn_feature_num
        self.factor=args.m
        
        # FC 1
        self.decoder_zx_layer_1_dp = nn.Dropout(p = self.dp_level)
        self.decoder_zx_layer_1_linear = nn.Linear(self.z_dim + self.x_dim, 50)
        
        # FC 2
        self.decoder_zx_layer_2_dp = nn.Dropout(p = self.dp_level)
        self.decoder_zx_layer_2_linear = nn.Linear(50, int(np.prod(self.cnn_feature_num)))
        
        # Iconv 1
        self.decoder_zx_layer_1_iconv = nn.ConvTranspose3d(in_channels=1, out_channels=self.factor, 
                                                           kernel_size=3, stride=1, 
                                                           padding=0, output_padding=(0,0,0), 
                                                           groups=1, bias=True, dilation=1) 
        self.decoder_zx_layer_1_bn = nn.BatchNorm3d(self.factor)
        d_out_4, h_out_4, w_out_4 = compute_conv_out_size(args.cnn_feature_num[1]*2, 
                                                          args.cnn_feature_num[2]*2, 
                                                          args.cnn_feature_num[3]*2, 
                                                          padding=[0,0,0], 
                                                          dilation=1, kernel_size=3, 
                                                          stride=1, UPorDW='up')
    
        # Iconv 2
        self.decoder_zx_layer_2_iconv = nn.ConvTranspose3d(in_channels=self.factor, out_channels=self.factor, 
                                                           kernel_size=3, stride=1, padding=0, 
                                                           output_padding=(0,0,0), groups=self.factor, 
                                                           bias=True, dilation=1) 
        self.decoder_zx_layer_2_bn = nn.BatchNorm3d(self.factor)
        d_out_3, h_out_3, w_out_3 = compute_conv_out_size(d_out_4*2, 
                                                          h_out_4*2, 
                                                          w_out_4*2, 
                                                          padding=[0,0,0], 
                                                          dilation=1, kernel_size=3, 
                                                          stride=1, UPorDW='up')
        # Iconv 3
        self.decoder_zx_layer_3_iconv = nn.ConvTranspose3d(in_channels=self.factor, out_channels=self.factor, 
                                                           kernel_size=3, stride=1, padding=0, 
                                                           output_padding=(0,0,0), groups=self.factor, 
                                                           bias=True, dilation=1) 
        self.decoder_zx_layer_3_bn = nn.BatchNorm3d(self.factor)
        d_out_2, h_out_2, w_out_2 = compute_conv_out_size(d_out_3*2, 
                                                          h_out_3*2, 
                                                          w_out_3*2, 
                                                          padding=[0,0,0], 
                                                          dilation=1, kernel_size=3, 
                                                          stride=1, UPorDW='up')
        
        # Iconv 4        
        self.decoder_zx_layer_4_iconv = nn.ConvTranspose3d(in_channels=self.factor, out_channels=1, 
                                                           kernel_size=3, stride=1, padding=(0,0,0), 
                                                           output_padding= (0,0,0), groups=1, 
                                                           bias=True, dilation=1) 
        d_out_1, h_out_1, w_out_1 = compute_conv_out_size(d_out_2*2, 
                                                          h_out_2*2, 
                                                          w_out_2*2, 
                                                          padding=[0,0,0], 
                                                          dilation=1, kernel_size=3, 
                                                          stride=1, UPorDW='up')
        
        self.scaling = [y.shape[2]/d_out_1, y.shape[3]/h_out_1, 
                        y.shape[4]/w_out_1]
    
    def forward(self, z_sample, x_target):
        z_x = torch.cat([z_sample, torch.mean(x_target,dim=1)], dim=1)
        z_x = F.leaky_relu(self.decoder_zx_layer_1_linear(self.decoder_zx_layer_1_dp(z_x)), 
                           self.lrlu_neg_slope)
        z_x = F.leaky_relu(self.decoder_zx_layer_2_linear(self.decoder_zx_layer_2_dp(z_x)), 
                           self.lrlu_neg_slope)
        z_x = z_x.view(x_target.shape[0], self.cnn_feature_num[0], self.cnn_feature_num[1],
                       self.cnn_feature_num[2], self.cnn_feature_num[3])
        z_x = F.leaky_relu(self.decoder_zx_layer_1_bn(self.decoder_zx_layer_1_iconv(
                F.interpolate(z_x, scale_factor=2))), self.lrlu_neg_slope)
        z_x = F.leaky_relu(self.decoder_zx_layer_2_bn(self.decoder_zx_layer_2_iconv(
                F.interpolate(z_x, scale_factor=2))), self.lrlu_neg_slope)
        z_x = F.leaky_relu(self.decoder_zx_layer_3_bn(self.decoder_zx_layer_3_iconv(
                F.interpolate(z_x, scale_factor=2))), self.lrlu_neg_slope)
        z_x = self.decoder_zx_layer_4_iconv(F.interpolate(z_x, scale_factor=2))
        y_hat = torch.sigmoid(F.interpolate(z_x, scale_factor=(self.scaling[0],
                                                               self.scaling[1],self.scaling[2])))
        return y_hat
   