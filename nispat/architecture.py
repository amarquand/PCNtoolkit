#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:31:59 2019

@author: seykia
"""
import torch
from torch import nn
from torch.nn import functional as F

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
        
        self.encoder_y_layer_1_conv = nn.Conv3d(in_channels = self.factor, out_channels=25, 
                                                kernel_size=3, stride=1, padding=0, 
                                                dilation=1, groups=1, bias=True) # in:(1,49,61,40) out:(5,47,59,38)
        self.encoder_y_layer_1_bn = nn.BatchNorm3d(25)
        self.encoder_y_layer_1_pool = nn.AvgPool3d(kernel_size=3, stride=2, padding=0, 
                                                   ceil_mode=False) # out: (5,23,29,18)
        self.encoder_y_layer_2_conv = nn.Conv3d(in_channels=25, out_channels=15, 
                                                kernel_size=3, stride=1, padding=0, 
                                                dilation=1, groups=1, bias=True) # out: (7,21,27,16)
        self.encoder_y_layer_2_bn = nn.BatchNorm3d(15)
        self.encoder_y_layer_2_pool = nn.AvgPool3d(kernel_size=3, stride=2, 
                                                   padding=0, ceil_mode=False) # out: (7,10,13,7)
        self.encoder_y_layer_3_conv = nn.Conv3d(in_channels=15, out_channels=10, 
                                                kernel_size=3, stride=1, padding=0, 
                                                dilation=1, groups=1, bias=True) # out: (10,8,11,5)
        self.encoder_y_layer_3_bn = nn.BatchNorm3d(10)
        self.encoder_y_layer_3_pool = nn.AvgPool3d(kernel_size=2, stride=2, padding=0, 
                                                   ceil_mode=False) # out: (10,4,5,2) = 400
        self.encoder_y_layer_4_dp = nn.Dropout(p = self.dp_level)
        self.encoder_y_layer_4_linear = nn.Linear(400, self.r_conv_dim)
        
        self.encoder_xy_layer_5_dp = nn.Dropout(p = self.dp_level)
        self.encoder_xy_layer_5_linear = nn.Linear(self.r_conv_dim + self.x_dim, 100)
        self.encoder_xy_layer_6_dp = nn.Dropout(p = self.dp_level)
        self.encoder_xy_layer_6_linear = nn.Linear(100, self.r_dim)

    def forward(self, x, y):
        y = self.encoder_y_layer_1_pool(F.leaky_relu(self.encoder_y_layer_1_bn(
                self.encoder_y_layer_1_conv(y)),self.lrlu_neg_slope))
        y = self.encoder_y_layer_2_pool(F.leaky_relu(self.encoder_y_layer_2_bn(
                self.encoder_y_layer_2_conv(y)),self.lrlu_neg_slope))
        y = self.encoder_y_layer_3_pool(F.leaky_relu(self.encoder_y_layer_3_bn(
                self.encoder_y_layer_3_conv(y)),self.lrlu_neg_slope))
        y = F.leaky_relu(self.encoder_y_layer_4_linear(self.encoder_y_layer_4_dp(
                y.view(y.shape[0],400))),self.lrlu_neg_slope)
        x_y = torch.cat((y, torch.mean(x, dim=1)), 1)
        x_y = F.leaky_relu(self.encoder_xy_layer_5_linear(
                self.encoder_xy_layer_5_dp(x_y)),self.lrlu_neg_slope)
        x_y = F.leaky_relu(self.encoder_xy_layer_6_linear(
                self.encoder_xy_layer_6_dp(x_y)),self.lrlu_neg_slope)
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
        
        self.decoder_zx_layer_1_dp = nn.Dropout(p = self.dp_level)
        self.decoder_zx_layer_1_linear = nn.Linear(self.z_dim + self.x_dim, 100)
        self.decoder_zx_layer_2_dp = nn.Dropout(p = self.dp_level)
        self.decoder_zx_layer_2_linear = nn.Linear(100, 400)
        self.decoder_zx_layer_3_iconv = nn.ConvTranspose3d(in_channels=10, out_channels=15, 
                                                           kernel_size=3, stride=1, 
                                                           padding=0, output_padding=(0,0,0), 
                                                           groups=1, bias=True, dilation=1) # out: (10,12,6) 
        self.decoder_zx_layer_3_bn = nn.BatchNorm3d(15)
        self.decoder_zx_layer_4_iconv = nn.ConvTranspose3d(in_channels=15, out_channels=25, 
                                                           kernel_size=3, stride=1, padding=0, 
                                                           output_padding=(0,0,0), groups=1, 
                                                           bias=True, dilation=1) # out: (22,26,14)
        self.decoder_zx_layer_4_bn = nn.BatchNorm3d(25)
        self.decoder_zx_layer_5_iconv = nn.ConvTranspose3d(in_channels=25, out_channels=1, 
                                                           kernel_size=3, stride=1, padding=(0,0,0), 
                                                           output_padding= (0,0,0), groups=1, 
                                                           bias=True, dilation=1) # out: (49,61,40) 
    
    def forward(self, z_sample, x_target):
        z_x = torch.cat([z_sample, torch.mean(x_target,dim=1)], dim=1)
        z_x = F.leaky_relu(self.decoder_zx_layer_1_linear(self.decoder_zx_layer_1_dp(z_x)), 
                           self.lrlu_neg_slope)
        z_x = F.leaky_relu(self.decoder_zx_layer_2_linear(self.decoder_zx_layer_2_dp(z_x)), 
                           self.lrlu_neg_slope)
        z_x = z_x.view(x_target.shape[0],10,4,5,2)
        z_x = F.leaky_relu(self.decoder_zx_layer_3_bn(self.decoder_zx_layer_3_iconv(
                F.interpolate(z_x, scale_factor=2))), self.lrlu_neg_slope)
        z_x = F.leaky_relu(self.decoder_zx_layer_4_bn(self.decoder_zx_layer_4_iconv(
                F.interpolate(z_x, scale_factor=2))), self.lrlu_neg_slope)
        y_hat = torch.sigmoid(self.decoder_zx_layer_5_iconv(F.interpolate(z_x, 
                                                                          scale_factor=(2.14,2.27,2.72))))
        return y_hat
   