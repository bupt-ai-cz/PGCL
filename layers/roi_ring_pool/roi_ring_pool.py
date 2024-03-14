#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 14:50:56 2019

@author: vasgaoweithu
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from . import roi_ring_pool_cuda

class RoIRingPoolFunction(Function):
    @staticmethod
    def forward(ctx, pooled_height, pooled_width, spatial_scale, scale_inner, scale_outer, features, rois):
        ctx.pooled_height = pooled_height
        ctx.pooled_width = pooled_width
        ctx.spatial_scale = spatial_scale
        ctx.scale_inner = scale_inner
        ctx.scale_outer = scale_outer
        features=features[0]
        ctx.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        # concat_boxes = torch.cat([b.bbox for b in rois], dim=0)
        outputs=[]
        for i in range(batch_size):
            num_rois = len(rois[i].bbox)
            output = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_()
            ctx.argmax = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_().int()

            ctx.rois = rois[i].bbox
            ctx.processed_rois = features.new(num_rois, 9).zero_()
            
            RectangularRing(rois[i].bbox, ctx.processed_rois, ctx.spatial_scale, ctx.scale_inner, ctx.scale_outer)
            roi_ring_pool_cuda.forward(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                                            features, ctx.processed_rois, output, ctx.argmax)
            outputs += [output]
        output = torch.cat([b for b in outputs],dim=0)
        return output
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        assert(ctx.feature_size is not None and grad_output.is_cuda)
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        grad_input = grad_output.new(batch_size, num_channels, data_height, data_width).zero_()

        roi_ring_pool_cuda.backward(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                                        grad_output, ctx.processed_rois, grad_input, ctx.argmax)
        return None, None, None, None, None, grad_input, None
    
roi_ring_pool = RoIRingPoolFunction.apply

class RoIRingPool(nn.Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale,  scale_inner, scale_outer):
        super(RoIRingPool, self).__init__()
        
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.scale_inner = scale_inner
        self.scale_outer = scale_outer
    def forward(self, features, rois):
        return roi_ring_pool(self.pooled_height, self.pooled_width, self.spatial_scale, self.scale_inner, self.scale_outer,
                                   features, rois)


def RectangularRing(ss_rois, processed_rois,spatial_scale, scale_inner, scale_outer):
    #widths = rois[:, 3] - rois[:, 1] + 1.0
    #heights = rois[:, 4] - rois[:, 2] + 1.0
    #ctr_x = rois[:, 1] + 0.5 * widths
    #ctr_y = rois[:, 2] + 0.5 * heights
    
    rois = ss_rois.clone()

    ctr_x = (rois[:, 0] + rois[:, 2]) / 2
    ctr_y = (rois[:, 1] + rois[:, 3]) / 2
    w_half = (rois[:, 2] - rois[:, 0]) / 2
    h_half = (rois[:, 3] - rois[:, 1]) / 2
    
    # processed_rois[:, 1] = torch.tensor(ctr_x - w_half * scale_outer, dtype=rois.dtype, device=rois.device)  ##.mul_(spatial_scale).add_(0.5).floor_()
    # processed_rois[:, 2] = torch.tensor(ctr_y - h_half * scale_outer, dtype=rois.dtype, device=rois.device)  ##.mul_(spatial_scale).add_(0.5).floor_()
    # processed_rois[:, 3] = torch.tensor(ctr_x + w_half * scale_outer, dtype=rois.dtype, device=rois.device)  ##.mul_(spatial_scale).add_(-0.5).ceil_()
    # processed_rois[:, 4] = torch.tensor(ctr_y + h_half * scale_outer, dtype=rois.dtype, device=rois.device)  ##.mul_(spatial_scale).add_(-0.5).ceil_()
    # processed_rois[:, 5] = torch.tensor(ctr_x - w_half * scale_inner, dtype=rois.dtype, device=rois.device)  ##.mul_(spatial_scale).add_(0.5).floor_()
    # processed_rois[:, 6] = torch.tensor(ctr_y - h_half * scale_inner, dtype=rois.dtype, device=rois.device)  ##.mul_(spatial_scale).add_(0.5).floor_()
    # processed_rois[:, 7] = torch.tensor(ctr_x + w_half * scale_inner, dtype=rois.dtype, device=rois.device)  ##.mul_(spatial_scale).add_(-0.5).ceil_()
    # processed_rois[:, 8] = torch.tensor(ctr_y + h_half * scale_inner, dtype=rois.dtype, device=rois.device)

    processed_rois[:, 1] = (ctr_x - w_half * scale_outer).clone().detach().float().to(rois.device)
    processed_rois[:, 2] = (ctr_y - h_half * scale_outer).clone().detach().float().to(rois.device)
    processed_rois[:, 3] = (ctr_x + w_half * scale_outer).clone().detach().float().to(rois.device)
    processed_rois[:, 4] = (ctr_y + h_half * scale_outer).clone().detach().float().to(rois.device)
    processed_rois[:, 5] = (ctr_x - w_half * scale_inner).clone().detach().float().to(rois.device)
    processed_rois[:, 6] = (ctr_y - h_half * scale_inner).clone().detach().float().to(rois.device)
    processed_rois[:, 7] = (ctr_x + w_half * scale_inner).clone().detach().float().to(rois.device)
    processed_rois[:, 8] = (ctr_y + h_half * scale_inner).clone().detach().float().to(rois.device)

    if scale_inner == 0:
        processed_rois[:, 5:] = 0

    return 1