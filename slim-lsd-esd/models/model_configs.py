# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from models.meanflow import MeanFlow
from models.lagrangian import Lagrangian
from models.eulerian import Eulerian
from models.flowmatching import FlowMatching
from models.unet import SongUNet
from models.mg_unet import UNetModel, attn_bools_to_dhariwal_attn_res
from models.edm2_unet import EDM2UNet

MODEL_ARCHS = {
    "song_unet": SongUNet,
    'mg_unet': UNetModel,
    'small_unet': SongUNet,
    'edm2_unet': EDM2UNet,
}

MODEL_CONFIGS = {
    "mg_unet": {
        'channel_mult': (1,2,2,2),
        'attention_resolutions': attn_bools_to_dhariwal_attn_res((False, False, True, False))
    },
    "song_unet": {
        "img_resolution": 32,
        "in_channels": 3,
        "out_channels": 3,
        "model_channels": 128,
	"num_blocks": 4,
	"channel_mult_noise": 2,
        "resample_filter": [1, 3, 3, 1],
        "channel_mult": [2, 2, 2],
        "encoder_type": "standard",
        "decoder_type": "standard",
    },
    "edm2_unet": {
        "img_resolution": 32,
        "img_channels": 3,
        "label_dim": 0,
        "model_channels": 128,
        "channel_mult": [2, 2, 2],
        "num_blocks": 4,
        "attn_resolutions": [16],
    },
}


def instantiate_model(args) -> nn.Module:
    
    configs = MODEL_CONFIGS[args.arch_name]
    print("using model config:", configs)
    configs['dropout'] = args.dropout
    arch = MODEL_ARCHS[args.arch_name]
    
    #if args.use_edm_aug:
    #    #configs['augment_dim'] = 6
    #    #print("using edm aug!")

    if args.arch_name == 'edm2_unet':
        print("USING EDM 2")


    if args.which == 'meanflow':
        print("USING MEANFLOW")
        model = MeanFlow(arch=arch, net_configs=configs, args=args)
    elif args.which == 'lagrangian':
        print("USING LAGRANGIAN")
        model = Lagrangian(arch=arch, net_configs=configs, args=args)
    elif args.which == 'eulerian':
        print("USING EULERIAN")
        model = Eulerian(arch=arch, net_configs=configs, args=args)
    elif args.which == 'flowmatching':
        model = FlowMatching(arch=arch, net_configs=configs, args=args)
    else:
        assert False

    return model
