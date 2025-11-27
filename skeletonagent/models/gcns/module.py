import copy as cp
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from ...utils import Graph, cache_checkpoint
from ..builder import BACKBONES
from .utils import gcn_unit, mstcn_unit, base_tcn
import os
from einops import rearrange
from ..clip import clip
import numpy as np

EPS = 1e-4


class GCN_Block(nn.Module):

    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, **kwargs):
        super().__init__()
        common_args = ['act', 'norm', 'g1x1']
        for arg in common_args:
            if arg in kwargs:
                value = kwargs.pop(arg)
                kwargs['tcn_' + arg] = value
                kwargs['gcn_' + arg] = value

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[1:4] != 'cn_'}
        assert len(kwargs) == 0

        self.gcn = gcn_unit(in_channels, out_channels, A, **gcn_kwargs)
        self.tcn = mstcn_unit(out_channels, out_channels, stride=stride, **tcn_kwargs)

        self.relu = nn.ReLU()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = base_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model.float()

    def forward(self,text):
        return self.model.encode_text(text)


@BACKBONES.register_module()
class GCN_Module_LLM(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=96,
                 ch_ratio=2,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 data_bn_type='MVC',
                 num_person=2,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.llm_model = kwargs.pop('llm_model', 'gpt4o')
        self.llm_modality = kwargs.pop('llm_modality', 'j')
        self.num_classes = kwargs.pop('num_classes', 60)
        self.kwargs = kwargs
        
        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * base_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(base_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)
        lw_kwargs[0].pop('g1x1', None)
        lw_kwargs[0].pop('gcn_g1x1', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        
        if self.num_classes == 60:
            dataset_name = 'nturgbd_60'
        elif self.num_classes == 120:
            dataset_name = 'nturgbd_120'
        elif self.num_classes == 99:
            dataset_name = 'finegym'
        elif self.num_classes == 400:
            dataset_name = 'k400'
        elif self.num_classes == 155:
            dataset_name = 'uav_human'
        
        if self.llm_modality == 'bm':
            self.llm_modality = 'b'
        elif self.llm_modality == 'jm':
            self.llm_modality = 'j'
        elif self.llm_modality == 'km':
            self.llm_modality = 'k'
        
        """
        *******************************
        """
        examplar_name = f"{dataset_name}_{self.llm_modality}_critical_joints.npy"
        examplar_file = np.load(os.path.join(os.path.dirname(__file__), 'cls_matrix', examplar_name))
        # Class V V
        self.examplar = nn.Parameter(torch.Tensor(examplar_file), requires_grad=False)
        self.num_point = A.size(1)
        self.to_joint_embedding = nn.Linear(in_channels, base_channels)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, base_channels))
        self.direct_fc = nn.Conv2d(base_channels * 4, 1, 1, 1)
        
        self.clip_model = 'ViT-B/32'
        self.linear_head = nn.ModuleDict()
        self.linear_head[self.clip_model] = nn.Linear(384,512)
        self.conv_init(self.linear_head[self.clip_model])
        self.logit_scale = nn.Parameter(torch.ones(1,1) * np.log(1 / 0.07))
        self.model_text_dict = nn.ModuleDict()
        model_, preprocess = clip.load(self.clip_model, 'cuda')
        del model_.visual
        model_text = TextCLIP(model_).cuda()
        self.model_text_dict[self.clip_model] = model_text
        """
        **********************************
        """
        
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        modules = []

        inflate_times = 0
        down_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(GCN_Block(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))
            down_times += (i in down_stages)

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained

    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)
    
    def conv_init(self, conv):
        if conv.weight is not None:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def forward(self, x):
        N, M, T, V, C = x.size()
        
        # N M T V C -> N C T V M
        x = x.permute(0, 4, 2, 3, 1).contiguous()
        x = rearrange(x, 'n c t v m -> (n m t) v c', m=M, v=V).contiguous()
        x = self.to_joint_embedding(x)
        _, _, C = x.size()
        # N*M*T V C = N*M*T V C + 1 V C
        x += self.pos_embedding[:, :self.num_point]
        x = rearrange(x, '(n m t) v c -> n (m v c) t', m=M, t=T).contiguous()
        
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        for i in range(self.num_stages-1):
            x = self.gcn[i](x)
        
        
        """
        *******************************
        *** Local Joint Constraints ***
        *******************************
        """
        # N*M C T V -> N*M C V
        x_1 = x.mean(2)
        # N*M C Class V  =  N*M C V  *  Class V V
        x_1 = torch.einsum('nmv,cvu->nmcu', x_1, self.examplar)
        # N*M C Class V -> N*M 1 Class V -> N*M Class V -> N*M Class
        x_1 = self.direct_fc(x_1).squeeze(1).mean(2)
        # N*M Class -> N M Class -> N Class
        x_1 = x_1.reshape(N, M, self.num_classes)
        x_1 = x_1.mean(dim=1)
        
        """
        **********************************
        *** Global Semantics Alignment ***
        **********************************
        """
        # N*M C T V
        c_new = x.size(1)
        x_2 = x.view(N, M, c_new, -1)
        x_2 = x_2.mean(3).mean(1)
        x_2 = self.linear_head[self.clip_model](x_2)
        
        
        x = x.reshape((N, M) + x.shape[1:])
        
        return x, x_1, x_2, self.model_text_dict, self.logit_scale


@BACKBONES.register_module()
class GCN_Module_LLM_K400(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=96,
                 ch_ratio=2,
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 data_bn_type='VC',
                 num_person=2,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.llm_model = kwargs.pop('llm_model', 'gpt4o')
        self.llm_modality = kwargs.pop('llm_modality', 'j')
        self.num_classes = kwargs.pop('num_classes', 60)
        self.kwargs = kwargs
        
        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)
        lw_kwargs[0].pop('g1x1', None)
        lw_kwargs[0].pop('gcn_g1x1', None)

        self.in_channels = in_channels
        self.base_channels = base_channels

        if self.num_classes == 60:
            dataset_name = 'nturgbd_60'
        elif self.num_classes == 120:
            dataset_name = 'nturgbd_120'
        elif self.num_classes == 99:
            dataset_name = 'finegym'
        elif self.num_classes == 400:
            dataset_name = 'k400'
        elif self.num_classes == 155:
            dataset_name = 'uav_human'
        
        if self.llm_modality == 'bm':
            self.llm_modality = 'b'
        elif self.llm_modality == 'jm':
            self.llm_modality = 'j'
        elif self.llm_modality == 'km':
            self.llm_modality = 'k'
        
        """
        *******************************
        """
        examplar_name = f"{dataset_name}_{self.llm_modality}_critical_joints.npy"
        examplar_file = np.load(os.path.join(os.path.dirname(__file__), 'cls_matrix', examplar_name))
        # Class V V
        self.examplar = nn.Parameter(torch.Tensor(examplar_file), requires_grad=False)
        self.direct_fc = nn.Conv2d(base_channels * 4, 1, 1, 1)        
        
        self.clip_model = 'ViT-B/32'
        self.linear_head = nn.ModuleDict()
        self.linear_head[self.clip_model] = nn.Linear(384,512)
        self.conv_init(self.linear_head[self.clip_model])
        self.logit_scale = nn.Parameter(torch.ones(1,1) * np.log(1 / 0.07))
        self.model_text_dict = nn.ModuleDict()
        model_, preprocess = clip.load(self.clip_model, 'cuda')
        del model_.visual
        model_text = TextCLIP(model_).cuda()
        self.model_text_dict[self.clip_model] = model_text
        """
        **********************************
        """

        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        modules = []
        if self.in_channels != self.base_channels:
            modules = [GCN_Block(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        down_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(GCN_Block(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))
            down_times += (i in down_stages)

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained

    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)
    
    def conv_init(self, conv):
        if conv.weight is not None:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)
        
        
        """
        *******************************
        *** Local Joint Constraints ***
        *******************************
        """
        # N*M C T V -> N*M C V
        x_1 = x.mean(2)
        # N*M C Class V  =  N*M C V  *  Class V V
        x_1 = torch.einsum('nmv,cvu->nmcu', x_1, self.examplar)
        # N*M C Class V -> N*M 1 Class V -> N*M Class V -> N*M Class
        x_1 = self.direct_fc(x_1).squeeze(1).mean(2)
        # N*M Class -> N M Class -> N Class
        x_1 = x_1.reshape(N, M, self.num_classes)
        x_1 = x_1.mean(dim=1)
        
        """
        **********************************
        *** Global Semantics Alignment ***
        **********************************
        """
        # N*M C T V
        c_new = x.size(1)
        x_2 = x.view(N, M, c_new, -1)
        x_2 = x_2.mean(3).mean(1)
        x_2 = self.linear_head[self.clip_model](x_2)
        

        x = x.reshape((N, M) + x.shape[1:])
        
        return x, x_1, x_2, self.model_text_dict, self.logit_scale
    