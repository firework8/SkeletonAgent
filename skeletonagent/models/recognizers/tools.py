import numpy as np
import torch
import torch.nn.functional as F
import random


"""
NTU Mix
"""

trunk_ori_index = [4, 3, 21, 2, 1]
left_hand_ori_index = [9, 10, 11, 12, 24, 25]
right_hand_ori_index = [5, 6, 7, 8, 22, 23]
left_leg_ori_index = [17, 18, 19, 20]
right_leg_ori_index = [13, 14, 15, 16]

trunk = [i - 1 for i in trunk_ori_index]
left_hand = [i - 1 for i in left_hand_ori_index]
right_hand = [i - 1 for i in right_hand_ori_index]
left_leg = [i - 1 for i in left_leg_ori_index]
right_leg = [i - 1 for i in right_leg_ori_index]
body_parts = [trunk, left_hand, right_hand, left_leg, right_leg]

@torch.no_grad()
def ske_swap_randscale(x, spa_l, spa_u, tem_l, tem_u, p=None):
    '''
    swap a batch skeleton
    T   100 --> 50 --> 25    
    S   25  --> 25 --> 25 (5 parts)
    '''
    # N M T V C -> N C T V M
    x = x.permute(0, 4, 2, 3, 1).contiguous()
    N, C, T, V, M = x.size()
    tem_downsample_ratio = 4

    # generate swap idx
    idx = torch.arange(N)
    if N > 2:
        n = torch.randint(1, N - 1, (1,))
        randidx = (idx + n) % N
    else:
        randidx = idx

    # ------ Spatial ------ #
    Cs = random.randint(spa_l, spa_u)
    # sample the parts index
    parts_idx = random.sample(body_parts, Cs)
    # generate spa_idx
    spa_idx = []
    for part_idx in parts_idx:
        spa_idx += part_idx
    spa_idx.sort()

    # ------ Temporal ------ #
    Ct = random.randint(tem_l, tem_u)
    tem_idx = random.randint(0, T // tem_downsample_ratio - Ct)
    rt = Ct * tem_downsample_ratio

    xst = x.clone()
    if p==None:
        p = random.random()
    if p > 0.25:
        N, C, T, V, M = xst.size()

        Ct_2 = random.randint(Ct, 25)
        tem_idx_2 = random.randint(0, T // tem_downsample_ratio - Ct_2)
        rt_2 = Ct_2 * tem_downsample_ratio

        xst_temp = xst[:, :, tem_idx_2 * tem_downsample_ratio: tem_idx_2 * tem_downsample_ratio + rt_2]

        xst_temp = xst_temp.permute(0, 4, 3, 1, 2).contiguous()
        xst_temp = xst_temp.view(N * M, V * C, -1)
        xst_temp = torch.nn.functional.interpolate(xst_temp, size=rt)
        xst_temp = xst_temp.view(N, M, V, C, rt)
        xst_temp = xst_temp.permute(0, 3, 4, 2, 1).contiguous()
        xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
                xst_temp[randidx][:, :, :, spa_idx, :]
        mask = torch.zeros(T // tem_downsample_ratio, V)
        mask[tem_idx:tem_idx + Ct, spa_idx] = 1
    else:
        lamb = random.random()
        xst = xst * (1 - lamb) + xst[randidx] * lamb
        mask = torch.zeros(T // tem_downsample_ratio, V) + lamb

    # N C T V M -> N M T V C
    xst = xst.permute(0, 4, 2, 3, 1).contiguous()
    
    return randidx, xst, mask.mean()


"""
K400 Mix
"""

trunk_ori_index_k400 = [1, 2, 3, 4, 5, 18, 19, 20]
left_hand_ori_index_k400 = [6, 8, 10]
right_hand_ori_index_k400 = [7, 9, 11]
left_leg_ori_index_k400 = [12, 14, 16]
right_leg_ori_index_k400 = [13, 15, 17]

trunk_k400 = [i - 1 for i in trunk_ori_index_k400]
left_hand_k400 = [i - 1 for i in left_hand_ori_index_k400]
right_hand_k400 = [i - 1 for i in right_hand_ori_index_k400]
left_leg_k400 = [i - 1 for i in left_leg_ori_index_k400]
right_leg_k400 = [i - 1 for i in right_leg_ori_index_k400]
body_parts_k400 = [trunk_k400, left_hand_k400, right_hand_k400, left_leg_k400, right_leg_k400]

@torch.no_grad()
def ske_swap_randscale_k400(x, spa_l, spa_u, tem_l, tem_u, p=None):
    '''
    swap a batch skeleton
    T   100 --> 50 --> 25
    S   25  --> 25 --> 25 (5 parts)
    '''
    # N M T V C -> N C T V M
    x = x.permute(0, 4, 2, 3, 1).contiguous()
    N, C, T, V, M = x.size()
    tem_downsample_ratio = 4

    # generate swap idx
    idx = torch.arange(N)
    if N > 2:
        n = torch.randint(1, N - 1, (1,))
        randidx = (idx + n) % N
    else:
        randidx = idx

    # ------ Spatial ------ #
    Cs = random.randint(spa_l, spa_u)
    # sample the parts index
    parts_idx = random.sample(body_parts_k400, Cs)
    # generate spa_idx
    spa_idx = []
    for part_idx in parts_idx:
        spa_idx += part_idx
    spa_idx.sort()

    # ------ Temporal ------ #
    Ct = random.randint(tem_l, tem_u)
    tem_idx = random.randint(0, T // tem_downsample_ratio - Ct)
    rt = Ct * tem_downsample_ratio

    xst = x.clone()
    if p==None:
        p = random.random()
    if p > 0.25:
        N, C, T, V, M = xst.size()

        Ct_2 = random.randint(Ct, 25)
        tem_idx_2 = random.randint(0, T // tem_downsample_ratio - Ct_2)
        rt_2 = Ct_2 * tem_downsample_ratio

        xst_temp = xst[:, :, tem_idx_2 * tem_downsample_ratio: tem_idx_2 * tem_downsample_ratio + rt_2]

        xst_temp = xst_temp.permute(0, 4, 3, 1, 2).contiguous()
        xst_temp = xst_temp.view(N * M, V * C, -1)
        xst_temp = torch.nn.functional.interpolate(xst_temp, size=rt)
        xst_temp = xst_temp.view(N, M, V, C, rt)
        xst_temp = xst_temp.permute(0, 3, 4, 2, 1).contiguous()
        xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
                xst_temp[randidx][:, :, :, spa_idx, :]
        mask = torch.zeros(T // tem_downsample_ratio, V)
        mask[tem_idx:tem_idx + Ct, spa_idx] = 1
    else:
        lamb = random.random()
        xst = xst * (1 - lamb) + xst[randidx] * lamb
        mask = torch.zeros(T // tem_downsample_ratio, V) + lamb

    # N C T V M -> N M T V C
    xst = xst.permute(0, 4, 2, 3, 1).contiguous()
    
    return randidx, xst, mask.mean()
    

"""
UAV Mix
"""

trunk_ori_index_uav = [1, 14, 15, 16, 17]
left_hand_ori_index_uav = [2, 3, 4]
right_hand_ori_index_uav = [5, 6, 7]
left_leg_ori_index_uav = [8, 9, 10]
right_leg_ori_index_uav = [11, 12, 13]

trunk_uav = [i - 1 for i in trunk_ori_index_uav]
left_hand_uav = [i - 1 for i in left_hand_ori_index_uav]
right_hand_uav = [i - 1 for i in right_hand_ori_index_uav]
left_leg_uav = [i - 1 for i in left_leg_ori_index_uav]
right_leg_uav = [i - 1 for i in right_leg_ori_index_uav]
body_parts_uav = [trunk_uav, left_hand_uav, right_hand_uav, left_leg_uav, right_leg_uav]

@torch.no_grad()
def ske_swap_randscale_uav(x, spa_l, spa_u, tem_l, tem_u, p=None):
    '''
    swap a batch skeleton
    T   100 --> 50 --> 25
    S   25  --> 25 --> 25 (5 parts)
    '''
    # N M T V C -> N C T V M
    x = x.permute(0, 4, 2, 3, 1).contiguous()
    N, C, T, V, M = x.size()
    tem_downsample_ratio = 4
    
    # generate swap idx
    idx = torch.arange(N)
    if N > 2:
        n = torch.randint(1, N - 1, (1,))
        randidx = (idx + n) % N
    else:
        randidx = idx

    # ------ Spatial ------ #
    Cs = random.randint(spa_l, spa_u)
    # sample the parts index
    parts_idx = random.sample(body_parts_uav, Cs)
    # generate spa_idx
    spa_idx = []
    for part_idx in parts_idx:
        spa_idx += part_idx
    spa_idx.sort()

    # ------ Temporal ------ #
    Ct = random.randint(tem_l, tem_u)
    tem_idx = random.randint(0, T // tem_downsample_ratio - Ct)
    rt = Ct * tem_downsample_ratio

    xst = x.clone()
    if p==None:
        p = random.random()
    if p > 0.25:
        N, C, T, V, M = xst.size()

        Ct_2 = random.randint(Ct, 25)
        tem_idx_2 = random.randint(0, T // tem_downsample_ratio - Ct_2)
        rt_2 = Ct_2 * tem_downsample_ratio

        xst_temp = xst[:, :, tem_idx_2 * tem_downsample_ratio: tem_idx_2 * tem_downsample_ratio + rt_2]

        xst_temp = xst_temp.permute(0, 4, 3, 1, 2).contiguous()
        xst_temp = xst_temp.view(N * M, V * C, -1)
        xst_temp = torch.nn.functional.interpolate(xst_temp, size=rt)
        xst_temp = xst_temp.view(N, M, V, C, rt)
        xst_temp = xst_temp.permute(0, 3, 4, 2, 1).contiguous()
        xst[:, :, tem_idx * tem_downsample_ratio: tem_idx * tem_downsample_ratio + rt, spa_idx, :] = \
                xst_temp[randidx][:, :, :, spa_idx, :]
        mask = torch.zeros(T // tem_downsample_ratio, V)
        mask[tem_idx:tem_idx + Ct, spa_idx] = 1
    else:
        lamb = random.random()
        xst = xst * (1 - lamb) + xst[randidx] * lamb
        mask = torch.zeros(T // tem_downsample_ratio, V) + lamb

    # N C T V M -> N M T V C
    xst = xst.permute(0, 4, 2, 3, 1).contiguous()
    
    return randidx, xst, mask.mean()


def skeleton_mix(keypoint, joint_cfg, num_classes, label):
    
    if joint_cfg == 'nturgb+d':
        randidx, mix_data, lamb = ske_swap_randscale(keypoint, 2, 3, 13, 20)
    elif joint_cfg == 'coco_new':
        randidx, mix_data, lamb = ske_swap_randscale_k400(keypoint, 2, 3, 13, 20)
    elif joint_cfg == 'uav':
        randidx, mix_data, lamb = ske_swap_randscale_uav(keypoint, 2, 3, 13, 20)
    
    label_onehot = torch.zeros((keypoint.shape[0], num_classes)).cuda()
    label_onehot.scatter_(1, label.unsqueeze(-1), 1)
    mix_label = label_onehot * (1 - lamb) + label_onehot[randidx] * lamb
    
    return mix_data, mix_label


def mix_chunk(cls_score, mix_label, x_1, x_2):
    
    cls_score, mix_output = torch.chunk(cls_score, 2, dim=0)
    mix_output = F.log_softmax(mix_output, dim=1)
    cls_mix = -torch.mean(torch.sum(mix_output * mix_label, dim=1))
    x_1, _ = torch.chunk(x_1, 2, dim=0)
    x_2, _ = torch.chunk(x_2, 2, dim=0)
    
    return cls_score, cls_mix, x_1, x_2
