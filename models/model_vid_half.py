import math, time
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
from base import BaseModel
from utils.losses import *
from models.encoder import Encoder
from models.modeling.deeplab import DeepLab as DeepLab_v3p
# from models.modeling.hrnet import HRNet_W48, HRNet_W48_OCR_B, classifier_OCR
# from models.modeling.net_utils import NLM, NLM_dot, NLM_woSoft, NLM_NC_woSoft, Batch_Contrastive
import numpy as np
import cv2
import kornia
from models.feature_memory import *
from memory_profiler import profile
from utils.module_helper import ModuleHelper

EPS = 1e-20


class VCL(BaseModel):
    def __init__(self, num_classes, conf, sup_loss=None, ignore_index=None, testing=False, pretrained=True):

        super(VCL, self).__init__()
        assert int(conf['supervised']) + int(conf['semi']) == 1, 'one mode only'
        if conf['supervised']:
            self.mode = 'supervised'
        elif conf['semi']:
            self.mode = 'semi'
        else:
            raise ValueError('No such mode choice {}'.format(self.mode))

        self.ignore_index = ignore_index

        self.num_classes = num_classes
        self.sup_loss_w = conf['supervised_w']
        self.sup_loss = sup_loss
        self.downsample = conf['downsample']
        self.backbone = conf['backbone']
        self.layers = conf['layers']
        self.out_dim = conf['out_dim']
        self.proj_final_dim = conf['proj_final_dim']
        self._xent_targets = dict()
        self.edgedrop_rate = 0.1
        self.temperature = 0.07
        self.base_temperature = 0.07
        self.ignore_mask = -100
        self.xent = nn.CrossEntropyLoss(ignore_index=self.ignore_mask, reduction="none")
        self.queue_len = conf['queue_len']
        self.max_samples = conf['max_samples']
        # self.gpu_mem = open("/home/zhengyu/ours_video/gpu_memory.txt", 'w+')
        self.register_buffer("segment_queue", torch.randn(num_classes, self.queue_len, self.proj_final_dim))
        self.segment_queue = F.normalize(self.segment_queue, p=2, dim=2)
        self.register_buffer("segment_queue_ptr", torch.zeros(num_classes, dtype=torch.long))
        # feature_memory = FeatureMemory(num_samples=labeled_samples, dataset=dataset, memory_per_class=256,
        #                                feature_size=256, n_classes=num_classes)
        # self.segment_queue =
        self.pos_sample_num = 500
        self.neg_sample_num = 1000
        assert self.layers in [50, 101]

        if self.backbone == 'deeplab_v3+':
            self.encoder = DeepLab_v3p(backbone='resnet{}'.format(self.layers))
            self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
            for m in self.classifier.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.SyncBatchNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif self.backbone == 'psp':
            self.encoder = Encoder(pretrained=pretrained)
            self.classifier = nn.Conv2d(self.out_dim, num_classes, kernel_size=1, stride=1)
        # elif self.backbone == 'hrnet48':
        #     self.encoder = HRNet_W48(arch='hrnet48', num_classes=num_classes)
        #     self.classifier = nn.Conv2d(self.out_dim, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        # elif self.backbone == 'hrnet48_ocr':
        #     self.encoder = HRNet_W48(arch='hrnet48', num_classes=num_classes)
        #     self.classifier = classifier_OCR(num_classes)
        else:
            raise ValueError("No such backbone {}".format(self.backbone))

        if self.mode == 'semi':
            if self.backbone == 'hrnet48' or 'hrnet48_ocr':
                self.project = nn.Sequential(
                    nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1),
                    ModuleHelper.BNReLU(self.out_dim, bn_type='torchsyncbn'),
                    nn.Conv2d(self.out_dim, self.proj_final_dim, kernel_size=1)
                )
            else:
                self.project = nn.Sequential(
                    nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, stride=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.out_dim, self.proj_final_dim, kernel_size=1, stride=1)
                )
            # self.nlm = NLM_woSoft()
            self.weight_unsup = conf['weight_unsup']
            self.weight_cycle = conf['weight_cycle']
            self.weight_inter = conf['weight_inter']
            self.temp = conf['temp']
            self.epoch_start_unsup = conf['epoch_start_unsup']
            self.epoch_start_cycle = conf['epoch_start_cycle']
            self.selected_num = conf['selected_num']
            self.step_save = conf['step_save']
            self.step_count = 0
            self.feature_bank = []
            self.pseudo_label_bank = []
            self.pos_thresh_value = conf['pos_thresh_value']
            self.stride = conf['stride']

    def affinity(self, x1, x2):
        in_t_dim = x1.ndim
        if in_t_dim < 4:  # add in time dimension if not there
            x1, x2 = x1.unsqueeze(-2), x2.unsqueeze(-2)
        A = torch.einsum('btnc,btmc->btnm', x1, x2)
        # if self.restrict is not None:
        #     A = self.restrict(A)
        return A.squeeze(1) if in_t_dim < 4 else A

    def stoch_mat(self, A, zero_diagonal=False, do_dropout=True, do_sinkhorn=False):
        ''' Affinity -> Stochastic Matrix '''
        if zero_diagonal:
            A = self.zeroout_diag(A)

        if do_dropout and self.edgedrop_rate > 0:
            A = A.float()
            A[torch.rand_like(A) < self.edgedrop_rate] = -1e20

        return F.softmax(A / self.temperature, dim=-1)

    # @profile(precision=4, stream=open('/home/zhengyu/ours_video/memory/memory_modelforward.log', 'w+'))
    def forward(self, x_l=None, target_l=None, x_ul=None, target_ul=None, c_f=None, c_b=None, curr_iter=None,
                epoch=None, gpu=None, gt_l=None, ul1=None, br1=None, \
                ul2=None, br2=None, flip=None, theta=None, ):
        if not self.training:
            enc = self.encoder(x_l)
            enc = self.classifier(enc)
            return F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)

        if self.mode == 'supervised':
            enc = self.encoder(x_l)
            enc = self.classifier(enc)
            output_l = F.interpolate(enc, size=x_l.size()[2:], mode='bilinear', align_corners=True)

            loss_sup = self.sup_loss(output_l, target_l, ignore_index=self.ignore_index,
                                     temperature=1.0) * self.sup_loss_w

            curr_losses = {'loss_sup': loss_sup}
            outputs = {'sup_pred': output_l}
            total_loss = loss_sup
            return total_loss, curr_losses, outputs

        elif self.mode == 'semi':
            # supervised
            enc = self.encoder(x_l)  # 4*256*80*80
            enc_logit = self.classifier(enc)  # 4*17*80*80
            output_l = F.interpolate(enc_logit, size=x_l.size()[2:], mode='bilinear', align_corners=True)
            loss_sup = self.sup_loss(output_l, target_l, ignore_index=self.ignore_index,
                                     temperature=1.0) * self.sup_loss_w

            curr_losses = {'loss_sup': loss_sup}
            outputs = {'sup_pred': output_l}
            total_loss = loss_sup
            # if epoch < self.epoch_start_cycle:
            #     return total_loss, curr_losses, outputs
            #####################Memory Construction############################################
            # print("1:{}".format(torch.cuda.memory_allocated(0)), file=self.gpu_mem)
            # model.eval()
            # proj_labeled_features_correct = model.projection_head(labeled_features_correct)
            # model.train()
            # with torch.no_grad():
            if epoch < 5:
                with torch.no_grad():
                    enc_l = self.project(enc.detach())
            else:
                enc_l = self.project(enc)
            enc_l = F.normalize(enc_l, 2, 1)
            key = enc_l.detach()  # enc.detach()
            lb_key = target_l.detach()
            pred_key = enc_logit.detach()
            self._dequeue_and_enqueue(key, lb_key, pred_key,
                                      segment_queue=self.segment_queue,
                                      segment_queue_ptr=self.segment_queue_ptr)
            if epoch < 5:
                return total_loss, curr_losses, outputs
            # print("2:{}".format(torch.cuda.memory_allocated(0)), file=self.gpu_mem)
            # ##################### intra-video loss #############################################
            B, T, C, h, w = c_f.shape
            # f_clip=c_f[batch_idx, :, :, :, :]
            # b_clip = c_b[batch_idx, :, :, :, :]
            f_encs = self.encoder(c_f.flatten(0, 1))  # B*T,256,80,80
            b_encs = self.encoder(c_b.flatten(0, 1))  # (B*T*256)*80*80

            # f_encs_up = []
            # for fid in range(B): f_encs_up.append(f_encs[fid*T].deepcopy())
            if self.downsample:
                f_encs = F.avg_pool2d(f_encs, kernel_size=2, stride=2)
                b_encs = F.avg_pool2d(b_encs, kernel_size=2, stride=2)
            f_encs_out = self.project(f_encs)  # [b, c, h, w] #B*T,128,80,80
            f_encs_out = F.normalize(f_encs_out, 2, 1)

            f1_encs = f_encs_out[1].unsqueeze(0)
            b, c, fh, fw = f_encs_out.shape
            for i in range(1, B): f1_encs = torch.cat((f1_encs, f_encs_out[5 * i].unsqueeze(0)), 0)
            f1_encs_warp = kornia.geometry.transform.warp_affine(f1_encs, theta, [fw, fh], mode='bilinear')
            f_encs_new = f_encs_out.clone()
            for i in range(0, B): f_encs_new[5 * i] = f1_encs_warp[i]
            f_encs_new_flatten = f_encs_new.flatten(-2, -1).view(B, T, f_encs_new.size(1), -1).permute(0, 1, 3, 2)

            b_encs_out = self.project(b_encs)  # [b, c, h, w]
            b_encs_out = F.normalize(b_encs_out, 2, 1)
            b_encs_flatten = b_encs_out.flatten(-2, -1).view(B, T, b_encs_out.size(1), -1).permute(0, 1, 3,
                                                                                                   2)  # 4*5*6400*256
            b_encs_flip_flatten = torch.flip(b_encs_flatten, dims=[1])

            new_seq = torch.cat((f_encs_new_flatten, b_encs_flip_flatten), dim=1)  # 4*10*6400*256

            # Compute walks
            M = torch.ones(B, 1, fh, fw).cuda()
            mask_prob = kornia.geometry.transform.warp_affine(M, theta, [fw, fh], mode='bilinear')
            ignore_ind = (mask_prob <= 0.5).flatten()  # .type(torch.float)

            walks = dict()
            A_fb = self.affinity(new_seq[:, :-1], new_seq[:, 1:])  # affinity矩阵  前后帧
            A12s = [self.stoch_mat(A_fb[:, i], do_dropout=True) for i in range(2 * T - 1)]  # softmax归一化的随机矩阵A
            # A21s = [self.stoch_mat(A_fb[:, i].transpose(-1, -2), do_dropout=True) for i in range(2 * T - 1)]
            A_mm = self.affinity(f_encs_new_flatten[:, 1:], b_encs_flatten[:, 1:])  # 对应帧
            # A_mm = self.affinity(f_encs_new[:, 1:-1], b_encs[:, 1:-1]) #对应帧
            # A_mm = torch.cat((A_mm,A_fb[:,T-1].unsqueeze(1)),dim=1)
            A12m = [self.stoch_mat(A_mm[:, i], do_dropout=True) for i in range(T - 1)]

            AAs = []
            for i in range(1, T - 1):  #
                g = A12s[:i + 1]
                g.append(A12m[i])
                g.extend(A12s[-(i + 1):])  # list相加为cat，构成闭环 #print(a[::-1]) ### 取从后向前的元素
                aar = g[0].float()
                for _a in g[1:]:
                    aar = (aar @ _a).float()
                AAs.append((f"r{i}", aar))
            for i, aa in AAs:
                walks[f"cyc {i}"] = [aa, self.xent_targets(aa)]  # xent_targets为0到1599的数据并重复B次，一维tensor
                # Compute loss
                xents = [torch.tensor([0.]).cuda()]
                # diags = dict()

                for name, (A, targets) in walks.items():
                    logits = torch.log(A + EPS).flatten(0, -2)
                    target = targets.clone()
                    target[ignore_ind] = self.ignore_mask
                    loss = self.xent(logits, target).mean()
                    # acc = (torch.argmax(logit, dim=-1) == target).float().mean()
                    # diags.update({f"{h} xent {name}": loss.detach(),
                    #               f"{h} acc {name}": acc})
                    xents += [loss]
            loss_cycle = self.weight_cycle * sum(xents[1:]) / max(1, len(xents) - 1)
            curr_losses['loss_cycle'] = loss_cycle
            total_loss = total_loss + loss_cycle
            # if epoch < self.epoch_start_unsup:
            #     return total_loss, curr_losses, outputs

            ##################### unsup loss #############################################
            # x_ul: [batch_size, 2, 3, H, W]
            x_ul1 = x_ul[:, 0, :, :, :]
            x_ul2 = x_ul[:, 1, :, :, :]

            enc_ul1_up = self.encoder(x_ul1)
            # #if self.downsample:
            enc_ul1 = F.avg_pool2d(enc_ul1_up, kernel_size=2, stride=2)
            output_ul1 = self.project(enc_ul1)  # [b, c, h, w]
            output_ul1 = F.normalize(output_ul1, 2, 1)

            enc_ul2_up = self.encoder(x_ul2)

            # # if self.downsample:
            enc_ul2 = F.avg_pool2d(enc_ul2_up, kernel_size=2, stride=2)
            output_ul2 = self.project(enc_ul2)  # [b, c, h, w]
            output_ul2 = F.normalize(output_ul2, 2, 1)

            # compute pseudo label
            with torch.no_grad():
                logits1 = self.classifier(enc_ul1)  # [batch_size, num_classes, h, w]
                logits2 = self.classifier(enc_ul2)
                pseudo_logits_1 = F.softmax(logits1, 1).max(1)[0].detach()  # [batch_size, h, w]
                pseudo_logits_2 = F.softmax(logits2, 1).max(1)[0].detach()
                pseudo_label1 = logits1.max(1)[1].detach()  # [batch_size, h, w]
                pseudo_label2 = logits2.max(1)[1].detach()

            # get overlap part
            output_feature_list1 = []
            output_feature_list2 = []
            pseudo_label_list1 = []
            pseudo_label_list2 = []
            pseudo_logits_list1 = []
            pseudo_logits_list2 = []

            for idx in range(x_ul1.size(0)):
                output_ul1_idx = output_ul1[idx]
                output_ul2_idx = output_ul2[idx]

                pseudo_label1_idx = pseudo_label1[idx]
                pseudo_label2_idx = pseudo_label2[idx]
                pseudo_logits_1_idx = pseudo_logits_1[idx]
                pseudo_logits_2_idx = pseudo_logits_2[idx]

                if flip[0][idx] == True:
                    output_ul1_idx = torch.flip(output_ul1_idx, dims=(2,))
                    pseudo_label1_idx = torch.flip(pseudo_label1_idx, dims=(1,))
                    pseudo_logits_1_idx = torch.flip(pseudo_logits_1_idx, dims=(1,))

                if flip[1][idx] == True:
                    output_ul2_idx = torch.flip(output_ul2_idx, dims=(2,))
                    pseudo_label2_idx = torch.flip(pseudo_label2_idx, dims=(1,))
                    pseudo_logits_2_idx = torch.flip(pseudo_logits_2_idx, dims=(1,))

                # 因为原图320，特征缩小了8倍，80×80，所以对应的原图空间特征也要/8
                ul1_t, br1_t, ul2_t, br2_t = torch.stack(ul1, 0), torch.stack(br1, 0), torch.stack(ul2, 0), torch.stack(
                    br2, 0)
                ul1_f, br1_f, ul2_f, br2_f = ul1_t // 8, br1_t // 8, ul2_t // 8, br2_t // 8
                output_feature_list1.append(
                    output_ul1_idx[:, ul1_f[0, idx]:br1_f[0, idx], ul1_f[1, idx]:br1_f[1, idx]].permute(1, 2,
                                                                                                        0).contiguous().view(
                        -1, output_ul1.size(1)))
                output_feature_list2.append(
                    output_ul2_idx[:, ul2_f[0, idx]:br2_f[0, idx], ul2_f[1, idx]:br2_f[1, idx]].permute(1, 2,
                                                                                                        0).contiguous().view(
                        -1, output_ul2.size(1)))
                pseudo_label_list1.append(
                    pseudo_label1_idx[ul1_f[0, idx]:br1_f[0, idx], ul1_f[1, idx]:br1_f[1, idx]].contiguous().view(-1))
                pseudo_label_list2.append(
                    pseudo_label2_idx[ul2_f[0, idx]:br2_f[0, idx], ul2_f[1, idx]:br2_f[1, idx]].contiguous().view(-1))
                pseudo_logits_list1.append(
                    pseudo_logits_1_idx[ul1_f[0, idx]:br1_f[0, idx], ul1_f[1, idx]:br1_f[1, idx]].contiguous().view(-1))
                pseudo_logits_list2.append(
                    pseudo_logits_2_idx[ul2_f[0, idx]:br2_f[0, idx], ul2_f[1, idx]:br2_f[1, idx]].contiguous().view(-1))
            output_feat1 = torch.cat(output_feature_list1, 0)  # [n, c] 所有重叠区域像素特征集合
            output_feat2 = torch.cat(output_feature_list2, 0)  # [n, c]
            pseudo_label1_overlap = torch.cat(pseudo_label_list1, 0)  # [n,] #所有重叠区域像素伪标签集合
            pseudo_label2_overlap = torch.cat(pseudo_label_list2, 0)  # [n,]
            pseudo_logits1_overlap = torch.cat(pseudo_logits_list1, 0)  # [n,]
            pseudo_logits2_overlap = torch.cat(pseudo_logits_list2, 0)  # [n,]
            assert output_feat1.size(0) == output_feat2.size(0)
            assert pseudo_label1_overlap.size(0) == pseudo_label2_overlap.size(0)
            assert output_feat1.size(0) == pseudo_label1_overlap.size(0)

            # concat across multi-gpus
            b, c, h, w = output_ul1.size()
            selected_num = self.selected_num
            output_ul1_flatten = output_ul1.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
            output_ul2_flatten = output_ul2.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
            selected_idx1 = np.random.choice(range(b * h * w), selected_num, replace=False)
            selected_idx2 = np.random.choice(range(b * h * w), selected_num, replace=False)
            output_ul1_flatten_selected = output_ul1_flatten[selected_idx1]
            output_ul2_flatten_selected = output_ul2_flatten[selected_idx2]
            output_ul_flatten_selected = torch.cat([output_ul1_flatten_selected, output_ul2_flatten_selected],
                                                   0)  # [2*kk, c]
            output_ul_all = self.concat_all_gather(output_ul_flatten_selected)  # [2*N, c]

            pseudo_label1_flatten_selected = pseudo_label1.view(-1)[selected_idx1]
            pseudo_label2_flatten_selected = pseudo_label2.view(-1)[selected_idx2]
            pseudo_label_flatten_selected = torch.cat([pseudo_label1_flatten_selected, pseudo_label2_flatten_selected],
                                                      0)  # [2*kk]
            pseudo_label_all = self.concat_all_gather(pseudo_label_flatten_selected)  # [2*N]

            self.feature_bank.append(output_ul_all)
            self.pseudo_label_bank.append(pseudo_label_all)
            if self.step_count > self.step_save:
                self.feature_bank = self.feature_bank[1:]
                self.pseudo_label_bank = self.pseudo_label_bank[1:]
            else:
                self.step_count += 1
            output_ul_all = torch.cat(self.feature_bank, 0)
            pseudo_label_all = torch.cat(self.pseudo_label_bank, 0)

            eps = 1e-7
            pos1 = (output_feat1 * output_feat2.detach()).sum(-1,
                                                              keepdim=True) / self.temp  # [n, 1] 共3573个重叠点 不同view的相同点为pos
            pos2 = (output_feat1.detach() * output_feat2).sum(-1, keepdim=True) / self.temp  # [n, 1]

            # compute loss1
            b = 8000

            def run1(pos, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap, neg_max1):
                # print("gpu: {}, i_1: {}".format(gpu, i))
                mask1_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label1_overlap.unsqueeze(
                    -1)).half()  # [n, b] 找到3573个重叠点与view中所有点不同label的点作为neg
                neg1_idx = (output_feat1 @ output_ul_idx.T) / self.temp  # [n, b] #将自身点之外的所有点当做neg 求相似度
                logits1_neg_idx = (torch.exp(neg1_idx - neg_max1) * mask1_idx).sum(-1)  # [n, ]
                return logits1_neg_idx

            def run1_0(pos, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap):
                # print("gpu: {}, i_1_0: {}".format(gpu, i))
                mask1_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label1_overlap.unsqueeze(
                    -1)).half()  # [n, b] 找到3573个重叠点与view中所有点不同label的点作为neg
                neg1_idx = (output_feat1 @ output_ul_idx.T) / self.temp  # [n, b] #将自身点之外的所有点当做neg 求相似度
                neg1_idx = torch.cat([pos, neg1_idx], 1)  # [n, 1+b] 将pos与neg串起来
                mask1_idx = torch.cat([torch.ones(mask1_idx.size(0), 1).half().cuda(), mask1_idx], 1)  # [n, 1+b]
                neg_max1 = torch.max(neg1_idx, 1, keepdim=True)[0]  # [n, 1]
                logits1_neg_idx = (torch.exp(neg1_idx - neg_max1) * mask1_idx).sum(-1)  # [n, ]
                return logits1_neg_idx, neg_max1

            N = output_ul_all.size(0)
            logits1_down = torch.zeros(pos1.size(0)).cuda()
            for i in range((N - 1) // b + 1):
                # print("gpu: {}, i: {}".format(gpu, i))
                pseudo_label_idx = pseudo_label_all[i * b:(i + 1) * b]  # 每次8000个点
                output_ul_idx = output_ul_all[i * b:(i + 1) * b]
                if i == 0:
                    logits1_neg_idx, neg_max1 = torch.utils.checkpoint.checkpoint(run1_0, pos1, output_feat1,
                                                                                  output_ul_idx, pseudo_label_idx,
                                                                                  pseudo_label1_overlap)
                else:
                    logits1_neg_idx = torch.utils.checkpoint.checkpoint(run1, pos1, output_feat1, output_ul_idx,
                                                                        pseudo_label_idx, pseudo_label1_overlap,
                                                                        neg_max1)
                logits1_down += logits1_neg_idx

            logits1 = torch.exp(pos1 - neg_max1).squeeze(-1) / (logits1_down + eps)

            pos_mask_1 = ((pseudo_logits2_overlap > self.pos_thresh_value) & (
                        pseudo_logits1_overlap < pseudo_logits2_overlap)).half()
            loss1 = -torch.log(logits1 + eps)
            loss1 = (loss1 * pos_mask_1).sum() / (pos_mask_1.sum().float() + 1e-12)

            # compute loss2
            def run2(pos, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap, neg_max2):
                # print("gpu: {}, i_2: {}".format(gpu, i))
                mask2_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label2_overlap.unsqueeze(-1)).half()  # [n, b]
                neg2_idx = (output_feat2 @ output_ul_idx.T) / self.temp  # [n, b]
                logits2_neg_idx = (torch.exp(neg2_idx - neg_max2) * mask2_idx).sum(-1)  # [n, ]
                return logits2_neg_idx

            def run2_0(pos, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap):
                # print("gpu: {}, i_2_0: {}".format(gpu, i))
                mask2_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label2_overlap.unsqueeze(-1)).half()  # [n, b]
                neg2_idx = (output_feat2 @ output_ul_idx.T) / self.temp  # [n, b]
                neg2_idx = torch.cat([pos, neg2_idx], 1)  # [n, 1+b]
                mask2_idx = torch.cat([torch.ones(mask2_idx.size(0), 1).half().cuda(), mask2_idx], 1)  # [n, 1+b]
                neg_max2 = torch.max(neg2_idx, 1, keepdim=True)[0]  # [n, 1]
                logits2_neg_idx = (torch.exp(neg2_idx - neg_max2) * mask2_idx).sum(-1)  # [n, ]
                return logits2_neg_idx, neg_max2

            N = output_ul_all.size(0)
            logits2_down = torch.zeros(pos2.size(0)).cuda()
            for i in range((N - 1) // b + 1):
                pseudo_label_idx = pseudo_label_all[i * b:(i + 1) * b]
                output_ul_idx = output_ul_all[i * b:(i + 1) * b]
                if i == 0:
                    logits2_neg_idx, neg_max2 = torch.utils.checkpoint.checkpoint(run2_0, pos2, output_feat2,
                                                                                  output_ul_idx, pseudo_label_idx,
                                                                                  pseudo_label2_overlap)
                else:
                    logits2_neg_idx = torch.utils.checkpoint.checkpoint(run2, pos2, output_feat2, output_ul_idx,
                                                                        pseudo_label_idx, pseudo_label2_overlap,
                                                                        neg_max2)
                logits2_down += logits2_neg_idx

            logits2 = torch.exp(pos2 - neg_max2).squeeze(-1) / (logits2_down + eps)

            pos_mask_2 = ((pseudo_logits1_overlap > self.pos_thresh_value) & (
                        pseudo_logits2_overlap < pseudo_logits1_overlap)).half()

            loss2 = -torch.log(logits2 + eps)
            loss2 = (loss2 * pos_mask_2).sum() / (pos_mask_2.sum().float() + 1e-12)

            loss_unsup = self.weight_unsup * (loss1 + loss2)
            # curr_losses['loss1'] = loss1
            # curr_losses['loss2'] = loss2
            curr_losses['loss_unsup'] = loss_unsup
            total_loss = total_loss + loss_unsup
            # if epoch < self.epoch_start_inter:
            #     return total_loss, curr_losses, outputs
            # return total_loss, curr_losses, outputs
            ##################### inter-video loss #############################################

            output_ul1_up = self.project(enc_ul1_up)
            output_ul1_up = F.normalize(output_ul1_up, 2, 1)

            output_ul2_up = self.project(enc_ul2_up)
            output_ul2_up = F.normalize(output_ul2_up, 2, 1)

            # compute pseudo label
            with torch.no_grad():
                logits1_up = self.classifier(enc_ul1_up)  # [batch_size, num_classes, h, w]
                logits2_up = self.classifier(enc_ul2_up)
                pseudo_logits1_up = F.softmax(logits1_up, 1).max(1)[0].detach()  # [batch_size, h, w]
                pseudo_logits2_up = F.softmax(logits2_up, 1).max(1)[0].detach()
                pseudo_label1_up = logits1_up.max(1)[1].detach()  # [batch_size, h, w]
                pseudo_label2_up = logits2_up.max(1)[1].detach()

            enc_feature_list1_up = []
            enc_feature_list2_up = []
            pseudo_logits_list1_up = []
            pseudo_logits_list2_up = []
            pseudo_label_list1_up = []
            pseudo_label_list2_up = []
            for idx in range(x_ul1.size(0)):
                enc_ul1_idx_up = output_ul1_up[idx]
                enc_ul2_idx_up = output_ul2_up[idx]
                pseudo_label1_idx_up = pseudo_label1_up[idx]
                pseudo_label2_idx_up = pseudo_label2_up[idx]
                pseudo_logits_1_idx_up = pseudo_logits1_up[idx]
                pseudo_logits_2_idx_up = pseudo_logits2_up[idx]
                if flip[0][idx] == True:
                    enc_ul1_idx_up = torch.flip(enc_ul1_idx_up, dims=(2,))
                    pseudo_label1_idx_up = torch.flip(pseudo_label1_idx_up, dims=(1,))
                    pseudo_logits_1_idx_up = torch.flip(pseudo_logits_1_idx_up, dims=(1,))
                if flip[1][idx] == True:
                    enc_ul2_idx_up = torch.flip(enc_ul2_idx_up, dims=(2,))
                    pseudo_label2_idx_up = torch.flip(pseudo_label2_idx_up, dims=(1,))
                    pseudo_logits_2_idx_up = torch.flip(pseudo_logits_2_idx_up, dims=(1,))
                # 因为原图320，特征缩小了8倍，80×80，所以对应的原图空间特征也要/8
                ul1_t, br1_t, ul2_t, br2_t = torch.stack(ul1, 0), torch.stack(br1, 0), torch.stack(ul2, 0), torch.stack(
                    br2, 0)
                ul1_f_up, br1_f_up, ul2_f_up, br2_f_up = ul1_t // 4, br1_t // 4, ul2_t // 4, br2_t // 4
                enc_feature_list1_up.append(
                    enc_ul1_idx_up[:, ul1_f_up[0, idx]:br1_f_up[0, idx], ul1_f_up[1, idx]:br1_f_up[1, idx]].permute(1,
                                                                                                                    2,
                                                                                                                    0).contiguous().view(
                        -1, output_ul1_up.size(1)))
                enc_feature_list2_up.append(
                    enc_ul2_idx_up[:, ul2_f_up[0, idx]:br2_f_up[0, idx], ul2_f_up[1, idx]:br2_f_up[1, idx]].permute(1,
                                                                                                                    2,
                                                                                                                    0).contiguous().view(
                        -1, output_ul2_up.size(1)))
                pseudo_label_list1_up.append(pseudo_label1_idx_up[ul1_f_up[0, idx]:br1_f_up[0, idx],
                                             ul1_f_up[1, idx]:br1_f_up[1, idx]].contiguous().view(-1))
                pseudo_label_list2_up.append(pseudo_label2_idx_up[ul2_f_up[0, idx]:br2_f_up[0, idx],
                                             ul2_f_up[1, idx]:br2_f_up[1, idx]].contiguous().view(-1))
                pseudo_logits_list1_up.append(pseudo_logits_1_idx_up[ul1_f_up[0, idx]:br1_f_up[0, idx],
                                              ul1_f_up[1, idx]:br1_f_up[1, idx]].contiguous().view(-1))
                pseudo_logits_list2_up.append(pseudo_logits_2_idx_up[ul2_f_up[0, idx]:br2_f_up[0, idx],
                                              ul2_f_up[1, idx]:br2_f_up[1, idx]].contiguous().view(-1))

            feats_, labels_ = self._anchor_sampling(enc_l, pred_key, target_l, output_ul1_up, output_ul2_up,
                                                    pseudo_logits1_up, pseudo_logits2_up, pseudo_label1_up,
                                                    pseudo_label2_up,
                                                    enc_feature_list1_up, enc_feature_list2_up, pseudo_label_list1_up,
                                                    pseudo_label_list2_up,
                                                    pseudo_logits_list1_up, pseudo_logits_list2_up, ul1_f_up, ul2_f_up,
                                                    br1_f_up, br2_f_up, epoch)
            feats_norm = F.normalize(feats_, dim=-1)
            # loss
            loss_inter = self.weight_inter * self._contrastive(feats_norm, labels_.squeeze(0), queue=self.segment_queue)
            curr_losses['loss_inter'] = loss_inter
            total_loss = total_loss + loss_inter
            # if epoch < self.epoch_start_inter:
            return total_loss, curr_losses, outputs

        else:
            raise ValueError("No such mode {}".format(self.mode))

    def _contrastive(self, X_anchor, y_anchor, queue=None):
        anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[
            1]  # 22*46*256  anchor_num:batch中所有class的个数， n_view:每个class中sample的个数

        y_anchor = y_anchor.contiguous().view(-1, 1)  # 22
        anchor_count = n_view  # 每个class中sample的个数
        anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1),
                                   dim=0)  # unbind 从dim进行切片，并返回切片的结果，返回的结果里面没有dim这个维度 得到n_view个class_num*n_dim的tensor
        # 再进行cat，就是将所有sample的特征并了起来，结合起来相当于从每个class依次取一个sample，循环往复拼接了起来（22*46）*256
        # X_contrast, y_contrast = self._sample_negative(
        #     queue)  # 将19*1000*256的memory变成了19000*256的memopry，标签也是按顺序 y则为1000个0接1000个1接1000个2
        # y_contrast = y_contrast.contiguous().view(-1, 1)  # 19*1000
        # contrast_count = 1
        # contrast_feature = X_contrast  # memory的特征（19*1000）*256

        contrast_feature = queue.view(-1, queue.shape[
            -1])  # 将19*1000*256的memory变成了19000*256的memopry，标签也是按顺序 y则为1000个0接1000个1接1000个2
        y_contrast = torch.arange(self.num_classes).unsqueeze(1).repeat(1, self.queue_len)
        y_contrast = y_contrast.contiguous().view(-1, 1).cuda()  # 19000
        contrast_count = 1

        mask = torch.eq(y_anchor,
                        y_contrast.T).half()  # torch.eq对两个tensor逐元素比较 个位置的0 1 #把一横一竖的tensor比较得到N*N的tensor，即标签一致的位置为1 22*19000
        mask = mask.repeat(anchor_count,
                           contrast_count).cuda()  # 将22*19000扩展到（22*49）*19000，所有anchor和sample的对应关系，所有正样本，即所有标签一致的对
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                        self.temperature)  # （22*46）*256 * 256*（19*1000） = （22*46）*（19*1000） div除法除以temperature，每个anchor与memory中每个sample相乘的结果
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # 减去最大相似度 1012*19000

        anchor_size = mask.shape[0]
        # positive sample
        pos_indice = torch.ones(anchor_size, self.queue_len)
        for i in range(anchor_size): pos_indice[i] = torch.randperm(self.queue_len)
        pos_indice = pos_indice[:, :self.pos_sample_num].long().flatten()
        pos_rand_x = torch.arange(anchor_size).unsqueeze(1).repeat([1, self.pos_sample_num]).flatten()
        pos_mask_all = mask.nonzero()[:, 1].view(anchor_size, -1)
        pos_rand_y = pos_mask_all[pos_rand_x, pos_indice]
        pos_sample_mask = torch.zeros_like(mask)
        pos_sample_mask[pos_rand_x, pos_rand_y] = 1

        # negative sample
        neg_mask = 1 - mask  # 所有标签不一致的pair
        neg_indice = torch.ones(anchor_size, self.queue_len * (self.num_classes - 1))
        for i in range(anchor_size): neg_indice[i] = torch.randperm(self.queue_len * (self.num_classes - 1))
        neg_indice = neg_indice[:, :self.neg_sample_num].long().flatten()
        neg_rand_x = torch.arange(anchor_size).unsqueeze(1).repeat([1, self.neg_sample_num]).flatten()
        neg_mask_all = neg_mask.nonzero()[:, 1].view(anchor_size, -1)
        neg_rand_y = neg_mask_all[neg_rand_x, neg_indice]
        neg_sample_mask = torch.zeros_like(neg_mask)
        neg_sample_mask[neg_rand_x, neg_rand_y] = 1

        neg_logits = torch.exp(logits) * neg_sample_mask
        neg_logits = neg_logits.sum(1, keepdim=True)  # 每个anchor所有负样本的和（22*46）*1

        exp_logits = torch.exp(logits)  # 1012*19000

        log_prob = logits - torch.log(exp_logits + neg_logits)  # 1012*19000+1012*1 把每个sample都与neg的和相加，
        # 在下面的计算中与pos的mask一乘则能得到infonce的分母exp（pos）+sum（exp（neg）），因为infonce每一个pos都要与所有neg相加，而不是pos的和加neg的和

        mean_log_prob_pos = (pos_sample_mask * log_prob).sum(1) / pos_sample_mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def xent_targets(self, A):
        B, N = A.shape[:2]  # A=aar  B N N
        key = '%s:%sx%s' % (str(A.device), B, N)  # CUDA:0:4X1600

        if key not in self._xent_targets:
            I = torch.arange(A.shape[-1])[None].repeat(B, 1)
            # arrange生成0-1599的数字,[None]将他变成1*1600维的矩阵，再repeat变成B*1600的矩阵，每行都为0-1599的数字
            self._xent_targets[key] = I.view(-1).to(A.device)  # 0.....1599,0,....1599,0,...1599,0....1599

        return self._xent_targets[key]

    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        with torch.no_grad():
            tensors_gather = [torch.ones_like(tensor)
                              for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

            output = torch.cat(tensors_gather, dim=0)
        return output

    def get_backbone_params(self):
        return self.encoder.get_backbone_params()

    def get_other_params(self):
        if self.mode == 'supervised':
            return chain(self.encoder.get_module_params(), self.classifier.parameters())
        elif self.mode == 'semi':
            return chain(self.encoder.get_module_params(), self.classifier.parameters(), self.project.parameters())
        else:
            raise ValueError("No such mode {}".format(self.mode))

    def _dequeue_and_enqueue(self, keys, labels, preds,
                             segment_queue, segment_queue_ptr):
        with torch.no_grad():
            batch_size = keys.shape[0]
            feat_dim = keys.shape[1]
            class_num = preds.shape[1]

            # labels = labels[:, ::self.network_stride, ::self.network_stride]
            labels_down = F.interpolate(labels.half().unsqueeze(1),
                                        size=(keys.shape[2], keys.shape[3]),
                                        mode='nearest').squeeze(1)
            probs = torch.softmax(preds, dim=1)
            _, pred_labels = torch.max(probs, dim=1)
            for bs in range(batch_size):
                this_feat = keys[bs].contiguous().view(feat_dim, -1).T
                this_label = labels_down[bs].contiguous().view(-1)
                this_label_ids = torch.unique(this_label)
                this_label_ids = [x for x in this_label_ids if x < 255]
                this_preds = pred_labels[bs].contiguous().view(-1)
                this_probs = probs[bs].contiguous().view(class_num, -1)

                for lb in this_label_ids:
                    # idxs = (this_label == lb).nonzero()
                    lb = lb.long()
                    idxs_easy = ((this_label == lb).half() * (this_preds == lb).half()).nonzero().squeeze(-1)
                    new_feat = this_feat[idxs_easy, :]
                    # new_weight = torch.softmax(torch.cat([weight_easy, weight_hard]), dim=0)
                    feat = torch.mean(new_feat, dim=0)
                    ptr = int(segment_queue_ptr[lb])
                    K = idxs_easy.shape[0]
                    if ptr + K <= self.queue_len:
                        segment_queue[lb, ptr:ptr + K, :] = F.normalize(this_feat[idxs_easy, :], dim=1)
                        segment_queue_ptr[lb] = segment_queue_ptr[lb] + K
                    elif ptr < self.queue_len and ptr + K > self.queue_len:
                        permK = torch.randperm(K)
                        segment_queue[lb, ptr:, :] = F.normalize(this_feat[permK[:(self.queue_len - ptr)], :],
                                                                 dim=1)
                        segment_queue_ptr[lb] = self.queue_len
                    elif ptr == self.queue_len:
                        segment_queue[lb, :, :] = torch.cat(
                            [segment_queue[lb, 1:, :], F.normalize(feat.unsqueeze(0), dim=1)], 0)

    def _anchor_sampling(self, enc_l, pred_key, target_l, output_ul1_up, output_ul2_up, pseudo_logits1_up,
                         pseudo_logits2_up, pseudo_label1_up, pseudo_label2_up, enc_feature_list1_up,
                         enc_feature_list2_up, pseudo_label_list1_up, pseudo_label_list2_up,
                         pseudo_logits_list1_up, pseudo_logits_list2_up, ul1_f_up, ul2_f_up, br1_f_up, br2_f_up,
                         epoch):
        # prob_l = F.softmax(pred_key, 1).max(1)[0].detach()
        b, c, w, h = output_ul1_up.size()
        n_anchor = self.max_samples
        feats_ = torch.zeros(0, n_anchor, c).cuda()
        labels_ = torch.zeros(0).cuda()

        alpha_t = 20
        easy_thresh = 0.95
        hard_thresh = 0.85
        # with torch.no_grad():
        #     # prob = torch.softmax(enc_feature_list1_up, dim=1)
        #     entropy = -torch.sum(pseudo_logits_list1_up * torch.log(pseudo_logits_list1_up + 1e-10), dim=1)
        #     low_thresh = np.percentile(
        #         entropy.cpu().numpy().flatten(), alpha_t
        #     )

        # supervised anchor sample
        target_l_down = F.interpolate(target_l.half().unsqueeze(1), size=enc_l.size()[2:], mode='nearest').squeeze(
            1)
        for idx in range(b):
            this_feat_l = enc_l[idx].contiguous().view(enc_l.shape[1], -1).permute(1, 0)
            this_y_pred = pred_key[idx].max(0)[1].contiguous().view(-1)
            this_y_l = target_l_down[idx].contiguous().view(-1)
            this_classes_l = torch.unique(this_y_l)
            this_classes_l = [x for x in this_classes_l if x != self.ignore_index]
            this_classes_l = [x for x in this_classes_l if
                              (this_y_l == x).nonzero().shape[0] > n_anchor]  # 删除样本数太少的样本
            for cls_id in this_classes_l:
                hard_indices = ((this_y_l == cls_id) & (this_y_pred != cls_id)).nonzero().squeeze(
                    1)  # 预测与标签不一致的anchor
                easy_indices = ((this_y_l == cls_id) & (this_y_pred == cls_id)).nonzero().squeeze(
                    1)  # 预测与标签一致的anchor
                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_anchor / 2 and num_easy >= n_anchor / 2:
                    num_hard_keep = n_anchor // 2
                    num_easy_keep = n_anchor - num_hard_keep
                elif num_hard >= n_anchor / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_anchor - num_easy_keep
                elif num_easy >= n_anchor / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_anchor - num_hard_keep
                else:
                    # Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception
                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                # indices = torch.cat((hard_indices, easy_indices), dim=0)
                this_class_feat = torch.cat([this_feat_l[hard_indices, :], this_feat_l[easy_indices, :]], 0)
                feats_ = torch.cat([feats_, this_class_feat.unsqueeze(0)], 0)
                labels_ = torch.cat([labels_, cls_id.unsqueeze(0)], 0)

        # # unsupervised anchor sample
        if epoch >= 50:
            for idx in range(b):
                inx_mask = torch.ones([h, w], device=pseudo_logits1_up.device)
                x_ul1_mask_idx = inx_mask.clone()
                x_ul1_mask_idx[ul1_f_up[0, idx]:br1_f_up[0, idx], ul1_f_up[1, idx]:br1_f_up[1, idx]] = 0

                x_ul1_anchor_idx = (x_ul1_mask_idx * (pseudo_logits1_up[idx] >= easy_thresh)).nonzero()
                x_ul1_anchors_feat_idx = output_ul1_up[idx, :, x_ul1_anchor_idx[:, 0],
                                         x_ul1_anchor_idx[:, 1]].permute(1, 0)
                x_ul1_anchors_lb_idx = pseudo_label1_up[idx, x_ul1_anchor_idx[:, 0], x_ul1_anchor_idx[:, 1]]

                pseudo_logits_list_max, max_logit_idx_of_12 = torch.max(
                    torch.stack([pseudo_logits_list1_up[idx], pseudo_logits_list2_up[idx]], 0), 0)
                enc_feature_together = torch.stack([enc_feature_list1_up[idx], enc_feature_list2_up[idx]], 0)
                enc_feature_max = enc_feature_together[max_logit_idx_of_12, range(enc_feature_together.shape[1]), :]

                easy_anchor_idx = ((pseudo_label_list1_up[idx] == pseudo_label_list2_up[idx]) * (
                        pseudo_logits_list_max >= easy_thresh)).nonzero().squeeze(-1)
                easy_anchor_feat_idx = enc_feature_max[easy_anchor_idx, :]
                easy_anchor_prob_idx = pseudo_logits_list_max[easy_anchor_idx]
                easy_anchor_lb_idx = pseudo_label_list1_up[idx][easy_anchor_idx]

                random_ul_anchor_feat = torch.cat(
                    [x_ul1_anchors_feat_idx, easy_anchor_feat_idx], 0)
                random_ul_anchor_lb = torch.cat(
                    [x_ul1_anchors_lb_idx, easy_anchor_lb_idx], 0)

                hard_ul_anchor_idx = ((pseudo_label_list1_up[idx] != pseudo_label_list2_up[idx]) * (
                        pseudo_logits_list_max >= hard_thresh)).nonzero().squeeze(-1)
                hard_ul_anchor_feat = enc_feature_max[hard_ul_anchor_idx, :]
                hard_ul_anchor_lb = pseudo_label_list1_up[idx][hard_ul_anchor_idx]

                this_classes = torch.unique(torch.cat([hard_ul_anchor_lb, random_ul_anchor_lb]))
                this_classes = [x for x in this_classes if x != self.ignore_index]
                # this_classes = [x for x in this_classes if self.segment_queue_ptr[x] == self.queue_len]
                this_classes = [x for x in this_classes if (random_ul_anchor_lb == x).half().sum() + (
                        hard_ul_anchor_lb == x).half().sum() > n_anchor]
                for cls_id in this_classes:
                    hard_indices = (hard_ul_anchor_lb == cls_id).nonzero().squeeze(-1)
                    random_indices = (random_ul_anchor_lb == cls_id).nonzero().squeeze(-1)
                    num_hard = hard_indices.shape[0]
                    num_random = random_indices.shape[0]
                    if num_hard >= n_anchor / 2 and num_random >= n_anchor / 2:
                        num_hard_keep = n_anchor // 2
                        num_random_keep = n_anchor - num_hard_keep
                    elif num_hard >= n_anchor / 2:
                        num_random_keep = num_random
                        num_hard_keep = n_anchor - num_random_keep
                    elif num_random >= n_anchor / 2:
                        num_hard_keep = num_hard
                        num_random_keep = n_anchor - num_hard_keep
                    else:
                        raise Exception(
                            'this shoud be never touched! {} {} {}'.format(num_hard, num_random, n_anchor))
                    perm = torch.randperm(num_hard)
                    hard_indices = hard_indices[perm[:num_hard_keep]]
                    perm = torch.randperm(num_random)
                    random_indices = random_indices[perm[:num_random_keep]]
                    # indices = torch.cat((hard_indices, random_indices), dim=0)
                    this_class_feat = torch.cat(
                        [hard_ul_anchor_feat[hard_indices, :], random_ul_anchor_feat[random_indices, :]], 0)
                    feats_ = torch.cat([feats_, this_class_feat.unsqueeze(0)], 0)
                    labels_ = torch.cat([labels_, cls_id.unsqueeze(0)], 0)

        return feats_, labels_