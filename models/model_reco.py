import math, time
from itertools import chain
import torch
import torch.nn.functional as F
from torch import nn
from base import BaseModel
from utils.losses import *
from models.encoder import Encoder
from models.modeling.deeplab import DeepLab as DeepLab_v3p
# from models.modeling.net_utils import NLM, NLM_dot, NLM_woSoft, NLM_NC_woSoft, Batch_Contrastive
import numpy as np
import cv2
import kornia
from models.feature_memory import *
from memory_profiler import profile
from lib.models.tools.module_helper import ModuleHelper
from models.module_list_reco import *
from models.loss_our import PixelContrastLoss
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
        self.gpu_mem = open("/home/zhengyu/ours_video/gpu_memory.txt", 'w+')
        self.register_buffer("segment_queue", torch.randn(num_classes, self.queue_len, self.proj_final_dim))
        self.segment_queue = F.normalize(self.segment_queue, p=2, dim=2)
        self.register_buffer("segment_queue_ptr", torch.zeros(num_classes, dtype=torch.long))
        # feature_memory = FeatureMemory(num_samples=labeled_samples, dataset=dataset, memory_per_class=256,
        #                                feature_size=256, n_classes=num_classes)
        # self.segment_queue =
        self.pos_sample_num = 256#1024
        self.neg_sample_num = 512#2048
        assert self.layers in [50, 101]
        self.crop_size = [320, 320]
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
        else:
            raise ValueError("No such backbone {}".format(self.backbone))

        if self.mode == 'semi':
            self.project = nn.Sequential(
                nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, stride=1),
                ModuleHelper.BNReLU(self.out_dim, bn_type='torchsyncbn'),
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

        self.CTLOSS = PixelContrastLoss(temperature=self.temperature,
                                        neg_num=20,
                                        memory_bank=None,
                                        mining=True)

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
            if epoch < self.epoch_start_cycle:
                return total_loss, curr_losses, outputs
            #####################Memory Construction############################################
            print("1:{}".format(torch.cuda.memory_allocated(0)), file=self.gpu_mem)
            # model.eval()
            # proj_labeled_features_correct = model.projection_head(labeled_features_correct)
            # model.train()
            # with torch.no_grad():
            # if epoch < 5:
            #     with torch.no_grad():
            #         enc_l = self.project(enc.detach())
            # else:
            enc_l = self.project(enc)
            enc_l = F.normalize(enc_l, 2, 1)
            ##################### unsup loss #############################################
            # x_ul: [batch_size, 2, 3, H, W]
            x_ul1 = x_ul[:, 0, :, :, :]
            # x_ul2 = x_ul[:, 1, :, :, :]

            # enc_ul1_up = self.encoder(x_ul1)
            # output_ul1_up = self.project(enc_ul1_up)
            # output_ul1_up = F.normalize(output_ul1_up, 2, 1)

            ##################### reco loss #############################################
            # with torch.no_grad():
            #     enc_ul1_up = self.encoder(x_ul1)
            #     logits1_up = self.classifier(enc_ul1_up) #[batch_size, num_classes, h, w]
            #     pred_u_large_raw = F.interpolate(logits1_up, size=target_l.shape[1:], mode='bilinear',
            #                                      align_corners=True)
            #     pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1)
            #
            #     # random scale images first
            #     train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
            #         batch_transform(x_ul1, pseudo_labels, pseudo_logits,
            #                         self.crop_size, (1.0, 1.0), apply_augmentation=False)
            #
            #     # apply mixing strategy: cutout, cutmix or classmix
            #     train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
            #         generate_unsup_data(train_u_aug_data, train_u_aug_label, train_u_aug_logits, mode='cutout')
            #
            #     # apply augmentation: color jitter + flip + gaussian blur
            #     train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
            #         batch_transform(train_u_aug_data, train_u_aug_label, train_u_aug_logits,
            #                         self.crop_size, (1.0, 1.0), apply_augmentation=True)
            #
            # enc_u = self.encoder(train_u_aug_data)
            # pred_u = self.classifier(enc_u)
            # rep_u = self.project(enc_u)
            # rep_u = F.normalize(rep_u, 2, 1)
            # # pred_u_large = F.interpolate(pred_u, size=train_l_label.shape[1:], mode='bilinear', align_corners=True)
            #
            # rep_all = torch.cat((enc_l, rep_u))
            # pred_all = torch.cat((enc_logit, pred_u))
            # with torch.no_grad():
            #     train_u_aug_mask = train_u_aug_logits.ge(0.7).float()
            #     train_l_label = target_l.clone().detach()
            #     train_l_label[train_l_label == 255] = -1
            #     mask_all = torch.cat(((train_l_label.unsqueeze(1) >= 0).float(), train_u_aug_mask.unsqueeze(1)))
            #     mask_all = F.interpolate(mask_all, size=pred_all.shape[2:], mode='nearest')
            #
            #     label_l = F.interpolate(label_onehot(train_l_label, self.num_classes), size=pred_all.shape[2:],
            #                             mode='nearest')
            #     label_u = F.interpolate(label_onehot(train_u_aug_label, self.num_classes),
            #                             size=pred_all.shape[2:], mode='nearest')
            #     label_all = torch.cat((label_l, label_u))
            #
            #     prob_l = torch.softmax(enc_logit, dim=1)
            #     prob_u = torch.softmax(pred_u, dim=1)
            #     prob_all = torch.cat((prob_l, prob_u))
            #
            # loss_reco = self.weight_inter * compute_reco_loss(rep_all, label_all, mask_all, prob_all, 0.97,
            #                               0.5, self.pos_sample_num, self.neg_sample_num)
            # curr_losses['loss_inter'] = loss_reco
            # total_loss = total_loss + loss_reco
            # # if epoch < self.epoch_start_inter:
            # return total_loss, curr_losses, outputs
            ##################### pixel contrast loss #############################################
            enc_ul1_up = self.encoder(x_ul1)
            with torch.no_grad():
                logits1_up = self.classifier(enc_ul1_up) #[batch_size, num_classes, h, w]
                pred_u_large_raw = F.interpolate(logits1_up, size=target_l.shape[1:], mode='bilinear',
                                                 align_corners=True)
                # pseudo_logits, pseudo_labels = torch.max(torch.softmax(logits1_up, dim=1), dim=1)
            teacher_seg = F.softmax(pred_u_large_raw, dim=1)
            unlabeled_pred = torch.max(teacher_seg, dim=1)[1]

            rep_u = self.project(enc_ul1_up)
            rep_u = F.normalize(rep_u, 2, 1)
            pred_seg = torch.cat([enc_logit,logits1_up])
            # if self.use_cb_consist:
            #     unlabeled_pred[(1 - ignore_mask_unlabeled[:, 0]) == 1] = 255
            pred_seg = torch.nn.functional.interpolate(pred_seg.float(),
                                                       (rep_u.shape[2], rep_u.shape[3]),
                                                       mode='bilinear')
            pred_seg = torch.max(pred_seg, dim=1)[1]  # student data
            target = torch.cat([target_l, unlabeled_pred])  # !!!!!!!!!! labeled+unlabeled cross set
            feat = torch.cat([enc_l, rep_u])
            feat_t = None

            ignore_mask_unlabeled = torch.ones(1).to(output_l.device)
            ignore_mask_unlabeled = ignore_mask_unlabeled[:, None, ...]
            # if self.args.mask_contrast:
            # sup_contrast_loss = self.CTLOSS(feats=feat, feats_t=feat_t, labels=target, predict=pred_seg,
            #                                     cb_mask=ignore_mask_unlabeled)
            # else:
            sup_contrast_loss = self.weight_inter * self.CTLOSS(feats=feat, feats_t=feat_t, labels=target, predict=pred_seg)
            curr_losses['loss_inter'] = sup_contrast_loss
            total_loss = total_loss + sup_contrast_loss
            # if epoch < self.epoch_start_inter:
            return total_loss, curr_losses, outputs
        else:
            raise ValueError("No such mode {}".format(self.mode))

    # def _contrastive(self, X_anchor, y_anchor, queue=None):
    #     anchor_num, n_view = X_anchor.shape[0], X_anchor.shape[
    #         1]  # 22*46*256  anchor_num:batch中所有class的个数， n_view:每个class中sample的个数
    #
    #     y_anchor = y_anchor.contiguous().view(-1, 1)  # 22
    #     anchor_count = n_view  # 每个class中sample的个数
    #     anchor_feature = torch.cat(torch.unbind(X_anchor, dim=1),
    #                                dim=0)  # unbind 从dim进行切片，并返回切片的结果，返回的结果里面没有dim这个维度 得到n_view个class_num*n_dim的tensor
    #     # 再进行cat，就是将所有sample的特征并了起来，结合起来相当于从每个class依次取一个sample，循环往复拼接了起来（22*46）*256
    #     # X_contrast, y_contrast = self._sample_negative(
    #     #     queue)  # 将19*1000*256的memory变成了19000*256的memopry，标签也是按顺序 y则为1000个0接1000个1接1000个2
    #     # y_contrast = y_contrast.contiguous().view(-1, 1)  # 19*1000
    #     # contrast_count = 1
    #     # contrast_feature = X_contrast  # memory的特征（19*1000）*256
    #
    #     contrast_feature = queue.view(-1, queue.shape[
    #         -1])  # 将19*1000*256的memory变成了19000*256的memopry，标签也是按顺序 y则为1000个0接1000个1接1000个2
    #     y_contrast = torch.arange(self.num_classes).unsqueeze(1).repeat(1, self.queue_len)
    #     y_contrast = y_contrast.contiguous().view(-1, 1).cuda()  # 19000
    #     contrast_count = 1
    #
    #     mask = torch.eq(y_anchor,
    #                     y_contrast.T).float()  # torch.eq对两个tensor逐元素比较 个位置的0 1 #把一横一竖的tensor比较得到N*N的tensor，即标签一致的位置为1 22*19000
    #     mask = mask.repeat(anchor_count,
    #                        contrast_count).cuda()  # 将22*19000扩展到（22*49）*19000，所有anchor和sample的对应关系，所有正样本，即所有标签一致的对
    #     anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
    #                                     self.temperature)  # （22*46）*256 * 256*（19*1000） = （22*46）*（19*1000） div除法除以temperature，每个anchor与memory中每个sample相乘的结果
    #     logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    #     logits = anchor_dot_contrast - logits_max.detach()  # 减去最大相似度 1012*19000
    #
    #     anchor_size = mask.shape[0]
    #     # positive sample
    #     pos_indice = torch.ones(anchor_size, self.queue_len)
    #     for i in range(anchor_size): pos_indice[i] = torch.randperm(self.queue_len)
    #     pos_indice = pos_indice[:, :self.pos_sample_num].long().flatten()
    #     pos_rand_x = torch.arange(anchor_size).unsqueeze(1).repeat([1, self.pos_sample_num]).flatten()
    #     pos_mask_all = mask.nonzero()[:, 1].view(anchor_size, -1)
    #     pos_rand_y = pos_mask_all[pos_rand_x, pos_indice]
    #     pos_sample_mask = torch.zeros_like(mask)
    #     pos_sample_mask[pos_rand_x, pos_rand_y] = 1
    #
    #     # negative sample
    #     neg_mask = 1 - mask  # 所有标签不一致的pair
    #     neg_indice = torch.ones(anchor_size, self.queue_len * (self.num_classes - 1))
    #     for i in range(anchor_size): neg_indice[i] = torch.randperm(self.queue_len * (self.num_classes - 1))
    #     neg_indice = neg_indice[:, :self.neg_sample_num].long().flatten()
    #     neg_rand_x = torch.arange(anchor_size).unsqueeze(1).repeat([1, self.neg_sample_num]).flatten()
    #     neg_mask_all = neg_mask.nonzero()[:, 1].view(anchor_size, -1)
    #     neg_rand_y = neg_mask_all[neg_rand_x, neg_indice]
    #     neg_sample_mask = torch.zeros_like(neg_mask)
    #     neg_sample_mask[neg_rand_x, neg_rand_y] = 1
    #
    #     neg_logits = torch.exp(logits) * neg_sample_mask
    #     neg_logits = neg_logits.sum(1, keepdim=True)  # 每个anchor所有负样本的和（22*46）*1
    #
    #     exp_logits = torch.exp(logits)  # 1012*19000
    #
    #     log_prob = logits - torch.log(exp_logits + neg_logits)  # 1012*19000+1012*1 把每个sample都与neg的和相加，
    #     # 在下面的计算中与pos的mask一乘则能得到infonce的分母exp（pos）+sum（exp（neg）），因为infonce每一个pos都要与所有neg相加，而不是pos的和加neg的和
    #
    #     mean_log_prob_pos = (pos_sample_mask * log_prob).sum(1) / pos_sample_mask.sum(1)
    #
    #     loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
    #     loss = loss.mean()
    #
    #     return loss

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

    # def _dequeue_and_enqueue(self, keys, labels, preds,
    #                          segment_queue, segment_queue_ptr):
    #     with torch.no_grad():
    #         batch_size = keys.shape[0]
    #         feat_dim = keys.shape[1]
    #         class_num = preds.shape[1]
    #
    #         # labels = labels[:, ::self.network_stride, ::self.network_stride]
    #         labels_down = F.interpolate(labels.float().unsqueeze(1),
    #                                     size=(keys.shape[2], keys.shape[3]),
    #                                     mode='nearest').squeeze(1)
    #         probs = torch.softmax(preds, dim=1)
    #         _, pred_labels = torch.max(probs, dim=1)
    #         for bs in range(batch_size):
    #             this_feat = keys[bs].contiguous().view(feat_dim, -1).T
    #             this_label = labels_down[bs].contiguous().view(-1)
    #             this_label_ids = torch.unique(this_label)
    #             this_label_ids = [x for x in this_label_ids if x < 255]
    #             this_preds = pred_labels[bs].contiguous().view(-1)
    #             this_probs = probs[bs].contiguous().view(class_num, -1)
    #
    #             for lb in this_label_ids:
    #                 # idxs = (this_label == lb).nonzero()
    #                 lb = lb.long()
    #                 idxs_easy = ((this_label == lb).float() * (this_preds == lb).float()).nonzero().squeeze(-1)
    #                 idxs_hard = ((this_label == lb).float() * (this_preds != lb).float()).nonzero().squeeze(-1)
    #                 # print(idxs_easy.shape)
    #                 # if
    #                 weight_easy = torch.ones(idxs_easy.shape[0], 1).cuda()
    #                 weight_hard = this_probs[lb, idxs_hard].unsqueeze(-1)
    #                 new_feat = torch.cat([this_feat[idxs_easy, :], this_feat[idxs_hard, :]], dim=0)
    #                 new_weight = torch.softmax(torch.cat([weight_easy, weight_hard]), dim=0)
    #                 feat = torch.sum(new_feat * new_weight.repeat(1, feat_dim), dim=0)
    #                 ptr = int(segment_queue_ptr[lb])
    #                 K = idxs_easy.shape[0]
    #                 if ptr + K <= self.queue_len:
    #                     segment_queue[lb, ptr:ptr + K, :] = F.normalize(this_feat[idxs_easy, :], dim=1)
    #                     segment_queue_ptr[lb] = segment_queue_ptr[lb] + K
    #                 elif ptr < self.queue_len and ptr + K > self.queue_len:
    #                     permK = torch.randperm(K)
    #                     segment_queue[lb, ptr:, :] = F.normalize(this_feat[permK[:(self.queue_len - ptr)], :], dim=1)
    #                     segment_queue_ptr[lb] = self.queue_len
    #                 elif ptr == self.queue_len:
    #                     segment_queue[lb, :, :] = torch.cat(
    #                         [segment_queue[lb, 1:, :], F.normalize(feat.unsqueeze(0), dim=1)], 0)
    #
    # def _anchor_sampling(self, enc_l, pred_key, target_l, output_ul1_up, output_ul2_up, pseudo_label1_up,
    #                      pseudo_label2_up, enc_feature_list1_up, enc_feature_list2_up, pseudo_label_list1_up,
    #                      pseudo_label_list2_up,
    #                      ul1_f_up, ul2_f_up, br1_f_up, br2_f_up):
    #     b, c, w, h = output_ul1_up.size()
    #     n_anchor = self.max_samples
    #     feats_ = torch.zeros(0, n_anchor, c).cuda()
    #     labels_ = torch.zeros(0).cuda()
    #     # # unsupervised anchor sample
    #     for idx in range(b):
    #         inx_mask = torch.ones(h, w)
    #         x_ul1_mask_idx = inx_mask.clone()
    #         x_ul1_mask_idx[ul1_f_up[0, idx]:br1_f_up[0, idx], ul1_f_up[1, idx]:br1_f_up[1, idx]] = 0
    #         x_ul1_anchor_idx = x_ul1_mask_idx.nonzero()
    #         x_ul1_anchors_feat_idx = output_ul1_up[idx, :, x_ul1_anchor_idx[:, 0], x_ul1_anchor_idx[:, 1]].permute(1, 0)
    #         x_ul1_anchors_lb_idx = pseudo_label1_up[idx, x_ul1_anchor_idx[:, 0], x_ul1_anchor_idx[:, 1]]
    #
    #         x_ul2_mask_idx = inx_mask.clone()
    #         x_ul2_mask_idx[ul2_f_up[0, idx]:br2_f_up[0, idx], ul2_f_up[1, idx]:br2_f_up[1, idx]] = 0
    #         x_ul2_anchor_idx = x_ul2_mask_idx.nonzero()
    #         x_ul2_anchors_feat_idx = output_ul2_up[idx, :, x_ul2_anchor_idx[:, 0], x_ul2_anchor_idx[:, 1]].permute(1, 0)
    #         x_ul2_anchors_lb_idx = pseudo_label2_up[idx, x_ul2_anchor_idx[:, 0], x_ul2_anchor_idx[:, 1]]
    #
    #         easy_anchor_idx = (pseudo_label_list1_up[idx] == pseudo_label_list2_up[idx]).nonzero().squeeze(-1)
    #         easy1_anchor_feat_idx = enc_feature_list1_up[idx][easy_anchor_idx, :]
    #         easy1_anchor_lb_idx = pseudo_label_list1_up[idx][easy_anchor_idx]
    #         easy2_anchor_feat_idx = enc_feature_list2_up[idx][easy_anchor_idx, :]
    #         easy2_anchor_lb_idx = pseudo_label_list2_up[idx][easy_anchor_idx]
    #
    #         random_ul_anchor_feat = torch.cat(
    #             [x_ul1_anchors_feat_idx, x_ul2_anchors_feat_idx, easy1_anchor_feat_idx, easy2_anchor_feat_idx], 0)
    #         random_ul_anchor_lb = torch.cat(
    #             [x_ul1_anchors_lb_idx, x_ul2_anchors_lb_idx, easy1_anchor_lb_idx, easy2_anchor_lb_idx], 0)
    #
    #         hard_ul_anchor_idx = (pseudo_label_list1_up[idx] != pseudo_label_list2_up[idx]).nonzero().squeeze(-1)
    #         hard1_anchor_feat_idx = enc_feature_list1_up[idx][hard_ul_anchor_idx, :]
    #         hard1_anchor_lb_idx = pseudo_label_list1_up[idx][hard_ul_anchor_idx]
    #         hard2_anchor_feat_idx = enc_feature_list2_up[idx][hard_ul_anchor_idx, :]
    #         hard2_anchor_lb_idx = pseudo_label_list2_up[idx][hard_ul_anchor_idx]
    #         hard_ul_anchor_feat = torch.cat([hard1_anchor_feat_idx, hard2_anchor_feat_idx], 0)
    #         hard_ul_anchor_lb = torch.cat([hard1_anchor_lb_idx, hard2_anchor_lb_idx], 0)
    #
    #         this_classes = torch.unique(torch.cat([hard_ul_anchor_lb, random_ul_anchor_lb]))
    #         this_classes = [x for x in this_classes if x != self.ignore_index]
    #         # this_classes = [x for x in this_classes if self.segment_queue_ptr[x] == self.queue_len]
    #         this_classes = [x for x in this_classes if (random_ul_anchor_lb == x).float().sum() + (
    #                 hard_ul_anchor_lb == x).float().sum() > n_anchor]
    #         for cls_id in this_classes:
    #             hard_indices = (hard_ul_anchor_lb == cls_id).nonzero().squeeze(-1)
    #             random_indices = (random_ul_anchor_lb == cls_id).nonzero().squeeze(-1)
    #             num_hard = hard_indices.shape[0]
    #             num_random = random_indices.shape[0]
    #             if num_hard >= n_anchor / 2 and num_random >= n_anchor / 2:
    #                 num_hard_keep = n_anchor // 2
    #                 num_random_keep = n_anchor - num_hard_keep
    #             elif num_hard >= n_anchor / 2:
    #                 num_random_keep = num_random
    #                 num_hard_keep = n_anchor - num_random_keep
    #             elif num_random >= n_anchor / 2:
    #                 num_hard_keep = num_hard
    #                 num_random_keep = n_anchor - num_hard_keep
    #             else:
    #                 raise Exception('this shoud be never touched! {} {} {}'.format(num_hard, num_random, n_anchor))
    #             perm = torch.randperm(num_hard)
    #             hard_indices = hard_indices[perm[:num_hard_keep]]
    #             perm = torch.randperm(num_random)
    #             random_indices = random_indices[perm[:num_random_keep]]
    #             # indices = torch.cat((hard_indices, random_indices), dim=0)
    #             this_class_feat = torch.cat(
    #                 [hard_ul_anchor_feat[hard_indices, :], random_ul_anchor_feat[random_indices, :]], 0)
    #             feats_ = torch.cat([feats_, this_class_feat.unsqueeze(0)], 0)
    #             labels_ = torch.cat([labels_, cls_id.unsqueeze(0)], 0)
    #
    #     # supervised anchor sample
    #     target_l_down = F.interpolate(target_l.float().unsqueeze(1), size=enc_l.size()[2:], mode='nearest').squeeze(1)
    #     for idx in range(b):
    #         this_feat_l = enc_l[idx].contiguous().view(enc_l.shape[1], -1).permute(1, 0)
    #         this_y_pred = pred_key[idx].max(0)[1].contiguous().view(-1)
    #         this_y_l = target_l_down[idx].contiguous().view(-1)
    #         this_classes_l = torch.unique(this_y_l)
    #         this_classes_l = [x for x in this_classes_l if x != self.ignore_index]
    #         this_classes_l = [x for x in this_classes_l if (this_y_l == x).nonzero().shape[0] > n_anchor]  # 删除样本数太少的样本
    #         for cls_id in this_classes_l:
    #             hard_indices = ((this_y_l == cls_id) & (this_y_pred != cls_id)).nonzero().squeeze(1)  # 预测与标签不一致的anchor
    #             easy_indices = ((this_y_l == cls_id) & (this_y_pred == cls_id)).nonzero().squeeze(1)  # 预测与标签一致的anchor
    #             num_hard = hard_indices.shape[0]
    #             num_easy = easy_indices.shape[0]
    #
    #             if num_hard >= n_anchor / 2 and num_easy >= n_anchor / 2:
    #                 num_hard_keep = n_anchor // 2
    #                 num_easy_keep = n_anchor - num_hard_keep
    #             elif num_hard >= n_anchor / 2:
    #                 num_easy_keep = num_easy
    #                 num_hard_keep = n_anchor - num_easy_keep
    #             elif num_easy >= n_anchor / 2:
    #                 num_hard_keep = num_hard
    #                 num_easy_keep = n_anchor - num_hard_keep
    #             else:
    #                 # Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
    #                 raise Exception
    #             perm = torch.randperm(num_hard)
    #             hard_indices = hard_indices[perm[:num_hard_keep]]
    #             perm = torch.randperm(num_easy)
    #             easy_indices = easy_indices[perm[:num_easy_keep]]
    #             # indices = torch.cat((hard_indices, easy_indices), dim=0)
    #             this_class_feat = torch.cat([this_feat_l[hard_indices, :], this_feat_l[easy_indices, :]], 0)
    #             feats_ = torch.cat([feats_, this_class_feat.unsqueeze(0)], 0)
    #             labels_ = torch.cat([labels_, cls_id.unsqueeze(0)], 0)
    #     return feats_, labels_
# -*-coding:utf-8-*-
