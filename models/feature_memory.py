"""
Implementation for the Memory Bank for pixel-level feature vectors
"""

import torch
import numpy as np
import random

class FeatureMemory:
    def __init__(self, num_samples, memory_per_class=2048, feature_size=256, n_classes=19):
        self.num_samples = num_samples
        self.memory_per_class = memory_per_class
        self.feature_size = feature_size
        self.memory = [None] * n_classes
        self.n_classes = n_classes
        self.per_class_samples_per_image = max(1, int(round(memory_per_class / num_samples)))
        # if dataset == 'cityscapes': # usually all classes in one image
        #     self.per_class_samples_per_image = max(1, int(round(memory_per_class / num_samples)))
        # elif dataset == 'pascal_voc': # usually only around 3 classes on each image, except background class
        #     self.per_class_samples_per_image = max(1, int(n_classes / 3 * round(memory_per_class / num_samples)))



    def add_features_from_sample_learned(self, features, class_labels, batch_size):
        """
        Updates the memory bank with some quality feature vectors per class
        Args:
            model: segmentation model containing the self-attention modules (contrastive_class_selectors)
            features: BxFxWxH feature maps containing the feature vectors for the contrastive (already applied the projection head)
            class_labels:   BxWxH  corresponding labels to the [features]
            batch_size: batch size

        Returns:

        """
        features = features.detach()
        class_labels = class_labels.detach().cpu().numpy()
        #每个class在每张图中中sample的特征数，这里为1，在一个batch中则为batch的size
        elements_per_class = batch_size * self.per_class_samples_per_image

        # for each class, save [elements_per_class]
        for c in range(self.n_classes):
            mask_c = class_labels == c  # get mask for class c
            selector = model.__getattr__('contrastive_class_selector_' + str(c))  # get the self attention moduel for class c
            features_c = features[mask_c, :] # get features from class c   属于c类的像素点个数*维度256
            if features_c.shape[0] > 0:
                if features_c.shape[0] > elements_per_class:
                    with torch.no_grad():
                        # get ranking scores
                        rank = selector(features_c)
                        rank = torch.sigmoid(rank)
                        # sort them
                        _, indices = torch.sort(rank[:, 0], dim=0)
                        indices = indices.cpu().numpy()
                        features_c = features_c.cpu().numpy()
                        # get features with highest rankings
                        features_c = features_c[indices, :]
                        new_features = features_c[:elements_per_class, :] #用rank将特征排序，然后取e个特征，即每类在每个batch中的个数
                else:
                    new_features = features_c.cpu().numpy()

                if self.memory[c] is None: # was empy, first elements
                    self.memory[c] = new_features

                else: # add elements to already existing list
                    # keep only most recent memory_per_class samples
                    self.memory[c] = np.concatenate((new_features, self.memory[c]), axis = 0)[:self.memory_per_class, :]



