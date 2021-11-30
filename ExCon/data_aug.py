"""
References:
1. https://github.com/HobbitLong/SupContrast/blob/master/util.py
"""


from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim



def aug_no_bbox_mc(inputs, input_batch2, targets, explainer, model, num_classes, p, tau, backup, opt):
    """
    inputs: shape (batch size, channels, side length, side length)
    """
    #tau = 0.5
    #p = 0.
    explanation = explainer.attribute(inputs, targets)
    # print("explanation shape: {} input shape: {}".format(explanation.shape, inputs.shape))
    # if explainer.method == "GradCAM":
    explanation = explanation.transpose(1, 2, 0)
    explanation = explanation - np.min(explanation, axis=(0, 1))
    explanation = explanation / (np.max(explanation, axis=(0, 1)) + 1e-8)
    explanation = (explanation <= tau).transpose(2, 0, 1)

    explanation = explanation[:, np.newaxis]
    explanation = np.repeat(explanation, 3, axis=1)
    # print("the explainer method is: {}".format(explainer.method))
    augmented = torch.from_numpy(np.where(explanation, np.zeros_like(inputs.cpu().numpy()), inputs.cpu().numpy()))

    if torch.cuda.is_available():
        augmented = augmented.cuda()
    _, preds_aug = torch.max(model(augmented).data, 1)
    _, preds_inputs = torch.max(model(inputs).data, 1)
    new_inputs = torch.zeros_like(inputs)
    new_targets = targets.clone()
    flag = np.zeros(len(targets))
    for i in range(len(targets)):
        # correct prediction on original and augmented image
        if preds_inputs[i].item() == targets[i].item() and preds_aug[i].item() == targets[i].item():
            new_inputs[i] = augmented[i]
        # incorrect prediction on original and correct prediction on augmented image
        elif preds_inputs[i].item() != targets[i].item() and preds_aug[i].item() == targets[i].item():
            new_inputs[i] = augmented[i]
        # incorrect prediction on original and augmented image
        else:
            new_inputs[i] = backup[i]
            flag[i] = 1
            # If we want to include the negative pairs in training ExCon,
            # we need to include the masked image with the wrong prediction as
            # a different training data in both batch 1 and batch 2 with a
            # background label.
            if opt.negative_pair == 1:
                new_label = torch.Tensor([opt.n_cls])
                if torch.cuda.is_available():
                    new_label = new_label.cuda()
                new_inputs = torch.cat((new_inputs, augmented[i:i+1]), dim=0)
                input_batch2 = torch.cat((input_batch2, augmented[i:i+1]), dim=0)
                new_targets = torch.cat((new_targets, new_label), dim=0)
    return new_inputs, input_batch2, new_targets, flag


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, random_transform, standard_transform, opt):
        self.random_transform = random_transform
        self.standard_transform = standard_transform
        self.opt = opt

    def __call__(self, x):
        if 'Ex' in self.opt.method:
            # # Add one more random cropping besides the one standard transformation and the one random cropping.
            # # Since if the masked image later on does not give a correct prediction, then we want to use
            # # the randomly cropped version of the image rather than the image with only standard transformation
            # return [self.standard_transform(x), self.standard_transform(x), self.random_transform(x), self.random_transform(x)]
            return [self.standard_transform(x), self.random_transform(x), self.random_transform(x)]
        else:
            return [self.random_transform(x), self.random_transform(x)]
