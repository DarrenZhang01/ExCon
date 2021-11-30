"""
An utility class for initializing different explainer objects.
"""

from torch import nn
import numpy as np
from captum.attr import DeepLift, IntegratedGradients, ShapleyValueSampling, LayerGradCam, Saliency
from captum.attr._utils.attribution import LayerAttribution


class Explainer(nn.Module):
    def __init__(self, method, model_name, dataset, model, training):
        super(Explainer, self).__init__()
        self.model = model
        self.explainer = None
        if method == "DeepLift":
            self.explainer = DeepLift(self.model)
        elif method == "IntegratedGradients":
            self.explainer = IntegratedGradients(self.model)
        elif method == "ShapleyValueSampling":
            self.explainer = ShapleyValueSampling(self.model)
        elif method == "Saliency":
            self.explainer = Saliency(self.model)
        elif method == "GradCAM":
            # Need to retrieve the `module` in the data parallel mode for multi-GPU processing.
            if model_name == 'vgg16':
                if dataset.startswith('cifar'):
                    self.explainer = LayerGradCam(self.model, self.model[0].features[21])
                else:
                    self.explainer = LayerGradCam(self.model, self.model[0].features[30])
            elif model_name == 'resnet56':
                self.explainer = LayerGradCam(self.model, self.model[0].layer3[8].conv2)
            elif model_name == 'resnet50':
                if dataset.startswith('cifar'):
                    if "CE" in training:
                        self.explainer = LayerGradCam(self.model, self.model.module.encoder.layer2[3].conv3)
                    else:
                        self.explainer = LayerGradCam(self.model, self.model[0].layer2[3].conv3)
                else:
                    if "CE" in training:
                        self.explainer = LayerGradCam(self.model, self.model.module.encoder.layer4[2].conv3)
                    else:
                        self.explainer = LayerGradCam(self.model, self.model[0].layer4[2].conv3)
            elif model_name == 'resnet18':
                if dataset.startswith('cifar'):
                    if "CE" in training:
                        self.explainer = LayerGradCam(self.model, self.model.module.encoder.layer2[1].conv2)
                    else:
                        self.explainer = LayerGradCam(self.model, self.model[0].layer2[1].conv2)
                else:
                    if "CE" in training:
                        self.explainer = LayerGradCam(self.model, self.model.module.encoder.layer4[1].conv2)
                    else:
                        self.explainer = LayerGradCam(self.model, self.model[0].layer4[1].conv2)
        self.method = method
        if dataset in ['cifar10', 'cifar100', 'SVHN']:
            self.img_size = (32, 32)
        elif dataset == 'ImageNet':
            self.img_size = (64, 64)

    def attribute(self, input, target, omit_channel=True):
        """
        omit_channel: bool, whether to omit the channel dimension in the explanations.
        """
        importances = self.explainer.attribute(inputs=input, target=target)
        if self.method == "GradCAM":
            importances = LayerAttribution.interpolate(importances,
                                                       self.img_size,
                                                       interpolate_mode="bilinear")
            if omit_channel is False:
                importances = importances.repeat((1, 3, 1, 1))
        else:
            if omit_channel is False:
                importances = importances
            else:
                importances = importances.sum(dim=1)
        importances = importances.squeeze().detach().cpu().numpy()
        return importances

    def normalize(self, importances):
        importances = importances - np.min(importances, axis=(0, 1))
        importances = importances / (np.max(importances, axis=(0, 1)) + 1e-8)
        return importances
