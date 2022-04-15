import numpy as np
from copy import deepcopy
import torch
from torch.nn import functional as F
from captum.attr import DeepLiftShap, Saliency
from lime import lime_image

# defining global variables
global values
global discriminatorLime

from torch.autograd.variable import Variable
from torch import Tensor, from_numpy, randn, full

import torch.nn as nn


def get_explanation_dz(generated_data, discriminator, prediction, XAItype="shap", cuda=True, trained_data=None,
                       labels=None, device=None, data_type="mnist") -> None:
    """
    This function calculates the explanation for given generated images using the desired xAI systems and the
    :param generated_data: data created by the generator
    :type generated_data: torch.Tensor
    :param discriminator: the discriminator model
    :type discriminator: torch.nn.Module
    :param prediction: tensor of predictions by the discriminator on the generated data
    :type prediction: torch.Tensor
    :param XAItype: the type of xAI system to use. One of ("shap", "lime", "saliency")
    :type XAItype: str
    :param cuda: whether to use gpu
    :type cuda: bool
    :param trained_data: a batch from the dataset
    :type trained_data: torch.Tensor
    :param data_type: the type of the dataset used. One of ("cifar", "mnist", "fmnist")
    :type data_type: str
    :return:
    :rtype:
    """

    # initialize temp values to all 1s
    if device.type == 'cpu':
        temp = values_target(size=generated_data.size(), value=1.0, cuda=False)
    else:
        temp = values_target(size=generated_data.size(), value=1.0, cuda=True)

    # mask values with low prediction
    mask = (prediction < 0.5).view(-1)
    indices = (mask.nonzero(as_tuple=False)).detach().cpu().numpy().flatten().tolist()

    data = generated_data[mask, :]

    if len(indices) > 1:
        if XAItype == "saliency":
            for i in range(len(indices)):
                explainer = Saliency(discriminator)
                temp[indices[i], :] = explainer.attribute(data[i, :].detach().unsqueeze(0))

        elif XAItype == "shap":
            for i in range(len(indices)):
                explainer = DeepLiftShap(discriminator)
                temp[indices[i], :] = explainer.attribute(data[i, :].detach().unsqueeze(0), trained_data, target=0)

        elif XAItype == "lime":
            explainer = lime_image.LimeImageExplainer()
            global discriminatorLime
            discriminatorLime = deepcopy(discriminator)
            discriminatorLime.cpu()
            discriminatorLime.eval()
            for i in range(len(indices)):
                if data_type == "cifar":
                    tmp = data[i, :].detach().cpu().numpy()
                    tmp = np.reshape(tmp, (32, 32, 3)).astype(np.double)
                    exp = explainer.explain_instance(tmp, batch_predict_cifar, num_samples=100)
                else:
                    tmp = data[i, :].squeeze().detach().cpu().numpy().astype(np.double)
                    exp = explainer.explain_instance(tmp, batch_predict, num_samples=100)
                _, mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=False, negative_only=False)
                temp[indices[i], :] = torch.tensor(mask.astype(np.float))
            del discriminatorLime
        else:
            raise Exception("wrong xAI type given")

    if device.type != 'cpu':
        temp = temp.cuda()
    set_values(normalize_vector(temp))


def get_explanation_dimg(generated_data, discriminator, prediction, XAItype="shap", cuda=True, trained_data=None,
                    labels=None, device=None, data_type="mnist") -> None:
    """
    This function calculates the explanation for given generated images using the desired xAI systems and the
    :param generated_data: data created by the generator
    :type generated_data: torch.Tensor
    :param discriminator: the discriminator model
    :type discriminator: torch.nn.Module
    :param prediction: tensor of predictions by the discriminator on the generated data
    :type prediction: torch.Tensor
    :param XAItype: the type of xAI system to use. One of ("shap", "lime", "saliency")
    :type XAItype: str
    :param cuda: whether to use gpu
    :type cuda: bool
    :param trained_data: a batch from the dataset
    :type trained_data: torch.Tensor
    :param data_type: the type of the dataset used. One of ("cifar", "mnist", "fmnist")
    :type data_type: str
    :return:
    :rtype:
    """

    # initialize temp values to all 1s
    if device.type == 'cpu':
        temp = values_target(size=generated_data.size(), value=1.0, cuda=False)
    else:
        temp = values_target(size=generated_data.size(), value=1.0, cuda=True)

    # mask values with low prediction
    mask = (prediction < 0.5).view(-1)
    indices = (mask.nonzero(as_tuple=False)).detach().cpu().numpy().flatten().tolist()

    data = generated_data[mask, :]
    labels = labels[mask, :]

    if len(indices) > 1:
        if XAItype == "saliency":
            for i in range(len(indices)):
                explainer = Saliency(discriminator)
                temp[indices[i], :] = explainer.attribute(data[i, :].detach().unsqueeze(0),
                                                          additional_forward_args=(labels[i, :].unsqueeze(0), device))

        elif XAItype == "shap":
            for i in range(len(indices)):
                explainer = DeepLiftShap(discriminator)
                temp[indices[i], :] = explainer.attribute(data[i, :].detach().unsqueeze(0),
                                                          additional_forward_args=labels[i, :].unsqueeze(0),
                                                          baselines=trained_data, target=0)

        elif XAItype == "lime":
            explainer = lime_image.LimeImageExplainer()
            global discriminatorLime
            discriminatorLime = deepcopy(discriminator)
            discriminatorLime.cpu()
            discriminatorLime.eval()
            for i in range(len(indices)):
                if data_type == "cifar":
                    tmp = data[i, :].detach().cpu().numpy()
                    tmp = np.reshape(tmp, (32, 32, 3)).astype(np.double)
                    exp = explainer.explain_instance(tmp, batch_predict_cifar, num_samples=100)
                else:
                    tmp = data[i, :].squeeze().detach().cpu().numpy().astype(np.double)
                    exp = explainer.explain_instance(tmp, batch_predict, num_samples=100)
                _, mask = exp.get_image_and_mask(exp.top_labels[0], positive_only=False, negative_only=False)
                temp[indices[i], :] = torch.tensor(mask.astype(np.float))
            del discriminatorLime
        else:
            raise Exception("wrong xAI type given")

    if device.type != 'cpu':
        temp = temp.cuda()
    set_values(normalize_vector(temp))

def explanation_hook_cifar(module, grad_input, grad_output):
    """
    This function creates explanation hook which is run every time the backward function of the gradients is called
    :param module: the name of the layer
    :param grad_input: the gradients from the input layer
    :param grad_output: the gradients from the output layer
    :return:
    """
    # import pydevd
    #
    # pydevd.settrace(suspend=False, trace_only_current_thread=True)

    # get stored mask
    temp = get_values()
    # print("XAI")
    # multiply with mask
    new_grad = grad_input[0] + 0.5 * (grad_input[0] * temp) # Adjust the parameter \theta (0.2 or 0.5)

    return (new_grad, )


def normalize_vector(vector: torch.tensor) -> torch.tensor:
    """ normalize np array to the range of [0,1] and returns as float32 values """
    vector -= vector.min()
    vector /= vector.max()
    vector[torch.isnan(vector)] = 0
    return vector.type(torch.float32)


def get_values() -> np.array:
    """ get global values """
    global values
    return values


def set_values(x: np.array) -> None:
    """ set global values """
    global values
    values = x


#  From vector_utils
def values_target(size: tuple, value: float, cuda: False) -> Variable:
    """ returns tensor filled with value of given size """
    result = Variable(full(size=size, fill_value=value))
    if cuda:
        result = result.cuda()
    return result


#  From explanation _utils
def batch_predict(images):
    """ function to use in lime xAI system for MNIST and FashionMNIST"""
    # convert images to greyscale
    images = np.mean(images, axis=3)
    # stack up all images
    batch = torch.stack([i for i in torch.Tensor(images)], dim=0)
    logits = discriminatorLime(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().numpy()


def batch_predict_cifar(images):
    """ function to use in lime xAI system for CIFAR10"""
    # stack up all images
    images = np.transpose(images, (0, 3, 1, 2))
    batch = torch.stack([i for i in torch.Tensor(images)], dim=0)
    logits = discriminatorLime(batch)
    probs = F.softmax(logits, dim=1).view(-1).unsqueeze(1)
    return probs.detach().numpy()
