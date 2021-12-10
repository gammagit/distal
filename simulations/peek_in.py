import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


class SaveEmbeddings:
    '''
    Defines a function hook that can be passed to model.modules.register_forward_hook()
    and saves the output of this module (Note: output of a module will contain a tensor for each feature map)
    Note: a hook does not necessarily need to be a class, but defining it this way helps with stateful closures
    (see Effective Python, Item 23)
    '''
    def __init__(self):
        self.embeddings = [] # A list of embeddings, each of which will be a tensor
        self.layer_names = [] # Layer names that correspond to the above embeddings

    def __call__(self, module, module_in, module_out):
        self.embeddings.append(module_out) # when this hook is called (during forward pass), save module_out to list
        self.layer_names.append(module.__class__.__name__) # Save layer name

    def clear(self):
        self.embeddings = []


def hook_embeddings(arg_model):
    '''
    Returns an object of type SaveEmbeddings hooked to all layers of arg_models.
    Specifically, out_acts.__call__() is the function hook that is passed to arg_model.layers.register_forward_hook()
        out_acts: an object of type SaveEmbeddings
        arg_model: torchvision model
    '''
    out_acts = SaveEmbeddings() # An object containing a function hook that can be passed to register_forward_hook()
    out_handles = [] # Will contain the handles for all the hooks so that the copy of embeddings can be detached after forward pass

    for layer in arg_model.modules():
        if isinstance(layer, torch.nn.modules.conv.Conv2d) or\
            isinstance(layer, torch.nn.AdaptiveAvgPool2d) or\
            isinstance(layer, torch.nn.Linear): # Save outputs of convolution layers
            handle = layer.register_forward_hook(out_acts) # Note all layers are linked to the same hook (out_acts), so will update out_acts.embeddings
            out_handles.append(handle)

    return (out_acts, out_handles)

    
def stim_from_pair(arg_imp, device):
    my_trans = transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])
    im1 = Image.open(arg_imp[0])
    im2 = Image.open(arg_imp[1])
    out_stim1 = my_trans(im1).unsqueeze(dim=0).to(device)
    out_stim2 = my_trans(im2).unsqueeze(dim=0).to(device)

    return (out_stim1, out_stim2)

def visualise_embedding(arg_embeddings, arg_layer):
    '''
    Plots a 2D array of activations, given a list of embeddings (one for each layer) and the index of layer
    Note each layer consists of array of feature maps, therefore the embeddings are displayed as a 2D array.
    '''
    emii = arg_embeddings.embeddings[arg_layer].cpu().detach().squeeze()
    nrows = int(np.sqrt(emii.size(0)))
    ncols = int(np.sqrt(emii.size(0)))
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0.1, wspace=0.1)
    idx = 0
    for row in axarr:
        for col in row:
            col.imshow(emii[idx])
            col.axis('off')
            idx += 1


def get_cosine_dists(arg_em1, arg_em2):
    '''
    Finds the cosine distance between two array of embeddings (each array will consist of a sequence of embeddings for one network)
    '''
    nlayers = len(arg_em1.embeddings)
    cos_sims = []

    # Compute cosine distance between two embeddings
    for layeri in range(nlayers):
        e1 = arg_em1.embeddings[layeri].flatten(start_dim=1).detach().cpu() # Flatten to vectorise each fmap and concatenate fmaps in each layer
        e2 = arg_em2.embeddings[layeri].flatten(start_dim=1).detach().cpu()
        cos_simi = torch.nn.functional.cosine_similarity(e1, e2, eps=1e-6).item()
        if 'Pool' in arg_em1.layer_names[layeri]:
            lname = 'GPool'
        else:
            lname = arg_em1.layer_names[layeri]
        name_layeri = '{0}: {1}'.format(layeri, lname)
        cos_sims.append((name_layeri, cos_simi))
    
    return cos_sims