# Header
import sys, os
from unittest import result
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import network.conv0 as conv0
import pandas
# import pdb
import gen_plots as myplt

regime = 'test' # 'train' / 'test'
model_name = 'vgg' # 'resnet', 'vgg', 'alexnet'
NEP = 5 # number of training epochs
usetrained = True # Use a pretrained network
learnconv = True # Enable learning in convolution layers

expt_dict = {
    1: {'expt':1, 'path':'./data/no_rot_scale_trans/', 'title':'No Augment'},
    2: {'expt':2, 'path':'./data/no_rot_scale/', 'title':'Translation'},
    3: {'expt':3, 'path':'./data/orig/', 'title':'Rot+Scale+Trans'},
    4: {'expt':4, 'path':'./data/teach_v1_new/', 'title':'Teach Relations'},
    5: {'expt':5, 'path':'./data/teach_v1_exclude_exact/', 'title':'Exclude exact'},
    6: {'expt':6, 'path':'./data/teach_v1_exclude_left/', 'title':'Exclude left'},
}

experiments = [1, 2, 3] # List of experiments (from expt_dict) to run
results = pandas.DataFrame() ### Initialize DataFrame
for expt in experiments:
    BASE_PATH = expt_dict[expt]['path']
    path_train = BASE_PATH + 'train'
    path_test_1 = BASE_PATH + 'test_b'
    path_test_2 = BASE_PATH + 'test_v1'
    path_test_3 = BASE_PATH + 'test_v2'
    path_test_cv = BASE_PATH + 'test_cv'
    model_file = BASE_PATH + 'train/trained_model_' + model_name + '.pt'

    if regime == 'train':
        training = True
    else:
        training = False
    my_trans = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ### Train the network
    if regime == 'train' or regime == 'retrain':
        train_data = torchvision.datasets.ImageFolder(root=path_train,
                                                    transform=my_trans)
        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=64,
                                                shuffle=True,
                                                num_workers=2)
        cv_data = torchvision.datasets.ImageFolder(root=path_test_cv,
                                                    transform=my_trans)
        cv_loader = torch.utils.data.DataLoader(cv_data,
                                                batch_size=64,
                                                shuffle=True,
                                                num_workers=2)

        if regime == 'retrain':
            model = torch.load(model_file)
        elif regime == 'train':
            nclasses = len(sorted(os.listdir(path_train)))
            if model_name == 'vgg':
                model = torchvision.models.vgg16(pretrained=usetrained)
            elif model_name == 'resnet':
                model = torchvision.models.resnet18(pretrained=usetrained)
            elif model_name == 'alexnet':
                model = torchvision.models.alexnet(pretrained=usetrained)
            if usetrained is True and learnconv is False:
                for param in model.parameters():
                    param.requires_grad = False
            if model_name == 'vgg':
                # model.classifier = nn.Sequential(
                #         nn.Linear(512*7*7, 4096),
                #         nn.ReLU(True),
                #         nn.Dropout(),
                #         nn.Linear(4096, 4096),
                #         nn.ReLU(True),
                #         nn.Dropout(),
                #         nn.Linear(4096, nclasses))
                num_feat = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_feat, nclasses)
            elif model_name == 'resnet':
                num_feat = model.fc.in_features
                model.fc = nn.Linear(num_feat, nclasses)
            elif model_name == 'alexnet':
                num_feat = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_feat, nclasses)

        ### Define loss and grad descent method
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5, weight_decay=1e-3)

        print("Training {} on {}".format(model_name, path_train), end="\n")
        conv0.train(model, train_loader, cv_loader, criterion, optimizer, NEP)

        torch.save(model, model_file)
        print('Finished training\n')
    else: # Test
        test_data_1 = torchvision.datasets.ImageFolder(root=path_test_1,
                                                    transform=my_trans)
        test_loader_1 = torch.utils.data.DataLoader(test_data_1,
                                                batch_size=64,
                                                shuffle=True,
                                                num_workers=2)
        test_data_2 = torchvision.datasets.ImageFolder(root=path_test_2,
                                                    transform=my_trans)
        test_loader_2 = torch.utils.data.DataLoader(test_data_2,
                                                batch_size=64,
                                                shuffle=True,
                                                num_workers=2)
        test_data_3 = torchvision.datasets.ImageFolder(root=path_test_3,
                                                    transform=my_trans)
        test_loader_3 = torch.utils.data.DataLoader(test_data_3,
                                                batch_size=64,
                                                shuffle=True,
                                                num_workers=2)
    #     test_data_4 = torchvision.datasets.ImageFolder(root=path_test_4,
    #                                                 transform=my_trans)
    #     test_loader_4 = torch.utils.data.DataLoader(test_data_4,
    #                                             batch_size=64,
    #                                             shuffle=True,
    #                                             num_workers=2)

        results_basis = {} # A dictionary for the current experiment that will be added to results Dataframe
        results_basis['expt'] = expt_dict[expt]['expt']
        results_basis['title'] = expt_dict[expt]['title']
        results_basis['condition'] = 'basis'
        results_basis['avg_accu'] = conv0.test(model_file, test_loader_1, path_test_1)[1]['avg_accu']
        results = results.append(results_basis, ignore_index=True)

        results_v1 = {} # A dictionary for the current experiment that will be added to results Dataframe
        results_v1['expt'] = expt_dict[expt]['expt']
        results_v1['title'] = expt_dict[expt]['title']
        results_v1['condition'] = 'v1'
        results_v1['avg_accu'] = conv0.test(model_file, test_loader_2, path_test_2)[1]['avg_accu']
        results = results.append(results_v1, ignore_index=True)

        results_v2 = {} # A dictionary for the current experiment that will be added to results Dataframe
        results_v2['expt'] = expt_dict[expt]['expt']
        results_v2['title'] = expt_dict[expt]['title']
        results_v2['condition'] = 'v2'
        results_v2['avg_accu'] = conv0.test(model_file, test_loader_3, path_test_3)[1]['avg_accu']
        results = results.append(results_v2, ignore_index=True)

        # conv0.test(model_file, test_loader_4, path_test_4)

        # print("Accuracies:\nBasis={}%, V1={}%, V2={}%".format(results_1['avg_accu'], results_2['avg_accu'], results_3['avg_accu']))
if regime == 'test':
    # print(results)
    myplt.plot_multpile_exp(results=results, model_name=model_name)