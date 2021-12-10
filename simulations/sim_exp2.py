# Header
import sys
import torch
import os
import torch.nn as nn
from torch.utils import model_zoo
import torchvision
import torchvision.transforms as transforms
import network.conv0 as conv0
import numpy as np
import errno
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import pandas
import gen_plots as myplt

def load_sin_model(model_name):

    model_urls = {
            'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
            'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
            'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
            'alexnet_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/0008049cd10f74a944c6d5e90d4639927f8620ae/alexnet_train_60_epochs_lr0.001-b4aa5238.pth.tar',
    }

    if "resnet50" in model_name:
        print("Using the ResNet50 architecture.")
        model = torchvision.models.resnet50(pretrained=False)
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = model_zoo.load_url(model_urls[model_name])
    elif "vgg16" in model_name:
        print("Using the VGG-16 architecture.")
       
        # download model from URL manually and save to desired location
        filepath = "./vgg16_train_60_epochs_lr0.01-6c6fcc9f.pth.tar"

        assert os.path.exists(filepath), "Please download the VGG model yourself from the following link and save it locally: https://drive.google.com/drive/folders/1A0vUWyU6fTuc-xWgwQQeBvzbwi6geYQK (too large to be downloaded automatically like the other models)"

        model = torchvision.models.vgg16(pretrained=False)
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        checkpoint = torch.load(filepath)

    elif "alexnet" in model_name:
        print("Using the AlexNet architecture.")
        model = torchvision.models.alexnet(pretrained=False)
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
        checkpoint = model_zoo.load_url(model_urls[model_name])
    else:
        raise ValueError("unknown model architecture.")

    model.load_state_dict(checkpoint["state_dict"])
    return model

# Parameters
regime = 'test' # 'train' / 'retrain' / 'test'
NEP = 5 #5 # number of training epochs
model_name = 'alexnet' # 'vgg' / 'alexnet' / 'resnet50_trained_on_SIN'
test_type = 'all_rot' # 'no_rot' (no rotations are trained), 'some_rot' (some rotations are trained), 'all_rot' (all rotations on some categories)
if test_type == 'all_rot':
    test_category = ['3'] # the category to test (should match the category that was left out).
usetrained = True # whether or not to use the pretrained model
learnconv = True # False
create_df = True # whether to create pandas dataframe for analysing results
load_old_results = False # when testing, whether to generate figures based on previously stored results

### Train with basis images & test with v1, v2
if test_type == 'no_rot':
    BASE_PATH = './data/isostim/' # Training set without rotation & scale var
elif test_type == 'some_rot':
    BASE_PATH = './data/isostim_train_rots/' # Training set with rotations [-45,0] on all categories
elif test_type == 'all_rot':
    BASE_PATH = './data/isostim_teach_rot/' # Training set with all rotations on subset of categories
path_train = BASE_PATH + 'train'
path_test = BASE_PATH + 'testgrid'
path_test_cv = BASE_PATH + 'test_cv'
if learnconv is True:
    results_dir = BASE_PATH + 'results/learnall/'
    model_file = BASE_PATH + 'train/' + model_name + '_trained_model_learnall.pt'
else:
    results_dir = BASE_PATH + 'results/onlyclass/'
    model_file = BASE_PATH + 'train/' + model_name + '_trained_model_onlyclass.pt'

if regime == 'train' or regime == 'retrain':
    training = True
else:
    training = False
my_trans = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])
    #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

### Train the network
if regime == 'train' or regime == 'retrain':
    train_data = torchvision.datasets.ImageFolder(root=path_train,
                                                transform=my_trans)
    train_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=32,
                                            shuffle=True,
                                            num_workers=2)
    cv_data = torchvision.datasets.ImageFolder(root=path_test_cv,
                                                transform=my_trans)
    cv_loader = torch.utils.data.DataLoader(cv_data,
                                            batch_size=32,
                                            shuffle=True,
                                            num_workers=2)

    if regime == 'retrain':
        model = torch.load(model_file)
    elif regime == 'train':
        nclasses = len(sorted(os.listdir(path_train)))

        ### Load a pretrained model
        if model_name == 'vgg':
            model = torchvision.models.vgg16(pretrained=usetrained)
        elif model_name == 'alexnet':
            model = torchvision.models.alexnet(pretrained=usetrained)
        elif model_name == 'resnet':
            model = torchvision.models.resnet50(pretrained=usetrained)
        elif model_name == 'resnet50_trained_on_SIN':
            model = load_sin_model(model_name)

        ### Freeze convolution layers if flag is set
        if usetrained is True and learnconv is False:
            for param in model.parameters():
                param.requires_grad = False

        ### Define a new output layer / classifier based on number of classes
        if model_name == 'vgg' or model_name == 'vgg_bn':
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
        elif model_name == 'resnet':
            num_feat = model.fc.in_features
            model.fc = nn.Linear(num_feat, nclasses)
        elif model_name == 'resnet50_trained_on_SIN':
            num_feat = model.module.fc.in_features
            model.module.fc = nn.Linear(num_feat, nclasses)

    ### Define loss and grad descent method
    criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5, weight_decay=1e-3)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 1e-5, weight_decay=1e-3)

    conv0.train(model, train_loader, cv_loader, criterion, optimizer, NEP)

    torch.save(model, model_file)
    print('Finished training')
else: # Test
    confusion_file = os.path.join(results_dir, 'confusion_matrix.npy')
    results_file = os.path.join(results_dir, 'results_df.pickle')
    cats = sorted(os.listdir(path_test))
    if load_old_results == True:
        confusion = np.load(confusion_file) # save
        results_df = pandas.read_pickle(results_file)
    else:
        results_df = pandas.DataFrame()
        nsh = len(os.listdir(path_test + '/1')) # get number of shears from first category
        confusion = [] # a 2D confusion matrix, each element contains a numpy array for testgrid perf
        for li, label in enumerate(cats):
            label_list = []
            for pi, prediction in enumerate(cats):
                label_list.append(np.reshape(np.zeros(pow(nsh, 2)), (nsh, nsh)))  # init confusion matrix
            confusion.append(label_list)
        for cix, cat in enumerate(cats):
            shears = sorted(os.listdir(path_test + '/' + cat),
                            key=lambda x: int(x.split('_')[1]))
            nsh = len(shears)
            accu = np.reshape(np.zeros(pow(nsh,2)), (nsh,nsh)) # matrix of accuracies
            for shix, shear in enumerate(shears):
                dists = sorted(os.listdir(path_test + '/' + cat + '/' + shear),
                            key=lambda x: int(x.split('_')[1]))
                for dix, dist in enumerate(dists):
                    data_path = path_test + '/' + cat + '/' + shear + '/' + dist
                    test_data = torchvision.datasets.ImageFolder(root=data_path,
                                                                transform=my_trans)
                    test_loader = torch.utils.data.DataLoader(test_data,
                                                            batch_size=32,
                                                            shuffle=True,
                                                            num_workers=2)
                    output = conv0.test(model_file, test_loader, data_path)
                    choices = output[0]
                    if create_df == True: # Create a dataframe with results of each trial
                        batch_results = output[2]
                        batch_shear = [shear] * len(batch_results) # list of repeated values of shear
                        batch_dist = [dist] * len(batch_results) # list of repeated values of dist
                        batch_cat = [cat] * len(batch_results) # list of repeated values of cat
                        batch_df = pandas.DataFrame(batch_results, columns=['correct'])
                        batch_df['shear'] = batch_shear
                        batch_df['rotation'] = batch_dist
                        batch_df['category'] = batch_cat
                        results_df = results_df.append(batch_df, ignore_index=True)
                    for pi, prediction in enumerate(cats):
                        confusion[cix][pi][dix][shix] = choices[pi]
        results_df.to_pickle(results_file)
        np.save(confusion_file, confusion)

    ncats = len(cats)
    plt.rcParams['font.size'] = '16'
    fig, axs = plt.subplots(ncats, ncats, dpi=200)
    for li, label in enumerate(cats):
        for pi, prediction in enumerate(cats):
            maski =  np.rot90(np.tri(confusion[li][pi].shape[0], k=-1))
            Ai = np.ma.array(confusion[li][pi], mask=maski)
            cmapi = cm.get_cmap('viridis', 100)
            cmapi.set_bad('gray')
            plt.imshow(Ai, interpolation='nearest', cmap=cmapi, vmin=0, vmax=100)
            im = axs[li, pi].imshow(Ai, interpolation='nearest', cmap=cmapi, vmin=0, vmax=100)
            # im = axs[li, pi].imshow(confusion[li][pi], vmin=0, vmax=100)
            if li == 0:
                axs[li, pi].text(3.5, -2, '{0}'.format(str(pi)), ha='center')
            if pi == 0:
                axs[li, pi].text(-4, 3.5, '{0}'.format(str(li)), va='center', rotation='vertical')
            axs[li, pi].set_yticklabels([])
            axs[li, pi].set_xticklabels([])
            axs[li, pi].set_yticks([])
            axs[li, pi].set_xticks([])

    fig.text(0.5, 0.04, 'Category predicted by CNN', ha='center')
    fig.text(0.08, 0.3, 'Category of Basis shape', ha='center', rotation='vertical')

    ### Place colorbar on it's own axis using subplots_adjust
    fig.colorbar(im, ax=axs.ravel().tolist())

    filename = results_dir + model_name + '_confusion_matrix.png'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    fig.savefig(filename)
    plt.clf()

    ### Plot accuracies for each category (diagonal elements of confusion)
    for li, label in enumerate(cats):
        mask =  np.rot90(np.tri(confusion[li][li].shape[0], k=-1))
        A = np.ma.array(confusion[li][li], mask=mask)
        cmap = cm.get_cmap('viridis', 100)
        cmap.set_bad('gray')
        plt.imshow(A, interpolation='nearest', cmap=cmap, vmin=0, vmax=100)
        plt.tick_params(
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            left=False,
            right=False,
            pad=0,
            labelbottom=False) # labels along the bottom edge are off
        plt.xlabel('Relational Distance')
        plt.ylabel('Coordinate Distance')
        plt.colorbar()
        filei = results_dir + model_name + '_accuracy_' + label + '.png'
        if not os.path.exists(os.path.dirname(filei)):
            try:
                os.makedirs(os.path.dirname(filei))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        plt.savefig(filei)
        plt.clf()

    if test_type == 'all_rot':
        results_df = results_df[results_df['category'].isin(test_category)]
    myplt.plot_sim_results(results_df, test_type, model_name)