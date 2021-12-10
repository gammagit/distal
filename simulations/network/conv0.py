# Header
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

### Define the network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        ### 3 input image channel, 24 output channels, 5x5 square convolution
        ### kernel
        self.conv1 = nn.Conv2d(3, 24, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(24, 32, 5)
        self.conv2_drop = nn.Dropout2d()
        self.conv2_bn = nn.BatchNorm2d(32)
        ### an affine operation: y = Wx + b
        self.fc1 = nn.Linear(32 * 11 * 11, 120)
        self.fc1_bn = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc2_bn = nn.BatchNorm1d(84)
#         self.fc3 = nn.Linear(84, 16) # For training with CIFAR-10
#         self.fc3 = nn.Linear(84, 6) # For training with original HS96
        self.fc3 = nn.Linear(84, 12) # For training with Teach_V1

    def forward(self, x):
        ### Max pooling over a (2, 2) window
        x = self.pool(F.relu(self.conv1(x)))
        ### If the size is a square you can only specify a single number
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1_bn(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def cross_validate(model, loader, device):
    """
    """
    model.eval()

    # Test on data set 1
    correct = 0
    total = 0
    for data in loader:
        images, labels = data
        images = Variable(images)
        images, labels = images.to(device), labels.to(device) # send to GPU
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Validation Accuracy: %d %%\n' % (100 * correct / total))


def train(model, train_loader, cv_loader, criterion, optimizer, num_epochs):

    ### Check if GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device) # send to GPU

    for epoch in range(num_epochs):
        model.train() # make sure it's in training env (esp after validation)
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device) # send to GPU

            ### initialise
            optimizer.zero_grad()

            ### feedforward
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()

            ### adjust weights
            optimizer.step()

            ### print stats
            running_loss += loss.item()
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()
            if i % 20 == 19:
                print('[%d/%d, %5d] loss: %.3f; Acc: %d %%' %
                      (epoch + 1, num_epochs, i + 1, running_loss / 20,
                       100 * running_correct / running_total))
                running_loss = 0.0
                running_correct = 0
                running_total = 0

        ### perform cross-validation
        cross_validate(model, cv_loader, device)


def test(model_file, data_loader, data_path):
    """ Runs the model on the test data set and prints metrics

    Args:

    """
    model = torch.load(model_file)

    ### Check if GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device) # send to GPU

    ### Set evaluation mode
    model.eval() # Required for Dropout, Batch Norm

    ### Initialise variable to record accuracy
    classes = sorted(os.listdir(data_path))
    nc = len(classes)
    correct = 0
    total = 0
    score_lab_sum = 0 # total score for output that corresponds to label
    score_cum_var = 0 # variance in score_lab_sum
    nvars = 1 # number of times variances added in the loop below
    score_total = 0 # total score for all outputs
    score_max = 0 # total score for largest output
    class_correct = list(0. for i in range(nc))
    class_total = list(0. for i in range(nc))
    score = list(0. for i in range(nc))
    label_count = list(0. for i in range(nc))
    total_prob = list(0. for i in range(nc))
    choices = list(0. for i in range(nc)) # vector containing count of times each label was chosen
    accuracies = list(0. for i in range(nc)) # vector containing accuracy for each class
    trial_results = [] # list of booleans containing the results (correct/incorrect) for each trial

    #### Perform test and update accuracy variables
    for data in data_loader:
        images, labels = data
        images = Variable(images)
        images, labels = images.to(device), labels.to(device) # send to GPU
        outputs = model(images)
        max_outputs, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        trial_results += list((predicted == labels).cpu().numpy().astype(int))
        ### Update score based on the column that corresponds to label
        score_lab_sum += outputs.gather(1, labels.view(-1,1)).sum().item()
        score_cum_var += outputs.gather(1, labels.view(-1,1)).var().item()
        nvars += 1 # used for dividing score_cum_var to get  "average" var
        score_total += outputs.sum().item()
        score_max += max_outputs.sum().item()
        clist = (predicted == labels).squeeze()
        clist = clist.data.cpu().numpy() # tensor to uint8 (for pytorch>0.3)
        for ii in range(labels.size()[0]):
            label = labels[ii]
            class_correct[label] += clist[ii]
            class_total[label] += 1
            prediction = predicted[ii]
            choices[prediction] += 1
        for jj in range(nc):
            ### Collect score data for label
            label = jj
            outputs_incats = outputs.gather(1, labels.view(-1,1)).squeeze() # scores for input categories
            maskjj = (labels.view(-1,1).squeeze() == label) # binary vec containing 1s where labels = jj
            maskjj_cuda = maskjj.to(device, dtype=torch.float32)
            score[jj] += torch.dot(outputs_incats, maskjj_cuda).item()
            label_count[jj] += maskjj.sum().item()

            ### Collect probability data for each label
            ### Note that, in contrast to score this considers outputs of other categories for an input
            outputs_probs = F.softmax(outputs, dim=1) # prob of all categories
            outputs_probs_incats = outputs_probs.gather(1, labels.view(-1,1)).squeeze() # input cats
            total_prob[jj] += torch.dot(outputs_probs_incats, maskjj_cuda).item()

    ### Convert to percentage
    for kk in range(nc):
        choices[kk] = 100 * choices[kk] / total

    results = {} # A dictionary containing all results
    results['avg_accu'] = 100 * correct / total # accuracy across classes
    for jj in range(nc):
        if class_total[jj] == 0:
            results['accu_' + classes[jj]] = 0
        else:
            results['accu_'+classes[jj]] = 100 * class_correct[jj] / class_total[jj]
        if label_count[jj] == 0:
            results['score_' + classes[jj]] = 0
            results['prob_'+classes[jj]] = 0
        else:
            results['score_'+classes[jj]] = score[jj] / label_count[jj]
            results['prob_'+classes[jj]] = total_prob[jj] / label_count[jj]

    print("Testing: {0}          ".format(data_path), end="\r")
    # print('\nTesting on dataset: ' + data_path)
    # print('Overall Accuracy: %d %%' % (100 * correct / total))
    # print('Avg score for Labeled output: %f' % (score_lab_sum / total))
    # print('Var in score for Labeled output: %f' % (score_cum_var / nvars))
    # print('Avg score for All outputs: %f' % (score_total / total))
    # print('Avg score for Max outputs: %f' % (score_max / total))
    # for jj in range(nc):
    #     print('Class %3s ... Accuracy: %4d %%, Avg score: %5.3f, Avg prob (softmax): %6.4f' % (
    #         classes[jj], 100 * class_correct[jj] / class_total[jj],
    #         score[jj] / label_count[jj],
    #         total_prob[jj] / label_count[jj]))

    # return 100 * correct / total
    return (choices, results, trial_results)
