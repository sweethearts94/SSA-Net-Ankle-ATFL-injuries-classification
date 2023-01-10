import os
import time
import torch
import numpy as np
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.backends import cudnn
from torchvision.datasets import ImageFolder
from torchvision.transforms import *
from torch.autograd import Variable
import models
import statistic
from sklearn.metrics import accuracy_score, average_precision_score,precision_score,f1_score,recall_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
best_acc = 0
n_epochs = 200
batch_size = 128
learning_rate = 1e-3
# weight = [1.2,1.1,1,1.5,1,1.2,1,1]
weight = [1,1,1,1,1,1,1,1]
dataset = ''
save = ""
os.makedirs(os.path.join(save,'cm_npy'),exist_ok=True)
os.makedirs(os.path.join(save,'result'),exist_ok=True)
os.makedirs(os.path.join(save,'models'),exist_ok=True)
# Data transforms
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=(0.8,1.8)),
    torchvision.transforms.ToTensor(),
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.ToTensor(),
])

train_set = ImageFolder(os.path.join(dataset,'train'), transform=train_transform)
valid_set = ImageFolder(os.path.join(dataset,'valid'), transform=test_transform)
test_set = ImageFolder(os.path.join(dataset,'test'), transform=test_transform)

train_set = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
valid_set = torch.utils.data.DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_set = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=0)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        args(val(value),n=1(number))
        val = val,sum += val*n,count+=n,avg = sum/count
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(model, loader, optimizer, epoch, n_epochs, print_freq=100):
    """
    train stage
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    accu = AverageMeter()
    labelsList,predList = [],[]

    # Model on train mode
    model.train()
    end = time.time()
    for batch_idx, (images,labels) in enumerate(loader):
        images,labels = images.to(device),labels.to(device)
        
        images = Variable(images)
        labels = Variable(labels)
        outputs = model(images)
        loss = F.cross_entropy(weight=torch.Tensor(weight).to(device),input=outputs,target=labels).to(device)
        pred = outputs.max(1, keepdim=True)[1]
        labelsList += labels.tolist()
        predList += pred.tolist()

        # measure accuracy and record loss
        batch_size = labels.size(0)
        accu.update(val=pred.eq(labels.view_as(pred)).sum().item() / batch_size, n=batch_size)
        losses.update(val=loss.item() / batch_size, n=batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Acc %.4f (%.4f)' % (accu.val, accu.avg),
                'LR %f '% optimizer.state_dict()['param_groups'][0]['lr'],
            ])
            print(res)
    cm = statistic.confusion_matrix(torch.Tensor(labelsList),torch.Tensor(predList).squeeze(),8)
    precisionList = [statistic.precision(cm,i) for i in range(8)]
    npvList = [statistic.npv(cm,i) for i in range(8)]
    ratioList = statistic.ratio(cm)
    rateList = statistic.percentage(cm)
    np.save(os.path.join(save,'cm_npy',str(epoch+1)+'train'+str(accu.avg)+'.npy'),cm)
    return losses.avg,cm,accu.avg,precisionList,npvList,ratioList,rateList

def test_epoch(model,epoch, loader, is_test=True):
    """
    test stage
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    accu = AverageMeter()

    # Model on eval mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        labelsList,predList = [],[]
        for batch_idx, (images,labels) in enumerate(loader):
            images, labels = images.to(device),labels.to(device)
            images = Variable(images)
            labels = Variable(labels)

            # compute output
            outputs = model(images)
            pred = outputs.max(1, keepdim=True)[1]
            labelsList += labels.tolist() 
            predList += pred.tolist()
            loss = F.cross_entropy(weight=torch.Tensor(weight).to(device),input=outputs,target=labels).to(device)
            # measure accuracy and record loss
            batch_size = labels.size(0)
            accu.update(pred.eq(labels.view_as(pred)).sum().item() / batch_size, batch_size)
            losses.update(loss.item() / batch_size, batch_size)
            batch_time.update(time.time() - end)
            end = time.time()

        cm = statistic.confusion_matrix(torch.Tensor(labelsList),torch.Tensor(predList).squeeze(),8)
        precisionList = [statistic.precision(cm,i) for i in range(8)]
        npvList = [statistic.npv(cm,i) for i in range(8)]
        ratioList = statistic.ratio(cm)
        rateList = statistic.percentage(cm)
        np.save(os.path.join(save,'cm_npy',str(epoch+1)+'test'+str(accu.avg)+'.npy'),cm)
    return losses.avg,cm,accu.avg,precisionList,npvList,ratioList,rateList

def train(model, train_set, valid_set, save, n_epochs=200,
          lr=learning_rate, wd=0.0001, momentum=0.95, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    model = model.to(device)
    
    # InitWeight
    def initWeights(module):
        if type(module) == nn.Conv2d and module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    model.apply(initWeights)

    model_wrapper = model
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    record_full_file = open(os.path.join(save, 'result','acc_best.txt'), 'w')
    print("Total parameters: {},model:{}".format(num_params,model),file=record_full_file)
    record_full_file.close()
    # Optimizer
    # optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, weight_decay=wd) 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[0.25 * n_epochs, 0.5 * n_epochs,0.75 * n_epochs],
                                                     gamma=0.1)

    # Start log
    with open(os.path.join(save, 'result','results.txt'), 'w') as f:
        f.write('epoch,train_loss,train_acc,valid_loss,valid_acc,test_acc,lr\n')

    # Train model
    best_acc = 0
    for epoch in range(n_epochs):
        train_loss,train_cm,train_acc,train_precisionList,train_npvList,train_ratioList,train_rateList = train_epoch(
            model=model_wrapper,
            loader=train_set,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
        )
        scheduler.step()
        valid_loss,valid_cm,valid_acc,valid_precisionList,valid_npvList,valid_ratioList,valid_rateList = test_epoch(
            model=model_wrapper,
            epoch=epoch,
            loader=valid_set,
            is_test=(not valid_set)
        )
        # Determine if model is the best
        if valid_set:
            if valid_acc > best_acc:
                best_acc = valid_acc
                print('New best acc: %.4f' % best_acc)
                record_file = open(os.path.join(save,'result', 'acc_best.txt'), 'a')
                print('{}, {:.5f}'.format(epoch+1,best_acc),file=record_file)
                record_file.close()
                torch.save(model.state_dict(), os.path.join(save,'models', 'best_model.pkl'))
        torch.save(model.state_dict(), os.path.join(save,'models', '{}_acc{:.5f}_model.pkl'.format(epoch+1,best_acc)))

        # Log results
        record_lossacc =  open(os.path.join(save, 'result','results.txt'), 'a')
        print((epoch + 1), train_loss, train_acc, valid_loss, valid_acc,optimizer.state_dict()['param_groups'][0]['lr'],file=record_lossacc)
        record_lossacc.close()
        record_other= open(os.path.join(save,'result', 'cm_acc_prec_npv.txt'), 'a')
        print((epoch+1),train_cm,'\n','\n',train_precisionList,'\n',train_npvList,'\n',train_ratioList,'\n',train_rateList,'\n',
        valid_cm,'\n',valid_precisionList,'\n',valid_npvList,'\n',valid_ratioList,'\n',valid_rateList,'\n',
        file=record_other)
        record_other.close()

def test(model,test_set):
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(save,'models', 'best_model.pkl')))
    test_results = test_epoch(
        model=model,
        epoch=n_epochs+1,
        loader=test_set,
        is_test=True
    )
    test_loss,test_cm,test_acc,test_precisionList,test_npvList,test_ratioList,test_rateList = test_results
    record_other= open(os.path.join(save,'result', 'cm_acc_prec_npv.txt'), 'a')
    print('test result:',test_loss,'\n',test_cm,'\n','\n',test_precisionList,'\n',test_npvList,'\n',test_ratioList,'\n',test_rateList,file=record_other)
    record_other.close()
    with open(os.path.join(save,'result', 'results.txt'), 'a') as f:
        f.write(',,,,,%0.5f\n' % test_acc)
    print('Final test acc: %.4f' % test_acc)

def main(seed=None):
    # model = ankleNet()
    model = models.SE234567_VGG11_bn()
    cudnn.benchmark = True  # 设置卷积神经网络加速

    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)

    # Train the model
    train(model=model, train_set=train_set, valid_set=valid_set, save=save, n_epochs=n_epochs, seed=seed)
    test(model=model, test_set=test_set)
    
    print('Done!')

if __name__ == '__main__':
    main()
