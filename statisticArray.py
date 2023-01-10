import statistic
import numpy as np

originArray = np.load(r'/root/ankle/smallValid.npy')
zuwaiArray = np.load(r'/root/ankle/smallZuwai.npy')
alexArray = np.load(r'/root/ankle/alexValid.npy')
vgg11Array = np.load(r'/root/ankle/vgg11Valid.npy')
ssaArray = np.load(r'/root/ankle/SSAnoweightValid.npy')

def statisticArray(cm):
    accList = [statistic.acc(cm,i) for i in range(4)]
    recallList = [statistic.recall(cm,i) for i in range(4)]
    precisionList = [statistic.precision(cm,i) for i in range(4)]
    specificityList = [statistic.specificity(cm,i) for i in range(4)]
    f1List = statistic.micro_f1(cm)
    npvList = [statistic.npv(cm,i) for i in range(4)]
    print('cm:{}'.format(cm))
    print('acc:{}'.format(accList))
    print('recall:{}'.format(recallList))
    print('specificity:{}'.format(specificityList))
    print('micro_f1:{}'.format(f1List))
    print('precision:{}'.format(precisionList))
    print('npv:{}'.format(npvList))
# print('\nalexNet:')
# statisticArray(alexArray)
# print('\nvgg11:')
# statisticArray(vgg11Array)
# print('\nSSANet:')
# statisticArray(originArray)
# print('\nSSANet+weightLoss valid result:')
# statisticArray(ssaArray)
# print('\nSSANet+weightLoss zuwai result:')
# statisticArray(zuwaiArray)

