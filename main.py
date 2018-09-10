import time

import torch
from torch.autograd import Variable

from dataset import TestDataset
from dataset import TrainDataset
from option import Options
from style_transfer_test import style_transfer


def train(args):
    


def test(args):
    # Data loading code
    dataset = TestDataset(args.contentPath, args.stylePath, args.fineSize)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    avgTime = 0
    cImg = torch.Tensor()
    sImg = torch.Tensor()
    csF = torch.Tensor()
    csF = Variable(csF)
    if (args.cuda):
        cImg = cImg.cuda(args.gpu)
        sImg = sImg.cuda(args.gpu)
        csF = csF.cuda(args.gpu)

    for i, (contentImg, styleImg, imname) in enumerate(loader):
        imname = imname[0]
        print('Transferring ' + imname)
        with torch.no_grad():
            cImg = torch.tensor(contentImg)
            sImg = torch.tensor(styleImg)
        if (args.cuda):
            cImg = cImg.cuda(args.gpu)
            sImg = sImg.cuda(args.gpu)
        start_time = time.time()
        # WCT Style Transfer
        style_transfer(cImg, sImg, imname, csF)
        end_time = time.time()
        print('Elapsed time is: %f' % (end_time - start_time))
        avgTime += (end_time - start_time)

    print('Processed %d images. Averaged time is %f' % ((i+1), avgTime/(i+1)))


def main():
    args = Options().parse()
    if args.subcommand == 'train':
        train(args)
    elif args.subcommand == 'test':
        test(args)
    else:
        raise ValueError("Experiment type error")
