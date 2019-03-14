import time
import os
import random
from pathlib import Path


from tqdm import tqdm
import numpy as np
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from torch.utils.serialization import load_lua
from torch.optim import Adam

from encoder_decoder import (Encoder1, Encoder2, Encoder3, Encoder4, Encoder5,
                             Decoder1, Decoder2, Decoder3, Decoder4, Decoder5)
from dataset import TestDataset
from dataset import TrainDataset
from option import Options
from utils import write_event, save_model
from wct import style_transfer


def train(args):
    relu_targets = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    print("Training decoder for relu_target:", args.relu_target)
    relu_target_id = relu_targets.index(args.relu_target) + 1

    if relu_target_id == 1:
        vgg = load_lua(args.vgg1)
        encoder = Encoder1(vgg)
        decoder = Decoder1()
        epochs = args.d1_epochs
        batch_size = args.d1_batch_size
    elif relu_target_id == 2:
        vgg = load_lua(args.vgg2)
        encoder = Encoder2(vgg)
        decoder = Decoder2()
        epochs = args.d2_epochs
        batch_size = args.d2_batch_size
    elif relu_target_id == 3:
        vgg = load_lua(args.vgg3)
        encoder = Encoder3(vgg)
        decoder = Decoder3()
        epochs = args.d3_epochs
        batch_size = args.d3_batch_size
    elif relu_target_id == 4:
        vgg = load_lua(args.vgg4)
        encoder = Encoder4(vgg)
        decoder = Decoder4()
        epochs = args.d4_epochs
        batch_size = args.d4_batch_size
    elif relu_target_id == 5:
        vgg = load_lua(args.vgg5)
        encoder = Encoder5(vgg)
        decoder = Decoder5()
        epochs = args.d5_epochs
        batch_size = args.d5_batch_size

    train_dataset = TrainDataset(os.path.join(args.dataset_dir, args.train_img_dir), args.img_size)
    val_dataset = TrainDataset(os.path.join(args.dataset_dir, args.val_img_dir), args.img_size)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=batch_size, pin_memory=torch.cuda.is_available())
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size//2, shuffle=False,
                                num_workers=batch_size//2, pin_memory=torch.cuda.is_available())

    if args.cuda and torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    optimizer = Adam(decoder.parameters(), lr=args.lr)
    loss_fn = MSELoss()

    run_id = args.run_id
    model_path = Path('model_{relu_target}_{run_id}.pt'.format(relu_target=relu_targets[relu_target_id-1],
                      run_id=run_id))
    log_file = open('train_{relu_target}_{run_id}.log'.format(relu_target=relu_targets[relu_target_id-1],
                    run_id=run_id), 'at', encoding='utf8')

    step = 0
    valid_losses = []

    for epoch in range(epochs):
        decoder.train()
        random.seed()
        tq = tqdm(total=len(train_dataloader)*batch_size)
        tq.set_description('Run Id {}, Relu Target {} Epoch {} of {}, lr {}'.format(
                           run_id, relu_targets[relu_target_id-1], epoch, epochs, args.lr))
        losses = []
        try:
            mean_loss = 0.
            for i, input_imgs in enumerate(train_dataloader):
                if args.cuda and torch.cuda.is_available():
                    input_imgs = input_imgs.cuda()
                encoded = encoder(input_imgs)
                decoded = decoder(encoded)
                encoded_decoded = encoder(decoded)
                pixel_loss = args.pixel_weight * loss_fn(decoded, input_imgs)
                feature_loss = args.feature_weight * loss_fn(encoded_decoded, encoded)
                loss = pixel_loss + feature_loss

                loss.backward()
                optimizer.step()

                step += 1

                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-args.log_fr:])
                tq.set_postfix(loss="{:.6f}".format(mean_loss))

                if i and (i % args.log_fr) == 0:
                    write_event(log_file, step, loss=mean_loss)
            write_event(log_file, step, loss=mean_loss)
            tq.close()
            save_model(decoder, relu_target_id, epoch, step, model_path)

            valid_loss = validation(args, encoder, decoder, loss_fn, val_dataloader, batch_size)
            valid_loss_metric = {'valid_loss': valid_loss}
            write_event(log_file, step, **valid_loss_metric)
            valid_losses.append(valid_loss)

        except KeyboardInterrupt:
            tq.close()
            print('Ctrl+C, saving snapshot')
            save_model(decoder, relu_target_id, epoch, step, model_path)
            print('Terminated.')
    print('Done.')


def validation(args, encoder, decoder, loss_fn, val_dataloader, batch_size):
    print("Validating Network...")
    decoder.eval()
    losses = []
    val_tq = tqdm(total=len(val_dataloader)*(batch_size//2))
    with torch.no_grad():
        for i, input_imgs in enumerate(val_dataloader):
            val_tq.set_description('Validating, batch {}'.format(i))
            if args.cuda and torch.cuda.is_available():
                input_imgs = input_imgs.cuda()

            encoded = encoder(input_imgs)
            decoded = decoder(encoded)
            encoded_decoded = encoder(decoded)

            pixel_loss = args.pixel_weight * loss_fn(decoded, input_imgs)
            feature_loss = args.feature_weight * loss_fn(encoded_decoded, encoded)
            loss = pixel_loss + feature_loss
            val_tq.update((batch_size)//2)
            val_tq.set_postfix(loss="{:.6f}".format(loss))
            losses.append(loss)
        val_tq.close()
        valid_loss = float(np.mean(losses))
        print('Valid loss: {:.5f}'.format(valid_loss))

    return valid_loss


def test(args):
    # Data loading code
    dataset = TestDataset(args.content_path, args.style_path, args.fine_size)
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    avg_time = 0

    csF = torch.Tensor()
    if args.cuda and torch.cuda.is_available():
        csF = csF.cuda(args.gpu)

    for i, (content_img, style_img, imname) in enumerate(loader):
        imname = imname[0]
        print('Transferring ' + imname)
        with torch.no_grad():
            cImg = torch.tensor(content_img)
            sImg = torch.tensor(style_img)
        if args.cuda and torch.cuda.is_available():
            cImg = cImg.cuda(args.gpu)
            sImg = sImg.cuda(args.gpu)
        start_time = time.time()
        # WCT Style Transfer
        style_transfer(args, cImg, sImg, imname, csF)
        end_time = time.time()
        print('Elapsed time is: %f' % (end_time - start_time))
        avg_time += (end_time - start_time)

    print('Processed %d images. Averaged time is %f' % ((i+1), avg_time/(i+1)))


def main():
    args = Options().parse()
    if args.subcommand == 'train':
        train(args)
    elif args.subcommand == 'test':
        test(args)
    else:
        raise ValueError("Experiment type error")


if __name__ == '__main__':
    main()
