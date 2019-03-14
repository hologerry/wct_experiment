from __future__ import division

import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.serialization import load_lua

from encoder_decoder_test import (decoder1, decoder2, decoder3, decoder4, decoder5,
                                  encoder1, encoder2, encoder3, encoder4, encoder5)


class WCT(nn.Module):
    def __init__(self, args):
        super(WCT, self).__init__()
        # load pre-trained network
        vgg1 = load_lua(args.vgg1)
        decoder1_torch = load_lua(args.decoder1)
        vgg2 = load_lua(args.vgg2)
        decoder2_torch = load_lua(args.decoder2)
        vgg3 = load_lua(args.vgg3)
        decoder3_torch = load_lua(args.decoder3)
        vgg4 = load_lua(args.vgg4)
        decoder4_torch = load_lua(args.decoder4)
        vgg5 = load_lua(args.vgg5)
        decoder5_torch = load_lua(args.decoder5)

        self.e1 = encoder1(vgg1)
        self.d1 = decoder1(decoder1_torch)
        self.e2 = encoder2(vgg2)
        self.d2 = decoder2(decoder2_torch)
        self.e3 = encoder3(vgg3)
        self.d3 = decoder3(decoder3_torch)
        self.e4 = encoder4(vgg4)
        self.d4 = decoder4(decoder4_torch)
        self.e5 = encoder5(vgg5)
        self.d5 = decoder5(decoder5_torch)

    def whiten_and_color(self, cF, sF):
        cFSize = cF.size()
        c_mean = torch.mean(cF, 1)  # c x (h x w)
        c_mean = c_mean.unsqueeze(1).expand_as(cF)
        cF = cF - c_mean

        contentConv = torch.mm(cF, cF.t()).div(cFSize[1]-1) + torch.eye(cFSize[0]).double()
        c_u, c_e, c_v = torch.svd(contentConv, some=False)

        k_c = cFSize[0]
        for i in range(cFSize[0]):
            if c_e[i] < 0.00001:
                k_c = i
                break

        sFSize = sF.size()
        s_mean = torch.mean(sF, 1)
        sF = sF - s_mean.unsqueeze(1).expand_as(sF)
        styleConv = torch.mm(sF, sF.t()).div(sFSize[1]-1)
        s_u, s_e, s_v = torch.svd(styleConv, some=False)

        k_s = sFSize[0]
        for i in range(sFSize[0]):
            if s_e[i] < 0.00001:
                k_s = i
                break

        c_d = (c_e[0:k_c]).pow(-0.5)
        step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
        step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
        whiten_cF = torch.mm(step2, cF)

        s_d = (s_e[0:k_s]).pow(0.5)
        targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
        targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
        return targetFeature

    def transform(self, cF, sF, csF, alpha):
        cF = cF.double()
        sF = sF.double()
        C = cF.size(0)
        # C, W, H = cF.size(0), cF.size(1), cF.size(2)
        # _, W1, H1 = sF.size(0), sF.size(1), sF.size(2)
        cFView = cF.view(C, -1)
        sFView = sF.view(C, -1)

        targetFeature = self.whiten_and_color(cFView, sFView)
        targetFeature = targetFeature.view_as(cF)
        ccsF = alpha * targetFeature + (1.0 - alpha) * cF
        ccsF = ccsF.float().unsqueeze(0)
        csF.data.resize_(ccsF.size()).copy_(ccsF)
        return csF


def style_transfer(args, content_img, style_img, imname, csF):
    wct = WCT(args)
    if args.cuda and torch.cuda.is_available():
        wct = wct.cuda(args.gpu)
    sF5 = wct.e5(style_img)
    cF5 = wct.e5(content_img)
    # print("Feturemap: E5 size:", sF5.size())
    sF5 = sF5.data.cpu().squeeze(0)
    cF5 = cF5.data.cpu().squeeze(0)
    csF5 = wct.transform(cF5, sF5, csF, args.alpha)
    Im5 = wct.d5(csF5)

    sF4 = wct.e4(style_img)
    cF4 = wct.e4(Im5)
    # print("Feturemap: E4 size:", sF4.size())
    sF4 = sF4.data.cpu().squeeze(0)
    cF4 = cF4.data.cpu().squeeze(0)
    csF4 = wct.transform(cF4, sF4, csF, args.alpha)
    Im4 = wct.d4(csF4)

    sF3 = wct.e3(style_img)
    cF3 = wct.e3(Im4)
    # print("Feturemap: E3 size:", sF3.size())
    sF3 = sF3.data.cpu().squeeze(0)
    cF3 = cF3.data.cpu().squeeze(0)
    csF3 = wct.transform(cF3, sF3, csF, args.alpha)
    Im3 = wct.d3(csF3)

    sF2 = wct.e2(style_img)
    cF2 = wct.e2(Im3)
    # print("Feturemap: E2 size:", sF2.size())
    sF2 = sF2.data.cpu().squeeze(0)
    cF2 = cF2.data.cpu().squeeze(0)
    csF2 = wct.transform(cF2, sF2, csF, args.alpha)
    Im2 = wct.d2(csF2)

    sF1 = wct.e1(style_img)
    cF1 = wct.e1(Im2)
    # print("Feturemap: E1 size:", sF1.size())
    sF1 = sF1.data.cpu().squeeze(0)
    cF1 = cF1.data.cpu().squeeze(0)
    csF1 = wct.transform(cF1, sF1, csF, args.alpha)
    Im1 = wct.d1(csF1)
    # save_image has this wired design to pad images with 4 pixels at default.
    vutils.save_image(Im1.data.cpu().float(), os.path.join(args.output_dir, imname))
    return
