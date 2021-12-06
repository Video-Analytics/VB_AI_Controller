from __future__ import print_function
import argparse

import os
from functools import reduce
import time
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import gc
import dbpn.utils


import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dbpn.dbpn_v1 import Net as DBPNLL
from dbpn.dbpn import Net as DBPN
from dbpn.data import get_eval_set


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_dir', type=str, default='input')
parser.add_argument('--test_dataset', type=str, default='vb')
parser.add_argument('--output', default='output/', help='Location to save checkpoint models')

parser.add_argument('--model_type', type=str, default='DBPNLL')
parser.add_argument('--model', default='models/PIRM2018_region3.pth', help='sr pretrained base model')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")

parser.add_argument('--chop_forward', type=bool, default=True)
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--self_ensemble', type=bool, default=False)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')

opt = parser.parse_args()


def eval(opt):

    gpus_list = range(opt.gpus)

    cuda = opt.gpu_mode

    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    print('===> Loading datasets')
    test_set = get_eval_set(os.path.join(opt.input_dir, opt.test_dataset), opt.upscale_factor)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize,
                                     shuffle=False)

    print('===> Building model')
    if opt.model_type == 'DBPNLL':
        model = DBPNLL(num_channels=3, base_filter=64, feat=256, num_stages=10, scale_factor=opt.upscale_factor)
    # elif opt.model_type == 'DBPN-RES-MR64-3':
    #    model = DBPNITER(num_channels=3, base_filter=64,  feat = 256, num_stages=3, scale_factor=opt.upscale_factor)
    else:
        model = DBPN(num_channels=3, base_filter=64, feat=256, num_stages=7, scale_factor=opt.upscale_factor)

    if cuda:
        model = torch.nn.DataParallel(model, device_ids=gpus_list)

    model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
    print('Super resolution model is loaded.')

    if cuda:
        model = model.cuda(gpus_list[0])

    model.eval()
    for batch in testing_data_loader:
        torch.cuda.empty_cache()

        input, name = batch[0], batch[2]
        input[0] = dbpn.utils.norm(input[0],vgg=True)

        with torch.no_grad():
            input = Variable(input)

        if cuda:
            input = input.cuda(gpus_list[0])

        t0 = time.time()
        if opt.chop_forward:
            with torch.no_grad():
                prediction = chop_forward(input, model, opt.upscale_factor)
        else:
            if opt.self_ensemble:
                with torch.no_grad():
                    prediction = x8_forward(input, model)
            else:
                with torch.no_grad():
                    prediction = model(input)
        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
        prediction = dbpn.utils.denorm(prediction.data[0].cpu(),vgg=True)
        save_img(prediction, name[0])

        del batch, input, name, prediction
        gc.collect()

def save_img(img, img_name):
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)

    # save img
    save_dir=os.path.join(opt.output,opt.test_dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_fn = save_dir +'/'+ img_name
    cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

def x8_forward(img, model, precision='single'):
    def _transform(v, op):
        if precision != 'single': v = v.float()

        v2np = v.data.cpu().numpy()
        if op == 'vflip':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'hflip':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 'transpose':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()
        
        ret = torch.Tensor(tfnp).cuda()

        if precision == 'half':
            ret = ret.half()
        elif precision == 'double':
            ret = ret.double()

        with torch.no_grad():
            ret = Variable(ret)

        del tfnp
        return ret

    inputlist = [img]
    for tf in 'vflip', 'hflip', 'transpose':
        inputlist.extend([_transform(t, tf) for t in inputlist])

    outputlist = [model(aug) for aug in inputlist]
    for i in range(len(outputlist)):
        if i > 3:
            outputlist[i] = _transform(outputlist[i], 'transpose')
        if i % 4 > 1:
            outputlist[i] = _transform(outputlist[i], 'hflip')
        if (i % 4) % 2 == 1:
            outputlist[i] = _transform(outputlist[i], 'vflip')
    
    output = reduce((lambda x, y: x + y), outputlist) / len(outputlist)

    del inputlist, outputlist
    return output
    
def chop_forward(x, model, scale, shave=16, min_size=10000, nGPUs=opt.gpus):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            with torch.no_grad():
                input_batch = torch.cat(inputlist[i:(i + nGPUs)], dim=0)
            if opt.self_ensemble:
                with torch.no_grad():
                    output_batch = x8_forward(input_batch, model)
            else:
                with torch.no_grad():
                    output_batch = model(input_batch)
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch, model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output = Variable(x.data.new(b, c, h, w))
    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    del b, c, h, w, shave, h_half, h_size, outputlist
    return output


# Method to process the red band of the image
def normalize_red(intensity):
    iI = intensity
    minI = 86
    maxI = 230
    minO = 0
    maxO = 255
    iO = (iI - minI) * (((maxO - minO) / (maxI - minI)) + minO)
    return iO

# Method to process the green band of the image
def normalize_green(intensity):
    iI = intensity
    minI = 90
    maxI = 225
    minO = 0
    maxO = 255
    iO = (iI - minI) * (((maxO - minO) / (maxI - minI)) + minO)
    return iO

# Method to process the blue band of the image
def normalize_blue(intensity):
    iI = intensity
    minI = 100
    maxI = 210
    minO = 0
    maxO = 255
    iO = (iI - minI) * (((maxO - minO) / (maxI - minI)) + minO)
    return iO

def contrast_stretch(img):
    # Split the red, green and blue bands from the Image
    multiBands = img.split()

    # Apply point operations that does contrast stretching on each color band
    normalizedRedBand = multiBands[0].point(normalize_red)
    normalizedGreenBand = multiBands[1].point(normalize_green)
    normalizedBlueBand = multiBands[2].point(normalize_blue)

    # Create a new image from the contrast stretched red, green and blue brands
    normalized_image = Image.merge("RGB", (normalizedRedBand, normalizedGreenBand, normalizedBlueBand))
    return normalized_image


def edge_enhance(img):
    edgehanced = img.filter(ImageFilter.EDGE_ENHANCE)
    return edgehanced


def pil_enhance(img):
    #Brightness
    im1 = ImageEnhance.Brightness(img)
    im1 = im1.enhance(1.10)
    #Contrast
    im2 = ImageEnhance.Contrast(im1)
    im2 = im2.enhance(1.25)
    #Sharpness
    im3 = ImageEnhance.Sharpness(im2)
    im3 = im3.enhance(1.20)
    #Color
    im4 = ImageEnhance.Color(im3)
    img_output = im4.enhance(1.20)
    return img_output

'''
input_dir = r'./input/vb/'
output_dir = r'./cache/vb/'

for image in os.listdir(input_dir):
    img = Image.open(input_dir+image)
    img = edge_enhance(img)
    img = pil_enhance(img)
    #img = contrast_stretch(img)
    img.save(output_dir+image, quality=90)
'''

eval(opt)




time.sleep(1)
print('Done.')
time.sleep(2)