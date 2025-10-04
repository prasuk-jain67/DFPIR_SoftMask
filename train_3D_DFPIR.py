import os, time, shutil
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.image as mpimg

import numpy as np
import time
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
from utils.dataset_utils import PromptTrainDataset,DenoiseTestDataset, DerainDehazeDataset
from net.model import ChannelShuffle_skip_textguaid
import subprocess
from torch.utils.data import DataLoader
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
import clip


parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--save_epoch', type=int, default=1,
                    help='save model per every N epochs')
parser.add_argument('--save_item', type=int, default=2000, #---save item--------------
                    help='save model per every N item')
parser.add_argument('--init_epoch', type=int, default=1, # -------------------
                    help='if finetune model, set the initial epoch')
parser.add_argument('--save_dir', type=str, default='./', # ----------------
                     help='save parameter dir')

parser.add_argument('--gpu', type=str, default="0,1", # -----------GPU
                    help='GPUs') 
parser.add_argument('--cuda', type=int, default=1) # -----------GPU
parser.add_argument('--pretrained_1', type=str, default=
        './', 
        help='training loss')
parser.add_argument('--epochs', type=int, default=500, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=5,help="Batch size to use per GPU") #------
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate of encoder.') # -----

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')
# path
parser.add_argument('--data_file_dir', type=str, default=r'data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default=r'train_data/noise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default=r'train_data/rain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default=r'train_data/haze/',
                    help='where training images of dehazing saves.')

parser.add_argument("--wblogger",type=str,default="promptir",help = "Determine to log to wandb or not and the project name")

# -------------------------Test datasets path----
parser.add_argument('--denoise_path', type=str, default="test/denoise/", help='save path of test noisy images')
parser.add_argument('--derain_path', type=str, default="test/derain/", help='save path of test raining images')
parser.add_argument('--dehaze_path', type=str, default="test/dehaze/", help='save path of test hazy images')
parser.add_argument('--output_path', type=str, default="output_chshuffle/3D/", help='output save path')


args = parser.parse_args()

psnr_max = 10
clip_model, _ = clip.load("ViT-B/32", device=args.cuda)
for param in clip_model.parameters():
    param.requires_grad = False  

    
inputext = ["Gaussian noise with a standard deviation of 15","Gaussian noise with a standard deviation of 25"
            ,"Gaussian noise with a standard deviation of 50","Rain degradation with rain lines"
            ,"Hazy degradation with normal haze"] # detailed description.

# inputext = ["Noise","Noise"
#             ,"Noise","Rain"
#             ,"Haze"] # Simple description

# denoise_splits = ["urban100/","bsd68/"] 
denoise_splits = ["bsd68/"]
derain_splits = ["Rain100L/"]
denoise_tests = []
derain_tests = []
base_path = args.denoise_path

derain_base_path = args.derain_path

args.derain_path = args.derain_path+"Rain100L/" 

for i in denoise_splits:
    args.denoise_path = os.path.join(base_path,i)
    denoise_testset = DenoiseTestDataset(args)
    denoise_tests.append(denoise_testset)
# ------------------------------------------------------------------------  

def train(train_loader, model, optimizer, epoch, epoch_total,criterionL1):
    loss_sum = 0
    losses = AverageMeter()
    
# -----------------------------------------------------------
    writer = SummaryWriter("./logs_train")
    psnr_tqdm = 10
    ssim_tqdm = 0.009
    loss_tqdm = 0.0

    model.train()
    start_time = time.time()
    global psnr_max

    loop_train = tqdm((train_loader), total = len(train_loader),leave=False,colour="magenta") 
    for i, ([clean_name, de_id], degrad_patch, clean_patch) in enumerate(loop_train):

        input_var = Variable(degrad_patch.cuda())
        target_var = Variable(clean_patch.cuda())

        result = [clean_name, de_id]
        img_id = result[1]
        img_id = img_id.tolist()  
        text_prompt_list = [inputext[idx] for idx in img_id]
# ------------------------------------------
        text_token = clip.tokenize(text_prompt_list).to(args.cuda) 
        text_code = clip_model.encode_text(text_token).to(dtype=torch.float32)  
# --------------------------------------------------
        output = model(input_var,text_code) # 
        loss = criterionL1(output,target_var)
        loss_sum+=loss.item()
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i % 10 == 0) and (i != 0):
            loss_avg = loss_sum / 10
            loss_sum = 0.0 
            loss_tqdm = loss_avg                                                   
            writer.add_scalar("train_loss", loss.item(), i)
            start_time = time.time()     
        if (i % args.save_item == 0) and (i != 0): 
            psnr_n15,ssim_n15,psnr_n25,ssim_n25,psnr_n50,ssim_n50,psnr_rain,ssim_rain,psnr_haze,ssim_haze =test( model, criterionL1)
            psnr_avr = (psnr_n15+psnr_n25+psnr_n50+psnr_rain+psnr_haze)/5
            ssim_avr = (ssim_n15+ssim_n25+ssim_n50+ssim_rain+ssim_haze)/5
            psnr_tqdm = psnr_avr
            ssim_tqdm  = ssim_avr

            if psnr_avr > psnr_max - 0.0001:
                psnr_max = max(psnr_avr, psnr_max)
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                    os.path.join(args.save_dir,
                                    'checkpoint_epoch_V1_{:0>4}_{}_p_n{:.2f}-{:.4f}_{:.2f}-{:.4f}_{:.2f}-{:.4f}_p_r{:.2f}-{:.4f}_p_h{:.2f}-{:.4f}avr{:.2f}-{:.4f}.pth.tar'
                                    .format(epoch, i//args.save_item, psnr_n15,ssim_n15,psnr_n25,ssim_n25,psnr_n50,ssim_n50,
                                        psnr_rain,ssim_rain,psnr_haze,ssim_haze,psnr_avr,ssim_avr)))
            else:
                torch.save({}, os.path.join(args.save_dir,
                                    'checkpoint_epoch_V1_{:0>4}_{}_p_n{:.2f}-{:.4f}_{:.2f}-{:.4f}_{:.2f}-{:.4f}_p_r{:.2f}-{:.4f}_p_h{:.2f}-{:.4f}avr{:.2f}-{:.4f}.pth.tar'
                                    .format(epoch, i//args.save_item, psnr_n15,ssim_n15,psnr_n25,ssim_n25,psnr_n50,ssim_n50,
                                        psnr_rain,ssim_rain,psnr_haze,ssim_haze,psnr_avr,ssim_avr)))
                
        loop_train.set_description(f'trainning->epoch:[{epoch}/{args.epochs}],item:[{i}/{len(train_loader)}]') 
        loop_train.set_postfix(loss = loss_tqdm,psnr = f'{psnr_tqdm:.4f}', ssim = f'{ssim_tqdm:.4f}')        
    writer.close()
    return losses.avg


def test(model, criterion):
    model.eval()
# -------------------------
    for testset,name in zip(denoise_tests,denoise_splits) :
        print('Start {} testing Sigma=15...'.format(name))
        psnr_g15,ssim_g15 = test_Denoise(model, testset, sigma=15,text_prompt=inputext[0])
        print('{}test ok psnr_g15:{:.4f} ssim_g15:{:.4f},'.format(name,psnr_g15,ssim_g15))

        print('Start {} testing Sigma=25...'.format(name))
        psnr_g25,ssim_g25 = test_Denoise(model, testset, sigma=25,text_prompt=inputext[1])
        print('{}test ok psnr_g25:{:.4f} ssim_g25:{:.4f},'.format(name,psnr_g25,ssim_g25))

        print('Start {} testing Sigma=50...'.format(name))
        psnr_g50,ssim_g50 = test_Denoise(model, testset, sigma=50,text_prompt=inputext[2])
        print('{}test ok psnr_g50:{:.4f} ssim_g50:{:.4f},'.format(name,psnr_g50,ssim_g50))
# ----------------------- 
    print('Start testing Rain100L rain streak removal...') # 
    derain_set = DerainDehazeDataset(args,addnoise=False,sigma=15)
    psnr_rain,ssim_rain = test_Derain_Dehaze(model, derain_set, task="derain",text_prompt=inputext[3])
    print('Rain100L test ok psnr_rain:{:.4f} ssim_rain:{:.4f},'.format(psnr_rain,ssim_rain))
# ----------------------
    print('Start testing SOTS...')
    psnr_haze,ssim_haze = test_Derain_Dehaze(model, derain_set, task="dehaze",text_prompt=inputext[4]) 
    print('dehaze test ok psnr_haze:{:.4f} ssim_haze:{:.4f},'.format(psnr_haze,ssim_haze))
    return psnr_g15,ssim_g15,psnr_g25,ssim_g25,psnr_g50,ssim_g50,psnr_rain,ssim_rain,psnr_haze,ssim_haze


def test_Denoise(net, dataset, sigma=15, text_prompt=""):
    output_path = args.output_path + 'denoise/' + str(sigma) + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0) 
    psnr = AverageMeter()
    ssim = AverageMeter() 
    text_token = clip.tokenize(text_prompt).to(args.cuda) 
    text_code = clip_model.encode_text(text_token).to(dtype=torch.float32)  
    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader,colour="magenta"):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
        
            restored = net(degrad_patch,text_code)          
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)    
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            psnr_value_formatted = "{:.2f}".format(temp_psnr)  
            filename = f"_{psnr_value_formatted}"
            save_image_tensor(restored, output_path + clean_name[0] + filename + '.png')
            # save_image_tensor(restored, output_path + clean_name[0] + '.png')
        print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))
    return psnr.avg,ssim.avg

def test_Derain_Dehaze(net, dataset, task="derain",text_prompt=""):
    output_path = args.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)
    psnr = AverageMeter()
    ssim = AverageMeter()
    text_token = clip.tokenize(text_prompt).to(args.cuda) 
    text_code = clip_model.encode_text(text_token).to(dtype=torch.float32) 
    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader,colour="magenta"):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch,text_code)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            psnr_value_formatted = "{:.2f}".format(temp_psnr)  
            filename = f"_{psnr_value_formatted}"
            save_image_tensor(restored, output_path + degraded_name[0] + filename + '.png')
            # save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))
    return psnr.avg,ssim.avg


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# -------------------------------------------------------
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(args.cuda) 

    model = ChannelShuffle_skip_textguaid(device=args.cuda)
    criterionL1 = nn.L1Loss()
    model.cuda()

    if os.path.exists(os.path.join(args.save_dir, 'checkpoint_{:0>4}.pth.tar'.format(args.init_epoch))):
        # load existing model
        model_info = torch.load(os.path.join(args.save_dir, 'checkpoint_{:0>4}.pth.tar'.format(args.init_epoch)))
        print('==> loading existing model:',
              os.path.join(args.save_dir, 'checkpoint_{:0>4}.pth.tar'.format(args.init_epoch)))
        model.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        scheduler.load_state_dict(model_info['scheduler'])
        cur_epoch = model_info['epoch']
    else:
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        # create model
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        cur_epoch = args.init_epoch

    if args.pretrained_1:
        if os.path.isfile(args.pretrained_1):
            print("=> loading model '{}'".format(args.pretrained_1))
            model_pretrained = torch.load(args.pretrained_1,map_location=torch.device('cuda:0'))        
            pretrained_dict = model_pretrained['state_dict']
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            print("=> no model found at '{}'".format(args.pretrained_1)) 
# -----------------------------------------------------------
    train_dataset = PromptTrainDataset(args)              
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    print('load dataset ok')
    psnr_n15,ssim_n15,psnr_n25,ssim_n25,psnr_n50,ssim_n50,psnr_rain,ssim_rain,psnr_haze,ssim_haze =test( model, criterionL1) 
    psnr_avr = (psnr_n15+psnr_n25+psnr_n50+psnr_rain+psnr_haze)/5
    ssim_avr = (ssim_n15+ssim_n25+ssim_n50+ssim_rain+ssim_haze)/5
    print('test ok! psnr_noise:{:.4f}-{:.4f}_{:.4f}-{:.4f}_{:.4f}-{:.4f},--psnr_rain:{:.4f} ssim_rain:{:.4f},--psnr_haze:{:.4f} ssim_haze:{:.4f}avr{:.4f}-{:.4f},'
          .format(psnr_n15,ssim_n15,psnr_n25,ssim_n25,psnr_n50,ssim_n50,psnr_rain,ssim_rain,psnr_haze,ssim_haze,psnr_avr,ssim_avr))           

    for epoch in range(cur_epoch, args.epochs + 1):
        loss = train(train_loader, model, optimizer, epoch, args.epochs + 1,criterionL1)
        scheduler.step()

        if epoch % args.save_epoch == 0:
            psnr_n15,ssim_n15,psnr_n25,ssim_n25,psnr_n50,ssim_n50,psnr_rain,ssim_rain,psnr_haze,ssim_haze =test( model, criterionL1)
            psnr_avr = (psnr_n15+psnr_n25+psnr_n50+psnr_rain+psnr_haze)/5
            ssim_avr = (ssim_n15+ssim_n25+ssim_n50+ssim_rain+ssim_haze)/5
            if psnr_avr > psnr_max - 0.01:
                psnr_max = max(psnr_avr, psnr_max)
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                    os.path.join(args.save_dir,
                                'checkpoint_epoch_V1_{:0>4}_p_n{:.2f}-{:.4f}_{:.2f}-{:.4f}_{:.2f}-{:.4f}_p_r{:.2f}-{:.4f}_p_h{:.2f}-{:.4f}avr{:.2f}-{:.4f}.pth.tar'
                                .format(epoch, psnr_n15,ssim_n15,psnr_n25,ssim_n25,psnr_n50,ssim_n50,
                                    psnr_rain,ssim_rain,psnr_haze,ssim_haze,psnr_avr,ssim_avr)))
            else:
                torch.save({}, os.path.join(args.save_dir,
                                'checkpoint_epoch_V1_{:0>4}_p_n{:.2f}-{:.4f}_{:.2f}-{:.4f}_{:.2f}-{:.4f}_p_r{:.2f}-{:.4f}_p_h{:.2f}-{:.4f}avr{:.2f}-{:.4f}.pth.tar'
                                .format(epoch, psnr_n15,ssim_n15,psnr_n25,ssim_n25,psnr_n50,ssim_n50,
                                    psnr_rain,ssim_rain,psnr_haze,ssim_haze,psnr_avr,ssim_avr)))
        print('Epoch [{0}]\t'
              'lr: {lr:.6f}\t'
              'Loss: {loss:.5f}'
            .format(
            epoch,
            lr=optimizer.param_groups[-1]['lr'],
            loss=loss))

            