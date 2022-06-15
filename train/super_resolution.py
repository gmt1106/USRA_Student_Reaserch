import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from einops import rearrange
import os
from skimage.metrics import peak_signal_noise_ratio
# use tensorboard with pytorch
from torch.utils.tensorboard import SummaryWriter
from models import *
from models.siren_pytorch import SirenNet
import lpips


# Train for SIREN
def sirenTrain(w0, leanrning_rate):

    homeDirectory = '../'
    dtype = None
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor


    ############## Load image ##############
    # Downsampler factor
    factor = 4 

    # set up original, HR, LR images for deep image prior 
    path_to_image = homeDirectory + 'data/sr/zebra_GT.png'
    img_orig_pil = Image.open(path_to_image)
    img_orig_np = np.array(img_orig_pil)
    img_orig_np = img_orig_np.transpose(2,0,1)
    img_orig_np = img_orig_np.astype(np.float32) / 255.
    # HR 
    # we usually need the dimensions to be divisible by a power of two (32 in this case)
    new_size = (img_orig_pil.size[0] - img_orig_pil.size[0] % 32, img_orig_pil.size[1] - img_orig_pil.size[1] % 32)
    bbox = [(img_orig_pil.size[0] - new_size[0])/2, 
            (img_orig_pil.size[1] - new_size[1])/2, 
            (img_orig_pil.size[0] + new_size[0])/2,
            (img_orig_pil.size[1] + new_size[1])/2,]
    img_HR_pil = img_orig_pil.crop(bbox)
    img_HR_np = np.array(img_HR_pil)
    img_HR_np = img_HR_np.transpose(2,0,1)
    img_HR_np = img_HR_np.astype(np.float32) / 255.
    # LR
    LR_size = [img_HR_pil.size[0] // factor, img_HR_pil.size[1] // factor]
    img_LR_pil = img_HR_pil.resize(LR_size, Image.ANTIALIAS)
    img_LR_np = np.array(img_LR_pil)
    img_LR_np = img_LR_np.transpose(2,0,1)
    img_LR_np = img_LR_np.astype(np.float32) / 255.

    net_input_width, net_input_height = img_HR_pil.size


    ############## Setup ##############
    # Setup input meshgrid 
    tensors = [torch.linspace(-1, 1, steps = net_input_height), torch.linspace(-1, 1, steps = net_input_width)]
    net_input = torch.stack(torch.meshgrid(*tensors), dim=-1).type(dtype)
    Deep_Image_Prior_net_input = rearrange(net_input, 'h w c -> () c h w', h = net_input_height, w = net_input_width).type(dtype).detach().requires_grad_()
    SIREN_net_input = rearrange(net_input, 'h w c -> (h w) c').type(dtype).detach().requires_grad_()
    input_depth = 2

    # Setup SIREN
    sirenNet = SirenNet(
        dim_in = input_depth,              # input dimension, ex. 2d coor
        dim_hidden = 256,                  # hidden dimension
        dim_out = 3,                       # output dimension, ex. rgb value
        num_layers = 5,                    # number of layers
        w0_initial = w0).type(dtype)      # different signals may require different omega_0 in the first layer - this is a hyperparameter


    ############## SIREN train ##############
    LR = 0.01
    num_iter = 900
    img_LR = torch.from_numpy(img_LR_np)[None, :].type(dtype)
    img_HR = torch.from_numpy(img_HR_np)[None, :].type(dtype)

    # Create optimizier
    parameters = sirenNet.parameters()
    optimizer = torch.optim.Adam(parameters, lr=LR)

    # Loss
    loss = nn.MSELoss().type(dtype) 

    # tensorboard log directory 
    log_dir = homeDirectory + './logs/experiment/Siren/super_resolution'

    # Create summary writer
    writer = SummaryWriter(log_dir)

    # Create log directory and save directory if it does not exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # LPIPS evaluation
    loss_fn_alex = lpips.LPIPS(net='alex').type(dtype)

    # Training loop
    for i in range(num_iter):

        # Apply the model to obtain scores (forward pass)
        out_HR = sirenNet.forward(SIREN_net_input)
        out_HR = rearrange(out_HR, '(h w) c -> () c h w', h = net_input_height, w = net_input_width)

        # Downsampling
        out_LR = nn.functional.interpolate(out_HR, scale_factor=1/factor, mode="bilinear", antialias=True)

        # Compute the loss 
        total_loss = loss(out_LR, img_LR)

        # Compute gradients    
        total_loss.backward()

        # Update parameters
        optimizer.step()
        
        # Zero the parameter gradients in the optimizer
        optimizer.zero_grad()

        # Save the results
        if i % 25 == 0:
            # Write output image to tensorboard, using keywords `image_output`
            #cliping 
            imageOutput = out_HR.detach().clone()
            imageOutput[imageOutput > 255] = 255
            imageOutput[imageOutput < 0] = 0
            writer.add_image("image_output", imageOutput, global_step=i, dataformats='NCHW')
            # Write loss to tensorboard, using keywords `loss`
            writer.add_scalar("loss", total_loss, global_step=i)
            # Write PSNR of LR image 
            psnr_LR = peak_signal_noise_ratio(img_LR_np, out_LR.detach().cpu().numpy()[0])
            writer.add_scalar("LR_PSNR", psnr_LR, global_step=i)
            # Write PSNR of HR image
            psnr_HR = peak_signal_noise_ratio(img_HR_np, out_HR.detach().cpu().numpy()[0])
            writer.add_scalar("HR_PSNR", psnr_HR, global_step=i)
            # Write LPIPS evaluation
            lpips_evaluation = loss_fn_alex(img_HR, out_HR)
            writer.add_scalar("LPIPS", lpips_evaluation, global_step=i)


