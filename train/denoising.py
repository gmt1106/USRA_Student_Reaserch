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
    # std of the noise
    sigma = 25
    sigma_ = sigma/255.

    # set up original and noisy images for deep image prior 
    path_to_image = homeDirectory + 'data/denoising/F16_GT.png'
    img_orig_pil = Image.open(path_to_image)
    img_orig_np = np.array(img_orig_pil)
    img_orig_np = img_orig_np.transpose(2,0,1)
    img_orig_np = img_orig_np.astype(np.float32) / 255.
    # original 
    # we usually need the dimensions to be divisible by a power of two (32 in this case)
    new_size = (img_orig_pil.size[0] - img_orig_pil.size[0] % 32, img_orig_pil.size[1] - img_orig_pil.size[1] % 32)
    bbox = [(img_orig_pil.size[0] - new_size[0])/2, 
            (img_orig_pil.size[1] - new_size[1])/2, 
            (img_orig_pil.size[0] + new_size[0])/2,
            (img_orig_pil.size[1] + new_size[1])/2,]
    img_orig_pil = img_orig_pil.crop(bbox)
    img_orig_np = np.array(img_orig_pil)
    img_orig_np = img_orig_np.transpose(2,0,1)
    img_orig_np = img_orig_np.astype(np.float32) / 255.
    # noisy
    img_noisy_np = np.clip(img_orig_np + np.random.normal(scale=sigma, size=img_orig_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np.clip(img_noisy_np*255,0,255).astype(np.uint8)
    img_noisy_pil = img_noisy_pil.transpose(1, 2, 0)
    img_noisy_pil = Image.fromarray(img_noisy_pil)

    net_input_width, net_input_height = img_orig_pil.size


    ############## Setup ##############
    # Setup input meshgrid 
    tensors = [torch.linspace(-1, 1, steps = net_input_height), torch.linspace(-1, 1, steps = net_input_width)]
    net_input = torch.stack(torch.meshgrid(*tensors), dim=-1).type(dtype)
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
    num_iter = 5000
    exp_weight = 0.99
    out_avg = None
    img_noisy = torch.from_numpy(img_noisy_np)[None, :].type(dtype)
    img_orig = torch.from_numpy(img_orig_np)[None, :].type(dtype)

    # Create optimizier
    parameters = sirenNet.parameters()
    optimizer = torch.optim.Adam(parameters, lr=leanrning_rate)

    # Loss
    loss = nn.MSELoss().type(dtype) 

    # tensorboard log directory 
    log_dir = homeDirectory + 'logs/experiment/Siren/denoising'

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
        out_orig = sirenNet.forward(SIREN_net_input)
        out_orig = rearrange(out_orig, '(h w) c -> () c h w', h = net_input_height, w = net_input_width)

        # Smoothing 
        if out_avg is None:
            out_avg = out_orig.detach()
        else:
            out_avg = out_avg * exp_weight + out_orig.detach() * (1 - exp_weight)

        # Compute the loss 
        total_loss = loss(out_orig, img_noisy)

        # Compute gradients    
        total_loss.backward()

        # Update parameters
        optimizer.step()
        
        # Zero the parameter gradients in the optimizer
        optimizer.zero_grad()

        # Save the results
        if i % 25 == 0:
             # Write output image to tensorboard, using keywords `image_output`
            writer.add_image("image_output", out_orig, global_step=i, dataformats='NCHW')
            # Write loss to tensorboard, using keywords `loss`
            writer.add_scalar("loss", total_loss, global_step=i)
            # Write PSNR of noisy image 
            psrn_noisy = peak_signal_noise_ratio(img_noisy_np, out_orig.detach().cpu().numpy()[0]) 
            writer.add_scalar("noisy_img_PSNR", psrn_noisy, global_step=i)
            # Write PSNR of orig image
            psrn_orig = peak_signal_noise_ratio(img_orig_np, out_orig.detach().cpu().numpy()[0])
            writer.add_scalar("orig_img_PSNR", psrn_orig, global_step=i)
            # Write PSNR of orig image with overage of output
            psrn_orig_sm = peak_signal_noise_ratio(img_orig_np, out_avg.detach().cpu().numpy()[0])
            writer.add_scalar("orig_img_sm_PSNR", psrn_orig_sm, global_step=i)
            # Write LPIPS evaluation
            lpips_evaluation = loss_fn_alex(img_orig, out_orig)
            writer.add_scalar("LPIPS", lpips_evaluation, global_step=i)

