from turtle import forward
import torch
import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self, num_input_channels, num_output_channels, kernel_size=3, act_fun=nn.ReLU):

        super(ConvBlock, self).__init__()

        self.act_fun = act_fun

        self.model = nn.Sequential()

        self.model.add_module("BatchNorm2d_ConvBlock_1", nn.BatchNorm2d(num_input_channels))
        self.model.add_module("Conv2d_ConvBlock_1", nn.Conv2d(num_input_channels, num_output_channels, 1, 1, padding=0))
        self.act_fun()

        self.model.add_module("BatchNorm2d_ConvBlock_2", nn.BatchNorm2d(num_output_channels))
        to_pad = kernel_size // 2
        self.model.add_module("Conv2d_ConvBlock_2", nn.Conv2d(num_output_channels, num_output_channels, kernel_size, 1, to_pad))
        self.act_fun()

        self.model.add_module("BatchNorm2d_ConvBlock_3", nn.BatchNorm2d(num_output_channels))
        self.model.add_module("Conv2d_ConvBlock_3", nn.Conv2d(num_output_channels, num_output_channels, 1, 1, padding=0))
        self.act_fun()

        # this is for the identity to match the channel size 
        self.identity_downsample = nn.Sequential(
            nn.Conv2d(num_input_channels, num_output_channels, 1, 1, padding=0),
            nn.BatchNorm2d(num_output_channels),
        )

    def forward(self, input):

        identity = self.identity_downsample(input)
        x_out = self.model(input) + identity
        return x_out


class ResNet(nn.Module):

    def __init__(self, kernel_size=3, act_fun=nn.ReLU):

        """
        Arguments:
            input_width: original image width(W) // 4
            input_height: original image height(H) // 4
        """

        super(ResNet, self).__init__()

        self.model = nn.Sequential()

        in_channels_dim = 3
        out_channels_dim = 32

        for i in range(2):
            # conv block with the output channel, c = 32
            self.model.add_module(f"Conv_Block_{i}", ConvBlock(in_channels_dim, out_channels_dim, kernel_size = kernel_size, act_fun = act_fun))
            # upsample 2d with bilinear interpolation to double w and h
            self.model.add_module(f"Upsample_{i}", nn.Upsample(scale_factor=2, mode='bilinear'))
            in_channels_dim = out_channels_dim
        
        # conv block with the output channel,  c = 3
        self.model.add_module("Conv_Block_2", ConvBlock(in_channels_dim, 3))

    def forward(self, input):

        return self.model(input)


class Concat(nn.Module):

    def __init__(self, dim, *args):

        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):

        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        return torch.cat(inputs, dim=self.dim)

    def __len__(self):

        return len(self._modules)

class Net(nn.Module):

    def __init__(self, num_input_channels=2, num_output_channels=3, 
        channels_down=[16, 32, 64, 128, 128], channels_up=[16, 32, 64, 128, 128], channels_skip=[4, 4, 4, 4, 4], 
        kernel_size_down=[3, 3, 3, 3, 3], kernel_size_up=[3, 3, 3, 3, 3], kernel_size_skip=1,
        need_sigmoid=True, need_bias=True, 
        pad='reflection', upsample_mode='bilinear', downsample_mode='stride',
        need1x1_up=True):

        """Assembles encoder-decoder with skip connections.
        Arguments:
            act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
            pad (string): zero|reflection (default: 'zero')
            upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
            downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')
        """
        super(Net, self).__init__()

        assert len(channels_down) == len(channels_up) == len(channels_skip)

        n_scales = len(channels_down) 

        upsample_mode = [upsample_mode]*n_scales
        downsample_mode = [downsample_mode]*n_scales
        
        last_scale = n_scales - 1 

        self.model = nn.Sequential()
        model_internal_block = self.model

        in_channels_dim = num_input_channels

        for i in range(n_scales):

            ########### Decoder Part ###########
            deeper_block = nn.Sequential()
            skip_block = nn.Sequential()

            if channels_skip[i] != 0:
                model_internal_block.add_module(f"Concat_Block_{i}", Concat(1, skip_block, deeper_block))
            else:
                model_internal_block.add_module(f"Deeper_Block_{i}", deeper_block)

            ### skip block ###
            out_channels_dim = channels_skip[i]
            if channels_skip[i] != 0:
                to_pad = int((kernel_size_skip - 1) / 2)
                if pad == 'reflection':
                    # this will keep the width and the height of the input the same after conv2d
                    skip_block.add_module(f"ReflectionPad_Skip_{i}", nn.ReflectionPad2d(to_pad))
                    to_pad = 0
                # nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding)
                skip_block.add_module(f"Conv2d_Skip_{i}", nn.Conv2d(in_channels_dim, out_channels_dim, kernel_size_skip, 1, padding=to_pad, bias=need_bias))
                # nn.BatchNorm2d(num_features)
                skip_block.add_module(f"BatchNorm2d_Skip_{i}", nn.BatchNorm2d(out_channels_dim))
                # nn.LeakyReLU(negative_slope)
                skip_block.add_module(f"LeakyReLU_Skip_{i}", nn.LeakyReLU(0.2, inplace=True))

            ### deeper block ###
            out_channels_dim = channels_down[i]
            to_pad = int((kernel_size_down[i] - 1) / 2)
            if pad == 'reflection':
                deeper_block.add_module(f"ReflectionPad_Deeper_{i}_1", nn.ReflectionPad2d(to_pad))
                to_pad = 0
            deeper_block.add_module(f"Conv2d_Deeper_{i}_1", nn.Conv2d(in_channels_dim, out_channels_dim, kernel_size_down[i], 2, padding=to_pad, bias=need_bias))
            deeper_block.add_module(f"BatchNorm2d_Deeper_{i}_1", nn.BatchNorm2d(out_channels_dim))
            deeper_block.add_module(f"LeakyReLU_Deeper_{i}_1", nn.LeakyReLU(0.2, inplace=True))

            to_pad = int((kernel_size_down[i] - 1) / 2)
            if pad == 'reflection':
                deeper_block.add_module(f"ReflectionPad_Deeper_{i}_2", nn.ReflectionPad2d(to_pad))
                to_pad = 0
            deeper_block.add_module(f"Conv2d_Deeper_{i}_2", nn.Conv2d(out_channels_dim, out_channels_dim, kernel_size_down[i], 1, padding=to_pad, bias=need_bias))
            deeper_block.add_module(f"BatchNorm2d_Deeper_{i}_2", nn.BatchNorm2d(out_channels_dim))
            deeper_block.add_module(f"LeakyReLU_Deeper_{i}_2", nn.LeakyReLU(0.2, inplace=True))

            ### middle block ###
            deeper_model_internal_block = nn.Sequential()
            
            k = None
            if i == last_scale:
                # The deepest
                k = channels_down[i]
            else:
                deeper_block.add_module(f"Deeper_Internal_Block_{i}", deeper_model_internal_block)
                # because the deeper_model_internal_block will have channels_up[i + 1] output channels
                k = channels_up[i + 1]

            # Upsample doesn't change the channel size. W and H is multipled by the scale_factor.
            deeper_block.add_module(f"Upsample_{i}", nn.Upsample(scale_factor=2, mode=upsample_mode[i]))


            ########### Encoder Part ###########
            in_channels_dim = channels_skip[i] + k
            out_channels_dim = channels_up[i]
            ### shallower block ###
            model_internal_block.add_module(f"BatchNorm2d_Encoder_{i}_1", nn.BatchNorm2d(in_channels_dim))

            to_pad = int((kernel_size_up[i] - 1) / 2)
            if pad == 'reflection':
                model_internal_block.add_module(f"ReflectionPad_Encoder_{i}", nn.ReflectionPad2d(to_pad))
                to_pad = 0
            model_internal_block.add_module(f"Conv2d_Encoder_{i}", nn.Conv2d(in_channels_dim, out_channels_dim, kernel_size_up[i], 1, padding=to_pad, bias=need_bias))
            model_internal_block.add_module(f"BatchNorm2d_Encoder_{i}_2", nn.BatchNorm2d(out_channels_dim))
            model_internal_block.add_module(f"LeakyReLU_Encoder_{i}", nn.LeakyReLU(0.2, inplace=True))

            if need1x1_up:
                to_pad = 0
                if pad == 'reflection':
                    model_internal_block.add_module(f"ReflectionPad_1x1_{i}", nn.ReflectionPad2d(to_pad))
                model_internal_block.add_module(f"Conv2d_1x1_{i}", nn.Conv2d(out_channels_dim, out_channels_dim, 1, 1, padding=to_pad, bias=need_bias))
                model_internal_block.add_module(f"BatchNorm2d_1x1_{i}", nn.BatchNorm2d(out_channels_dim))
                model_internal_block.add_module(f"LeakyReLU_1x1_{i}", nn.LeakyReLU(0.2, inplace=True))

            in_channels_dim = channels_down[i]
            model_internal_block = deeper_model_internal_block

        to_pad = 0
        if pad == 'reflection':
            self.model.add_module(f"ReflectionPad_{n_scales}", nn.ReflectionPad2d(to_pad))
        self.model.add_module(f"Conv2d_{n_scales}", nn.Conv2d(channels_up[0], num_output_channels, 1, 1, padding=to_pad, bias=need_bias))
        if need_sigmoid:
            self.model.add_module(f"Sigmoid_{n_scales}", nn.Sigmoid())

        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def forward(self, input):

        return self.model(input)

