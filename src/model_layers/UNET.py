import functools

import torch
from torch import nn

class BatchNorm(nn.Module):
    """
    A batch normalization module.

    Args:
        num_features (int): Number of features in the input tensor.
        epsilon (float, optional): A small value added to the denominator for numerical stability. Default is 0.001.
        dim (int, optional): The dimension along which normalization is performed. Default is 1.

    Attributes:
        dim (int): The dimension along which normalization is performed.
        gamma (nn.Parameter): Learnable scale parameter.
        beta (nn.Parameter): Learnable shift parameter.
        running_mean (nn.Parameter): Running mean of the input.
        running_var (nn.Parameter): Running variance of the input.
        epsilon (float): A small value added to the denominator for numerical stability.

    Methods:
        forward(x): Performs batch normalization on the input tensor x.

    Examples:
        # Create a BatchNorm instance with 64 input features
        bn = BatchNorm(64)
        # Apply batch normalization to an input tensor x
        output = bn(x)
    """
    
    def __init__(self, num_features, epsilon=0.001, dim=1):
        super(BatchNorm, self).__init__()
        self.dim = dim
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.running_mean = nn.Parameter(torch.ones(num_features))
        self.running_var = nn.Parameter(torch.zeros(num_features))
        self.epsilon = epsilon

    def forward(self, x):
        x = x.permute([0, 2, 3, 1])
        x = (self.gamma * (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)) + self.beta
        x = x.permute([0, 3, 1, 2])
        return x
        
class UNET(nn.Module):
    """
    Implementation of a U-Net model for image segmentation.
    With reference to https://github.com/UCRajkumar/ecSeg/blob/master/src/model_layers/models.py

    Attributes:
        num_encoder_blocks (int): Number of encoder blocks.
        num_sequential_blocks (int): Number of sequential blocks.
        num_decoder_blocks (int): Number of decoder blocks.
        output_channels (int): Number of output channels.

    Methods:
        forward(x): Forward pass through the U-Net model.
        load_weights(val_model): Load weights from a pre-trained model.

    Examples:
        # Create a UNET instance
        model = UNET()
        # Forward pass
        output = model(x)
    """

    def __init__(self):
        super(UNET, self).__init__()
        
        self.num_encoder_blocks = 4
        self.num_sequential_blocks = 5
        self.num_decoder_blocks = 4
        self.output_channels = 4

        get_encoder_block_channels = lambda input_: [(input_, input_*2), (input_*2, input_*2), (input_, input_*2)]
        get_decoder_block_channels = lambda input_: [(input_, input_//2), (input_//2, input_//2), (input_, input_//2)]

        initial_block_channels = [(1, 32), (32, 32)]
        encoder_channels = functools.reduce(lambda a,b:a+b, [get_encoder_block_channels(initial_block_channels[-1][-1]*(2**i)) for i in range(self.num_encoder_blocks)])
        sequential_channels = [(encoder_channels[-1][-1], encoder_channels[-1][-1]) for _ in range(self.num_sequential_blocks)]
        decoder_channels = functools.reduce(lambda a,b:a+b, [get_decoder_block_channels(encoder_channels[-1][-1]//(2**i)) for i in range(self.num_decoder_blocks)])
        final_block_channels = [(decoder_channels[-1][-1], self.output_channels)]
        
        channels = initial_block_channels + encoder_channels + sequential_channels + decoder_channels + final_block_channels
        dilation = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (4, 4), (8, 8), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1),(1, 1)]
        kernel_size = [3 if not i in (19, len(dilation)) else 1 for i in range(1,len(dilation)+1)]
        padding =  ['same' if i != len(dilation) else 'valid' for i in range(1,len(dilation)+1)]
        
        channels_t = [(sequential_channels[-1][-1] >> i, sequential_channels[-1][-1] >> (i+1)) for i in range(self.num_decoder_blocks)]
    
        assert len(dilation) == len(channels)
        self.conv2d = nn.ModuleList([nn.Conv2d(i, j, k, padding=p, dilation=d, bias=True) for k, d, p, (i, j) in zip(kernel_size, dilation, padding, channels)])
        self.batch_norm = nn.ModuleList([BatchNorm(j) for i, j in channels[:-1]])
        
        self.conv2d_transpose = nn.ModuleList([nn.ConvTranspose2d(i, j, 2, stride=2, padding=0) for i, j in channels_t])
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.3)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2)
        self.activation = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = x/255 # lambda layer
        inter = self.conv2d[0](x)
        inter = self.batch_norm[0](inter)
        inter = self.leaky_relu(inter)
        
        inter = self.conv2d[1](inter)
        inter = self.batch_norm[1](inter)
        
        output = x + inter
        output = self.leaky_relu(output)
        save_states = [[2, output]]
        
        ### Encoder Block
        module_index = 2
        for _ in range(self.num_encoder_blocks):
            input_pooled = self.max_pool(save_states[-1][1])
            conv_1_output = self.conv2d[module_index](input_pooled)
            conv_1_output = self.batch_norm[module_index](conv_1_output)
            conv_1_output = self.leaky_relu(conv_1_output)

            conv_2_output = self.conv2d[module_index+1](conv_1_output)
            conv_2_output = self.batch_norm[module_index+1](conv_2_output)

            conv_3_output = self.conv2d[module_index+2](input_pooled)
            conv_3_output = self.batch_norm[module_index+2](conv_3_output)

            output = conv_2_output + conv_3_output
            output = self.leaky_relu(output)
            save_states.append([module_index+3, output])
            module_index += 3
        
        ### Sequential Block
        for _ in range(self.num_sequential_blocks):
            output = self.conv2d[module_index](output)
            output = self.batch_norm[module_index](output)
            output = self.leaky_relu(output)
            module_index += 1
        
        ### Decoder Block
        for i, (j, prev_layer_output) in enumerate(save_states[:-1][::-1]):
            output = self.conv2d_transpose[i](output)
            output = torch.cat([output, prev_layer_output], axis=1)

            if i == self.num_decoder_blocks - 1:
                return output

            inter = self.conv2d[module_index](output)
            inter = self.batch_norm[module_index](inter)
            inter = self.leaky_relu(inter)
            
            inter = self.conv2d[module_index+1](inter)
            inter = self.batch_norm[module_index+1](inter)

            output = self.conv2d[module_index+2](output)
            output = self.batch_norm[module_index+2](output)

            output = inter + output
            output = self.leaky_relu(output)
            module_index += 3
        
        output = self.conv2d[module_index](output)
        return self.activation(output)

    def load_weights(self, val_model):
        for i, conv_layer in enumerate(self.conv2d, start=1):
            conv_layer.weight.data = torch.Tensor(val_model.get_layer(f'conv2d_{i}').weights[0].numpy()).permute([3, 2, 0, 1])
            conv_layer.bias.data = torch.Tensor(val_model.get_layer(f'conv2d_{i}').weights[1].numpy())
        
        for i, batch_norm in enumerate(self.batch_norm, start=1):
            batch_norm.gamma.data = torch.Tensor(val_model.get_layer(f'batch_normalization_{i}').weights[0].numpy())
            batch_norm.beta.data = torch.Tensor(val_model.get_layer(f'batch_normalization_{i}').weights[1].numpy())
            batch_norm.running_mean.data = torch.Tensor(val_model.get_layer(f'batch_normalization_{i}').weights[2].numpy())
            batch_norm.running_var.data = torch.Tensor(val_model.get_layer(f'batch_normalization_{i}').weights[3].numpy())

        for i, conv_transpose_layer in enumerate(self.conv2d_transpose, start=1):
            conv_transpose_layer.weight.data = torch.Tensor(val_model.get_layer(f'conv2d_transpose_{i}').weights[0].numpy()).permute([3, 2, 0, 1])
            conv_transpose_layer.bias.data = torch.Tensor(val_model.get_layer(f'conv2d_transpose_{i}').weights[1].numpy())