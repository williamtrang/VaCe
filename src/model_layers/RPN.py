import torch
from torch import nn

class RPN(nn.Module):
    def __init__(self, num_refanchors=9, unet_dimension = 64, hidden_size=512, use_activation_function=False):
        super(RPN, self).__init__()
        
        self.rpn_layer_1 = nn.Conv2d(unet_dimension, hidden_size, 3, padding='same')
        self.rpn_layer_2 = nn.Conv2d(hidden_size, num_refanchors, 1, padding='valid')
        self.activation = nn.GELU()
        #self.sigmoid = nn.Sigmoid()
        self.use_activation_function = use_activation_function
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0005)

                                                
    def forward(self, feat_map):
        intermediate_output = self.rpn_layer_1(feat_map)
        if self.use_activation_function:
            intermediate_output = self.activation(intermediate_output)
        intermediate_output = self.rpn_layer_2(intermediate_output)
        
        return intermediate_output