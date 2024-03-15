import torch
from torch import nn

class RPN(nn.Module):
    """Initialize a region proposal network
    With reference to https://github.com/endernewton/tf-faster-rcnn/blob/master/lib/nets/network.py
    ,https://github.com/tryolabs/luminoth/blob/master/luminoth/models/fasterrcnn/rpn.py and
    https://github.com/UCRajkumar/ecSeg/blob/master/src/model_layers/model_RPN.py

    Args:
        feat_map (tensor): The parameters of the genome.
        num_refanchors (num, optional): Number of reference anchors placed on the feature map. Default is 9
        unet_dimension (num, optional): The number of dimensions in feature map. Default is 64.
        hidden_size (num, optional): The number of outputs in hidden layers. Default is 512.
        use_activation_function (bool): use activation or not

    Methods:
        forward(x): Gets RPN output on feature map x.

    Examples:
        # Create an RPN instance with 9 reference anchors and 64 dimensions in feat map
        rpn = RPN(9, 64)
        # Get RPN of image x
        rpn_output = rpn(x)
        

    """

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