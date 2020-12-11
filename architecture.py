import torch.nn as nn

class ColorizerModel(nn.Module):

    def __init__(self, input_size=128):
        super(ColorizerModel, self).__init__()

        self.upsample = nn.Sequential()


    def forward(self, input):

        output = self.upsample(input)
        return input
