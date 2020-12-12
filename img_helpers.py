import torch
from skimage import color


def tensor_lab_to_rgb(img_l, img_ab):
    """ Converts LAB image to RGB

    Inputs: img_l shape = (1, H, W)
            img_ab shape = (2, H, W)

    Output: shape = (H, W, 3)
    """
    assert img_l.shape[0] == 1
    assert img_ab.shape[0] == 2

    temp_l = img_l.permute(1, 2, 0)
    temp_ab = img_ab.permute(1, 2, 0)

    temp_lab = torch.cat((temp_l, temp_ab), dim=2)

    assert temp_lab.shape[2] == 3

    return color.lab2rgb(temp_lab)
