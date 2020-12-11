# Numpy
import numpy as np

# PyTorch
import torch
from torchvision import datasets

# Color
from skimage import color


class LABImageFolder(datasets.ImageFolder):
    """ LABImageFolder: Derives datasets.ImageFolder
    Arguments: Path (string), Transforms (transforms.Compose)
    """

    def __getitem__(self, index):
        """ Loads RGB images from specified folder,
            applies specified transforms,
            converts RGB to L*A*B

        Returns: luminance image (tensor of shape (H, W, 1)),
            ab image (tensor of shape (H, W, 2)),
            target class (int)
        """

        # Get the path string and class of the image, then load it
        path, target = self.imgs[index]
        img_rgb = self.loader(path)

        if self.transform is not None:
            # Apply transformations
            img_rgb = self.transform(img_rgb)

        # Reorder shape from (3, H, W) -> (H, W, 3)
        img_rgb = np.transpose(img_rgb, axes=[1, 2, 0])

        # Convert from RGB to LAB
        img_lab = color.rgb2lab(img_rgb)

        # Extract ab and luminance channels
        img_ab = img_lab[:, :, 1:3]
        img_l = img_lab[:, :, 0]

        # Convert channel data to tensors
        img_ab = torch.from_numpy(img_ab)
        # Shape (H, W) -> (H, W, 1)
        img_l = torch.from_numpy(img_l).unsqueeze(2)

        return img_l, img_ab, target
