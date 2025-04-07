import torch


class DictTransforms:
    def __init__(self,
                 dict_transform : dict,
                 ):
        self.dict_transform = dict_transform

    def __call__(self, sample):
        # Apply your transforms to the 'image' key
        for key, function in self.dict_transform.items():
            sample[key] = function(sample[key])
        return sample


class SelectChannels:
    def __init__(self, channels):
        self.channels = channels

    def __call__(self, tensor):
        return tensor[self.channels]


class Unsqueeze:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, tensor):
        return tensor.unsqueeze(dim=self.dim)


class ConvertType:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, tensor):
        return tensor.to(self.dtype)


class AddMeanChannels:
    """
    Add missing channels to the tensor based on the mean values. Results in zeros after standardization.
    """
    def __init__(self, mean):
        self.mean = mean
        self.mean_tensor = None

    def __call__(self, tensor):
        if self.mean_tensor is None:
            # Init tensor with mean values
            self.mean_tensor = (torch.ones([len(self.mean) - len(tensor), *tensor.shape[1:]]) *
                                torch.tensor(self.mean)[len(tensor):, None, None])
        # Add mean values for missing channels
        tensor = torch.concat([tensor, self.mean_tensor])
        return tensor
