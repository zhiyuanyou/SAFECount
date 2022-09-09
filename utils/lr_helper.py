from torch.optim.lr_scheduler import ExponentialLR, StepLR


def get_scheduler(optimizer, config):
    if config["type"] == "StepLR":
        return StepLR(optimizer, **config["kwargs"])
    elif config["type"] == "ExponentialLR":
        return ExponentialLR(optimizer, **config["kwargs"])
    else:
        raise NotImplementedError
