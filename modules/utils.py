def to_numpy(x):
    return x.detach().cpu().numpy()  # convert a torch tensor to a numpy array
