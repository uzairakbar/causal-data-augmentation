import torch


MODELS = {
    'linear': lambda input_dim: torch.nn.Sequential(
        torch.nn.Linear(input_dim, 1, bias=False)
    ),
    '2-layer': lambda input_dim: torch.nn.Sequential(
        torch.nn.Linear(input_dim, 20),
        torch.nn.LeakyReLU(0.2),
        torch.nn.Linear(20, 1)
    ),
    'cmnist': lambda input_dim: torch.nn.Sequential(
        torch.nn.Linear(input_dim, 256),
        torch.nn.ReLU(True),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(True),
        torch.nn.Linear(256, 1),
        torch.nn.Sigmoid()
    ),
    'rmnist': lambda input_dim: torch.nn.Sequential(
        torch.nn.Unflatten(1, torch.Size([1, 28, 28])),
        torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, bias=False),
        torch.nn.BatchNorm2d(32), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
        torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False),
        torch.nn.BatchNorm2d(64), torch.nn.ReLU(), torch.nn.MaxPool2d(2, 2),
        torch.nn.Flatten(1, -1),
        torch.nn.Linear(64 * 4 * 4, 128),
        torch.nn.Dropout(0.5),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10),
        torch.nn.LogSoftmax(dim=1)
    ),
}
