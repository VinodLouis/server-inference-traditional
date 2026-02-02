from torchvision import transforms


class Transfrormer:
    
    def __init__(self):
        self.cifar_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
