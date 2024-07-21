from .resnet import ResNet18
from .convnext import convnextv2_atto_cifar
from .mlp_mixer import MLPMixer

def build_model(model):
    if model == 'resnet-18':
        return ResNet18()
    elif model == 'mlp_mixer':
        return MLPMixer(in_channels=3, image_size=32, patch_size=4, num_classes=10,
                    dim=128, depth=8, token_dim=256, channel_dim=512)
    elif model == 'convnext':
        return convnextv2_atto_cifar(num_classes=10)
    else:
        Exception("Model type not supported!")