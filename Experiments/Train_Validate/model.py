import torch.nn as nn
import segmentation_models_pytorch as smp
class AttentionUNet(nn.Module):
    def __init__(self, encoder_name='resnet34', pretrained=True):
        super(AttentionUNet, self).__init__()
        # Use a pre-trained encoder (either ResNet or EfficientNet)
        self.model = smp.Unet(
            encoder_name=encoder_name,        # Choose the encoder ('resnet34', 'efficientnet-b3', etc.)
            encoder_weights='imagenet' if pretrained else None,  # Use ImageNet weights if pretrained
            in_channels=3,                    # Number of input channels (RGB images)
            classes=3,                        # Number of output channels (RGB output)
            decoder_attention_type='scse'     # Use spatial and channel-wise attention
        )
        
    def forward(self, x):
        return self.model(x)
