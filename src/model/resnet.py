import torchvision
from torch import nn

class ResNet(nn.Module):
    def __init__(self, input_channels, n_class):
        self.super().__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(
            input_channels,
            self.resnet.conv1.out_channels,
            self.resnet.conv1.kernel_size,
            self.resnet.conv1.stride,
            self.resnet.conv1.padding,
            bias=self.resnet.conv1.bias,
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, n_class)

    def forward(self, data_object, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        return {"logits": self.net(data_object)}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
