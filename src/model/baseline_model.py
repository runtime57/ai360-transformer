from torch import nn
from torch.nn import Sequential
import torch

class BaselineModel(nn.Module):
    """
    Simple MLP
    """

    def __init__(self, vocabulary_size, num_layers = 10, d_model = 512, nhead = 8, dropout = 0.1):
        """
        Args:
            vocabulary_size,
            d_model,
            nhead,
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.vocabulary_size = vocabulary_size
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=4 * self.d_model,
            dropout=dropout,
            batch_first=True
        )


        self.maskedencoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )

        self.lastlinear = nn.Sequential(
            nn.Linear(self.d_model, self.vocabulary_size),
        ) # logits are output

        self.embedings = nn.Embedding(self.vocabulary_size, d_model)

    def forward(self, data_object, **batch):
        """
        Model forward method.

        Args:
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """

        context_size = data_object.shape[1]
        data_object = self.embedings(data_object)

        mask = nn.Transformer.generate_square_subsequent_mask(context_size)

        prelogits = self.maskedencoder(
            src=data_object,
            mask=mask
        )
        logits = self.lastlinear(prelogits)

        return {"logits" : logits}


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
