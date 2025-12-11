import typing
import torch

import iara.ml.models.base_model as iara_model
import iara.ml.models.mlp as iara_mlp
import iara.utils

class CNN(iara_model.BaseModel):

    def __init__(self,
                 input_shape: typing.Iterable[int],

                 conv_n_neurons: typing.List[int],
                 conv_activation: torch.nn.Module = torch.nn.ReLU,
                 conv_pooling: typing.Optional[torch.nn.Module] = torch.nn.MaxPool2d,
                 conv_pooling_size: typing.List[int] = [2,2],
                 conv_dropout: float = 0.5,
                 batch_norm: typing.Optional[torch.nn.Module] = torch.nn.BatchNorm2d,
                 kernel_size: int = 5,
                 padding: int = None,

                 classification_n_neurons: typing.Union[int, typing.Iterable[int]] = 128,
                 n_targets: int = 1,
                 classification_dropout: float = 0,
                 classification_norm: typing.Optional[torch.nn.Module] = torch.nn.BatchNorm1d,
                 classification_hidden_activation: torch.nn.Module = None,
                 classification_output_activation: torch.nn.Module = None):
        super().__init__()


        classification_hidden_activation = conv_activation if classification_hidden_activation is None else classification_hidden_activation
        padding = padding if padding is not None else int((kernel_size-1)/2)

        if len(input_shape) != 3:
            raise UnboundLocalError(f"CNN expects as input an image in the format: \
                                    channel x width x height (current {input_shape})")

        self.input_shape = input_shape

        conv_layers = []
        conv = [self.input_shape[0]]
        conv.extend(conv_n_neurons)

        for i in range(1, len(conv)):
            conv_layers.append(torch.nn.Conv2d(conv[i - 1], conv[i],
                                               kernel_size=kernel_size, padding=padding))
            if batch_norm is not None:
                conv_layers.append(batch_norm(conv[i]))
            if conv_dropout != 0 and i != 0:
                conv_layers.append(torch.nn.Dropout2d(p=conv_dropout))
            conv_layers.append(conv_activation())
            if conv_pooling is not None:
                conv_layers.append(conv_pooling(*conv_pooling_size))

        self.conv_layers = torch.nn.Sequential(*conv_layers)

        test_shape = [1]
        test_shape.extend(input_shape)
        test_tensor = torch.rand(test_shape, dtype=torch.float32)
        device = next(self.parameters()).device
        test_tensor = test_tensor.to(device)
        self.conv_layers = self.conv_layers.to(device)

        test_tensor = self.to_feature_space(test_tensor)

        self.mlp = iara_mlp.MLP(input_shape = test_tensor.shape,
                        hidden_channels = classification_n_neurons,
                        norm_layer = classification_norm,
                        n_targets = n_targets,
                        activation_layer = classification_hidden_activation,
                        activation_output_layer = classification_output_activation,
                        dropout=classification_dropout)


    def to_feature_space(self, data: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(data)


    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = self.to_feature_space(data)
        data = self.mlp(data)
        return data