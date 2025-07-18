import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import Union, List, Dict, Any, cast
import torch.nn.functional as F
import math

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
class Classifier(nn.Module):
    def __init__(self, input_size=1024, output_size=2):
        super(Classifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        y_pred = self.linear1(x)
        return y_pred


class ProjectionHead(nn.Module):
    def __init__(self, input_size=1024, nn_size=256, encoding_size=100):
        super(ProjectionHead, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, nn_size)
        self.linear2 = torch.nn.Linear(nn_size, encoding_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


class EvaClassifier(nn.Module):
    def __init__(self, input_size=1024, nn_size=512, output_size=2):
        super(EvaClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, nn_size)
        self.linear2 = torch.nn.Linear(nn_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


class AccNet(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 5,
        classifier_input_size: int = 1,
        classifier_layer_size: int = 2048,
        init_weights: bool = True,
    ) -> None:
        super(AccNet, self).__init__()
        self.features = features

        if init_weights:
            self._initialize_weights()

    def forward(self, x) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = cnn1()

    def forward(self, x):
        feats = self.feature_extractor(x)

        return feats


class SSLNET(nn.Module):
    def __init__(
        self, output_size=2, input_size=1024, number_nn=1024, flatten_size=1024
    ):
        super(SSLNET, self).__init__()
        self.feature_extractor = cnn1()
        self.rotation_classifier = Classifier(
            input_size=flatten_size, output_size=output_size
        )

    def forward(self, x):
        feats = self.feature_extractor(x)
        rotation_y = self.rotation_classifier(feats)
        # axis_y = self.axis_classifier(feats)

        return rotation_y


def make_layers(
    cfg: List[Union[str, int]], batch_norm: bool = False
) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    i = 0
    while i < len(cfg):
        v = cfg[i]
        if v == "M":
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            i += 1
            my_kernel_size = cfg[i]

            v = cast(int, v)
            my_kernel_size = cast(int, my_kernel_size)

            conv1d = nn.Conv1d(
                in_channels, v, kernel_size=my_kernel_size, padding=1
            )
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv1d, nn.ReLU(inplace=True)]
            in_channels = v
        i += 1
    return nn.Sequential(*layers)


class ConvBNReLU(nn.Module):
    """Convolution + batch normalization + ReLU is a common trio"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
    ):
        super(ConvBNReLU, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=bias,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.main(x)


class CNN(nn.Module):
    """Typical CNN design with pyramid-like structure"""

    def __init__(self, in_channels=3, num_filters_init=3):
        super(CNN, self).__init__()

        self.layer1 = ConvBNReLU(
            in_channels, num_filters_init, 3, 1, 1, bias=False
        )  # 900 -> 225
        self.layer2 = ConvBNReLU(
            num_filters_init, num_filters_init, 3, 1, 1, bias=False
        )  # 225 -> 56

        self.layer3 = ConvBNReLU(
            num_filters_init, num_filters_init * 2, 3, 1, 1, bias=False
        )  # 56 -> 14
        self.layer4 = ConvBNReLU(
            num_filters_init * 2, num_filters_init * 2, 3, 1, 1, bias=False
        )  # 14 -> 7

        self.layer5 = ConvBNReLU(
            num_filters_init * 2, num_filters_init * 4, 12, 1, 1, bias=False
        )  # 7 -> 3
        self.layer6 = ConvBNReLU(
            num_filters_init * 4, num_filters_init * 4, 12, 1, 1, bias=False
        )  # 6 -> 1

        self.layer7 = ConvBNReLU(
            num_filters_init * 4, num_filters_init * 8, 36, 1, 1, bias=False
        )  # 6 -> 1
        self.layer8 = ConvBNReLU(
            num_filters_init * 8, num_filters_init * 8, 36, 1, 1, bias=False
        )  # 7 -> 3

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.layer1(x)
        # print(out.size())

        out = self.layer2(out)
        # print(out.size())
        out = self.max_pool(out)

        out = self.layer3(out)
        # print(out.size())

        out = self.layer4(out)
        out = self.max_pool(out)

        out = self.layer5(out)
        # print(out.size())
        out = self.layer6(out)
        out = self.max_pool(out)
        out = self.layer7(out)

        out = self.layer8(out)
        out = self.max_pool(out)

        out = torch.flatten(out, 1)
        return out


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes=2,
        in_cnn_channels=3,
        num_cnn_filters_init=32,
        lstm_layer=3,
        lstm_nn_size=1024,
        model_device="cpu",
        dropout_p=0,
        bidrectional=False,
        batch_size=10,
    ):
        super(CNNLSTM, self).__init__()
        self.feature_extractor = CNN(
            in_channels=in_cnn_channels, num_filters_init=num_cnn_filters_init
        )
        self.lstm = nn.LSTM(
            input_size=4608,
            hidden_size=lstm_nn_size,
            num_layers=lstm_layer,
            bidirectional=bidrectional,
        )
        if bidrectional is True:
            fc_feature_size = lstm_nn_size * 2
        else:
            fc_feature_size = lstm_nn_size
        self.fc1 = nn.Linear(fc_feature_size, fc_feature_size)
        self.dropout_layer = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(fc_feature_size, num_classes)
        self.fc_feature_size = fc_feature_size
        self.model_device = model_device

        self.lstm_layer = lstm_layer
        self.batch_size = batch_size
        self.lstm_nn_size = lstm_nn_size
        self.bidrectional = bidrectional

    def init_hidden(self, batch_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        init_lstm_layer = self.lstm_layer
        if self.bidrectional:
            init_lstm_layer = self.lstm_layer * 2
        hidden_a = torch.randn(
            init_lstm_layer,
            batch_size,
            self.lstm_nn_size,
            device=self.model_device,
        )
        hidden_b = torch.randn(
            init_lstm_layer,
            batch_size,
            self.lstm_nn_size,
            device=self.model_device,
        )

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, x, seq_lengths):
        # x dim: batch_size x C x F_1
        # we will need to do the packing of the sequence
        # dynamically for each batch of input
        # 1. feature extractor

        x = self.feature_extractor(x)  # x dim: total_epoch_num * feature size
        feature_size = x.size()[-1]

        # 2. lstm
        seq_tensor = torch.zeros(
            len(seq_lengths),
            seq_lengths.max(),
            feature_size,
            dtype=torch.float,
            device=self.model_device,
        )
        start_idx = 0
        for i in range(len(seq_lengths)):
            current_len = seq_lengths[i]
            current_series = x[
                start_idx : start_idx + current_len, :
            ]  # num_time_step x feature_size
            current_series = current_series.view(
                1, current_series.size()[0], -1
            )
            seq_tensor[i, :current_len, :] = current_series
            start_idx += current_len

        seq_lengths_ordered, perm_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[perm_idx]
        packed_input = pack_padded_sequence(
            seq_tensor, seq_lengths_ordered.cpu().numpy(), batch_first=True
        )

        # x dim for lstm: #  batch_size_rnn x Sequence_length x F_2
        # uncomment for random init state
        # hidden = self.init_hidden(len(seq_lengths))
        packed_output, _ = self.lstm(packed_input)
        output, input_sizes = pad_packed_sequence(
            packed_output, batch_first=True
        )

        # reverse back to the original order
        _, unperm_idx = perm_idx.sort(0)
        lstm_output = output[unperm_idx]

        # reverse back to the originaly shape
        # total_epoch_num * fc_feature_size
        fc_tensor = torch.zeros(
            seq_lengths.sum(),
            self.fc_feature_size,
            dtype=torch.float,
            device=self.model_device,
        )

        start_idx = 0
        for i in range(len(seq_lengths)):
            current_len = seq_lengths[i]
            current_series = lstm_output[
                i, :current_len, :
            ]  # num_time_step x feature_size
            current_series = current_series.view(current_len, -1)
            fc_tensor[start_idx : start_idx + current_len, :] = current_series
            start_idx += current_len
        # print("lstm time: ", end-start)

        # 3. linear readout
        x = self.fc1(fc_tensor)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [
        64,
        3,
        64,
        3,
        "M",
        128,
        3,
        128,
        3,
        128,
        3,
        "M",
        256,
        3,
        256,
        3,
        256,
        3,
        256,
        3,
        "M",
        512,
        3,
        512,
        3,
        512,
        3,
        512,
        3,
        "M",
        512,
        3,
        512,
        3,
        512,
        3,
        512,
        3,
        "M",
        1024,
        30,
    ],  # converted one FC to ConV
    "B": [
        64,
        12,
        64,
        12,
        "M",
        128,
        24,
        128,
        24,
        "M",
        256,
        24,
        256,
        24,
        "M",
        512,
        24,
        512,
        24,
        "M",
        512,
        24,
        512,
        48,
        "M",
    ],
    "C": [
        64,
        12,
        64,
        12,
        "M",
        128,
        24,
        128,
        24,
        "M",
        256,
        24,
        256,
        24,
        "M",
        512,
        48,
        512,
        48,
        "M",
        512,
        48,
        512,
        92,
        "M",
    ],
    "D": [32, 12, 64, 12, "M", 128, 24, 128, 48, "M", 256, 48, 256, 96, "M"],
}


def _cnn(
    cfg: str, batch_norm: bool, pretrained: bool, **kwargs: Any
) -> AccNet:
    if pretrained:
        kwargs["init_weights"] = False
    model = AccNet(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def cnn1(pretrained: bool = False, **kwargs: Any) -> AccNet:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale
    Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    classifier_input_size = 28  # this shouldn't change
    return _cnn(
        "A",
        True,
        pretrained,
        classifier_input_size=classifier_input_size,
        **kwargs,
    )


def cnn3(pretrained: bool = False, **kwargs: Any) -> AccNet:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale
    Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    classifier_input_size = 32  # this shouldn't change
    return _cnn(
        "B",
        False,
        pretrained,
        classifier_input_size=classifier_input_size,
        **kwargs,
    )


def cnn5(pretrained: bool = False, **kwargs: Any) -> AccNet:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale
     Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    classifier_input_size = 42  # this shouldn't change
    return _cnn(
        "C",
        False,
        pretrained,
        classifier_input_size=classifier_input_size,
        **kwargs,
    )


def cnnSmall(pretrained: bool = False, **kwargs: Any) -> AccNet:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale
    Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`._
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    classifier_input_size = 12  # this shouldn't change
    classifier_layer_size = 1000
    return _cnn(
        "D",
        False,
        pretrained,
        classifier_input_size=classifier_input_size,
        classifier_layer_size=classifier_layer_size,
        **kwargs,
    )


class Downsample(nn.Module):
    r"""Downsampling layer that applies anti-aliasing filters.
    For example, order=0 corresponds to a box filter (or average downsampling
    -- this is the same as AvgPool in Pytorch), order=1 to a triangle filter
    (or linear downsampling), order=2 to cubic downsampling, and so on.
    See https://richzhang.github.io/antialiased-cnns/ for more details.
    """

    def __init__(self, channels=None, factor=2, order=1):
        super(Downsample, self).__init__()
        assert factor > 1, "Downsampling factor must be > 1"
        self.stride = factor
        self.channels = channels
        self.order = order

        # Figure out padding and check params make sense
        # The padding is given by order*(factor-1)/2
        # so order*(factor-1) must be divisible by 2
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0, (
            "Misspecified downsampling parameters."
            "Downsampling factor and order must be such "
            "that order*(factor-1) is divisible by 2"
        )
        self.padding = int(order * (factor - 1) / 2)

        box_kernel = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box_kernel)
        kernel /= np.sum(kernel)
        kernel = torch.Tensor(kernel)
        self.register_buffer(
            "kernel", kernel[None, None, :].repeat((channels, 1, 1))
        )

    def forward(self, x):
        return F.conv1d(
            x,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1],
        )


class ResBlock(nn.Module):
    r""" Basic bulding block in Resnets:

       bn-relu-conv-bn-relu-conv
      /                         \
    x --------------------------(+)->

    """

    def __init__(
        self, in_channels, out_channels, kernel_size=5, stride=1, padding=2
    ):

        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)

        x = x + identity

        return x


class Resnet(nn.Module):
    r"""The general form of the architecture can be described as follows:

    x->[Conv-[ResBlock]^m-BN-ReLU-Down]^n->y

    In other words:

            bn-relu-conv-bn-relu-conv                        bn-
           /                         \                      /
    x->conv --------------------------(+)-bn-relu-down-> conv ----

    """

    def __init__(
        self,
        output_size=1,
        n_channels=3,
        is_eva=False,
        resnet_version=1,
        epoch_len=10,
        is_mtl=False,
        is_simclr=False,
        is_transformer=False,
    ):
        super(Resnet, self).__init__()

        # Architecture definition. Each tuple defines
        # a basic Resnet layer Conv-[ResBlock]^m]-BN-ReLU-Down
        # isEva: change the classifier to two FC with ReLu
        # For example, (64, 5, 1, 5, 3, 1) means:
        # - 64 convolution filters
        # - kernel size of 5
        # - 1 residual block (ResBlock)
        # - ResBlock's kernel size of 5
        # - downsampling factor of 3
        # - downsampling filter order of 1
        # In the below, note that 3*3*5*5*4 = 900 (input size)
        if resnet_version == 1:
            if epoch_len == 5:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 3, 1),
                    (512, 5, 0, 5, 3, 1),
                ]
            elif epoch_len == 10:
                cgf = [
                    (64, 5, 2, 5, 2, 2),
                    (128, 5, 2, 5, 2, 2),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 3, 1),
                ]
            else:
                cgf = [
                    (64, 5, 2, 5, 3, 1),
                    (128, 5, 2, 5, 3, 1),
                    (256, 5, 2, 5, 5, 1),
                    (512, 5, 2, 5, 5, 1),
                    (1024, 5, 0, 5, 4, 0),
                ]
        else:
            cgf = [
                (64, 5, 2, 5, 3, 1),
                (64, 5, 2, 5, 3, 1),
                (128, 5, 2, 5, 5, 1),
                (128, 5, 2, 5, 5, 1),
                (256, 5, 2, 5, 4, 0),
            ]  # smaller resnet
        in_channels = n_channels
        feature_extractor = nn.Sequential()
        for i, layer_params in enumerate(cgf):
            (
                out_channels,
                conv_kernel_size,
                n_resblocks,
                resblock_kernel_size,
                downfactor,
                downorder,
            ) = layer_params
            feature_extractor.add_module(
                f"layer{i+1}",
                Resnet.make_layer(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    n_resblocks,
                    resblock_kernel_size,
                    downfactor,
                    downorder,
                ),
            )
            in_channels = out_channels

        self.feature_extractor = feature_extractor
        self.is_mtl = is_mtl

        # Classifier input size = last out_channels in previous layer
        if is_eva:
            self.classifier = EvaClassifier(
                input_size=out_channels, output_size=output_size
            )
        elif is_transformer:
            self.classifier = EvaTransformerClassifier(
                input_size=out_channels,
                embed_dim=512,
                num_heads=8,
                num_layers=2,
                output_size=output_size,
            )
        elif is_mtl:
            self.aot_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.scale_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.permute_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
            self.time_w_h = Classifier(
                input_size=out_channels, output_size=output_size
            )
        elif is_simclr:
            self.classifier = ProjectionHead(
                input_size=out_channels, encoding_size=output_size
            )

        weight_init(self)

    @staticmethod
    def make_layer(
        in_channels,
        out_channels,
        conv_kernel_size,
        n_resblocks,
        resblock_kernel_size,
        downfactor,
        downorder=1,
    ):
        r""" Basic layer in Resnets:

        x->[Conv-[ResBlock]^m-BN-ReLU-Down]->

        In other words:

                bn-relu-conv-bn-relu-conv
               /                         \
        x->conv --------------------------(+)-bn-relu-down->

        """

        # Check kernel sizes make sense (only odd numbers are supported)
        assert (
            conv_kernel_size % 2
        ), "Only odd number for conv_kernel_size supported"
        assert (
            resblock_kernel_size % 2
        ), "Only odd number for resblock_kernel_size supported"

        # Figure out correct paddings
        conv_padding = int((conv_kernel_size - 1) / 2)
        resblock_padding = int((resblock_kernel_size - 1) / 2)

        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                conv_kernel_size,
                1,
                conv_padding,
                bias=False,
                padding_mode="circular",
            )
        ]

        for i in range(n_resblocks):
            modules.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    resblock_kernel_size,
                    1,
                    resblock_padding,
                )
            )

        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(True))
        modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)

    def forward(self, x):
        feats = self.feature_extractor(x)

        if self.is_mtl:
            aot_y = self.aot_h(feats.view(x.shape[0], -1))
            scale_y = self.scale_h(feats.view(x.shape[0], -1))
            permute_y = self.permute_h(feats.view(x.shape[0], -1))
            time_w_h = self.time_w_h(feats.view(x.shape[0], -1))
            return aot_y, scale_y, permute_y, time_w_h
        else:
            y = self.classifier(feats.view(x.shape[0], -1))
            return y
        return y


def weight_init(self, mode="fan_out", nonlinearity="relu"):

    for m in self.modules():

        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(
                m.weight, mode=mode, nonlinearity=nonlinearity
            )

        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(64, 64, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(128, 128, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(128, 256, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(256, 256, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(256, 512, 3, stride=3, padding=1),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(256, 256, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(256, 128, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 128, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(128, 64, 7, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 64, 7, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(
                64, 3, 5, stride=3, padding=3, output_padding=1
            ),
            nn.ReLU(True),
        )

        weight_init(self)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class EncoderMLP(nn.Module):
    def __init__(self, output_size):
        super(EncoderMLP, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(64, 64, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(128, 128, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(128, 256, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(256, 256, 5, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(256, 512, 3, stride=3, padding=1),
            nn.ReLU(True),
        )

        self.classifier = EvaClassifier(
            input_size=512, output_size=output_size
        )

        weight_init(self)

    def forward(self, x):
        feats = self.encoder(x)
        y = self.classifier(feats.view(x.shape[0], -1))
        return y

        
class IMUMLPClassifier(nn.Module):
    def __init__(self, input_dim=300*3, embed_dim=64, num_classes=5):
        super().__init__()
        self.flatten = nn.Flatten()
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        self.hidden_dim = hidden_dim = embed_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x_embedding = self.embedding(self.flatten(x))
        x_encoded = self.mlp(x_embedding)
        output = self.head(x_encoded)
        return output
    
class EvaTransformerClassifier(nn.Module):
    def __init__(self, input_size=1024, embed_dim=512, num_heads=8, num_layers=2, output_size=2, dropout=0.1):
        super(EvaTransformerClassifier, self).__init__()
        
        # Input projection to match transformer dimension
        self.input_projection = nn.Linear(input_size, embed_dim)
        
        # Positional encoding (learnable, for a fixed sequence length of 1)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True  # Use batch_first=True if PyTorch >= 1.9.0
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, output_size)
    
    def forward(self, x):
        # Input: [batch, input_size] (ResNet features)
        batch_size = x.size(0)
        
        # Project input to embedding dimension
        x = self.input_projection(x)  # [batch, embed_dim]
        x = x.unsqueeze(1)  # [batch, 1, embed_dim] - sequence length = 1
        
        # Add positional encoding
        x = x + self.pos_embedding
        
        # Apply transformer
        x = self.transformer_encoder(x)  # [batch, 1, embed_dim]
        
        # Remove sequence dimension
        x = x.squeeze(1)  # [batch, embed_dim]
        
        # Classification
        x = self.dropout(x)
        out = self.head(x)
        return out
    
class IMUTransformerClassifier(nn.Module):
    def __init__(self, input_dim=3, embed_dim=128, seq_length=300, num_heads=4, num_layers=2, num_classes=5):
        super().__init__()
        # Feature extraction components
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.pos_embedding = nn.Embedding(seq_length, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                dim_feedforward=embed_dim*4)  # batch_first=True not supported in PyTorch 1.7.0
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
    
    def feature_extractor(self, x):
        """Extract features before classification head"""
        B, T, _ = x.shape
        x = self.embedding(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_embedding(pos)
        x = x.transpose(0, 1)  # Transpose to (T, B, embed_dim)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # Transpose back to (B, T, embed_dim)
        features = x.mean(dim=1)  # Global average pooling over time
        return features
    
    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.head(features)
        return out

class DeepConvLSTM(nn.Module): 
    """Deep ConvLSTM model for time series classification.
    Args:
        input_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
    """
    def __init__(self, input_channels=3, num_classes=5):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)
    
    def feature_extractor(self, x):
        """Extract features before classification head"""
        # x shape: [batch, channels, time]
        x = self.conv_layers(x)
        # Transpose for LSTM: [batch, time, features]
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        # Take final time step
        features = x[:, -1, :]  # [batch, hidden_size]
        return features
    
    def forward(self, x):
        features = self.feature_extractor(x)
        x = self.dropout(features)
        return self.fc(x)

class AttnTCN(nn.Module): 
    """Attention-based Temporal Convolutional Network for time series classification.
    Args:
        input_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, input_channels=3, num_classes=5, num_heads=4, embed_dim=128):
        super().__init__()
        # Temporal convolutions
        self.tcn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, stride=1, padding=2, dilation=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, stride=1, padding=4, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=8, dilation=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        # Self-attention layer
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(128)
        # Classification head
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)
    
    def feature_extractor(self, x):
        """Extract features before classification head"""
        # x shape: [batch, channels, time]
        batch_size = x.size(0)
        # Apply TCN
        x = self.tcn(x)  # [batch, 128, time]
        # Prepare for self-attention (seq_len, batch, features)
        x = x.permute(2, 0, 1)
        # Apply self-attention
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output  # Residual connection
        x = self.layer_norm(x)
        # Global temporal pooling
        x = x.permute(1, 0, 2)  # [batch, time, features]
        features = torch.mean(x, dim=1)  # [batch, features]
        return features
    
    def forward(self, x):
        features = self.feature_extractor(x)
        x = self.dropout(features)
        return self.fc(x)
    
import math
import torch
import torch.nn as nn

class HARTransformer(nn.Module):  # NOT TESTED YET
    """Transformer model for Human Activity Recognition (HAR).
    Args:
        input_channels (int): Number of input channels.
        seq_length (int): Length of the input sequence.
        num_classes (int): Number of output classes.
        d_model (int): Dimension of the model.
    """
    def __init__(self, input_channels=3, seq_length=300, num_classes=5, d_model=128, num_heads=4, num_layers=4):
        super().__init__()
        # Initial embedding
        self.embedding = nn.Sequential(
            nn.Conv1d(input_channels, d_model, kernel_size=5, padding=2),
            nn.ReLU()
        )
        # Positional encoding
        self.register_buffer('pos_encoding', self._create_pos_encoding(seq_length, d_model))
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
    
    def _create_pos_encoding(self, seq_length, d_model):
        pos = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pos_enc = torch.zeros(seq_length, d_model)
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        return pos_enc
    
    def feature_extractor(self, x):
        """Extract features before classification head"""
        # x shape: [batch, channels, time]
        batch_size = x.size(0)
        
        # Initial embedding
        x = self.embedding(x)  # [batch, d_model, time]
        x = x.permute(0, 2, 1)  # [batch, time, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding.unsqueeze(0)
        
        # Transformer
        x = self.transformer(x)
        
        # Global pooling
        features = torch.mean(x, dim=1)  # [batch, d_model]
        
        return features
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)
    
class BiGRUClassifier(nn.Module):
    def __init__(self, input_channels=3, hidden_size=128, num_layers=2, num_classes=5):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(0.5)
        # *2 because of bidirectional
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
    
    def feature_extractor(self, x):
        """Extract features before classification head"""
        # Input: [batch, channels=3, time=300]
        x = x.permute(0, 2, 1)  # Convert to [batch, time=300, channels=3]
        # GRU forward pass
        gru_out, _ = self.gru(x)  # Output: [batch, time=300, hidden_size*2]
        # Use final output
        features = gru_out[:, -1, :]  # Take last time step: [batch, hidden_size*2]
        return features
    
    def forward(self, x):
        features = self.feature_extractor(x)
        x = self.dropout(features)
        return self.classifier(x)

class CNNGRUClassifier(nn.Module):
    def __init__(self, input_channels=3, hidden_size=128, num_classes=5):
        super().__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
    
    def feature_extractor(self, x):
        """Extract features before classification head"""
        # Input: [batch, channels=3, time=300]
        # CNN feature extraction
        x = self.cnn(x)  # Output: [batch, 128, time=300]
        # Prepare for GRU
        x = x.permute(0, 2, 1)  # Convert to [batch, time=300, features=128]
        # GRU temporal modeling
        gru_out, _ = self.gru(x)  # Output: [batch, time=300, hidden_size*2]
        # Global max pooling (alternative to using last output)
        features = torch.max(gru_out, dim=1)[0]  # [batch, hidden_size*2]
        return features
    
    def forward(self, x):
        features = self.feature_extractor(x)
        x = self.dropout(features)
        return self.classifier(x)
    
class AttentionGRUClassifier(nn.Module):
    def __init__(self, input_channels=3, hidden_size=128, num_classes=5):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        # Input: [batch, channels=3, time=300]
        x = x.permute(0, 2, 1)  # Convert to [batch, time=300, channels=3]
        
        # GRU forward pass
        gru_out, _ = self.gru(x)  # Output: [batch, time=300, hidden_size*2]
        
        # Attention weights
        attn_weights = self.attention(gru_out)  # [batch, time=300, 1]
        attn_weights = torch.softmax(attn_weights.squeeze(-1), dim=1)  # [batch, time=300]
        
        # Attended representation
        x = torch.sum(gru_out * attn_weights.unsqueeze(-1), dim=1)  # [batch, hidden_size*2]
        
        x = self.dropout(x)
        return self.classifier(x)