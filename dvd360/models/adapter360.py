# from https://github.com/TencentARC/T2I-Adapter/blob/main/ldm/modules/encoders/adapter.py

import torch.nn as nn
from einops import rearrange


def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


def conv_S(in_planes, out_planes, stride=1, padding=1):
    # as is descriped, conv S is 1x3x3
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=1,
        padding=padding,
        bias=True,
    )


def conv_T(in_planes, out_planes, stride=1, padding=1):
    # conv T is 3x1x1
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(3, 1, 1),
        stride=1,
        padding=padding,
        bias=True,
    )


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock_pseudo3d(nn.Module):
    def __init__(
        self, in_c, out_c, down, ksize=3, sk=False, use_conv=True, frame_num=16
    ):
        super().__init__()
        ps = ksize // 2
        self.down = down
        self.frame_num = frame_num

        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            # print('n_in')
            self.in_conv = None

        self.block1 = conv_S(out_c, out_c, stride=1, padding=(0, 1, 1))
        self.act = nn.ReLU()
        self.block2 = conv_T(out_c, out_c, stride=1, padding=(1, 0, 0))

        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:
            x = self.in_conv(x)

        x = rearrange(x, "(b f) c h w -> b c f h w", f=self.frame_num)
        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)

        if self.skep is not None:
            h = h + self.skep(x)
        else:
            h = h + x

        h = rearrange(h, "b c f h w -> (b f) c h w ")
        return h


class Adapter360(nn.Module):
    def __init__(
        self,
        channels=[320, 640, 1280, 1280],
        nums_rb=3,
        cin=64,
        ksize=3,
        sk=False,
        use_conv=True,
        frame_num=16,
    ):
        super(Adapter360, self).__init__()
        self.unshuffle = nn.PixelUnshuffle(8)
        self.channels = channels
        self.nums_rb = nums_rb
        self.body = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i != len(channels) - 1) and (j == 0):
                    self.body.append(
                        ResnetBlock_pseudo3d(
                            channels[i - 1 if i > 0 else 0],
                            channels[i],
                            down=True,
                            ksize=ksize,
                            sk=sk,
                            use_conv=use_conv,
                            frame_num=frame_num,
                        )
                    )
                else:
                    self.body.append(
                        ResnetBlock_pseudo3d(
                            channels[i],
                            channels[i],
                            down=False,
                            ksize=ksize,
                            sk=sk,
                            use_conv=use_conv,
                            frame_num=frame_num,
                        )
                    )
        self.body = nn.ModuleList(self.body)
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)

    def forward(self, x):
        # unshuffle
        x = self.unshuffle(x)
        # extract features
        features = []
        x = self.conv_in(x)

        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)
            features.append(x)

        return features
