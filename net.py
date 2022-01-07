import common
import torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter, Softmax
import fusion_strategy
import torch.nn.functional as F
from att import POA, CHA, COA


def make_model(args, parent=False):
    return Student_sr(args)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Student Network SR part
class Student_sr(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(Student_sr, self).__init__()

        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale
        act = nn.ReLU(True)

        modules_head = [conv(16, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        modules_result = [conv(16, 1, kernel_size)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.uperfe = common.Upsampler(conv, scale, n_feats, act=False)
        self.tail = conv(n_feats, 16, kernel_size)
        self.result = nn.Sequential(*modules_result)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        sr_uper_fe = self.uperfe(res)
        sr_feature = self.tail(sr_uper_fe)
        x = self.result(sr_feature)

        return x, sr_uper_fe, sr_feature

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            # out = F.normalize(out)
            out = F.relu(out, inplace=True)
            # out = self.dropout(out)
        return out


# Dense convolution unit
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


# Dense Block unit
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels+out_channels_def*2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


# teacher network
class Teacher_net(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(Teacher_net, self).__init__()
        denseblock = DenseBlock
        nb_filter = [16, 64, 32, 16]
        kernel_size = 3
        stride = 1

        # encoder
        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.DB1 = denseblock(nb_filter[0], kernel_size, stride)

        # decoder
        self.conv2 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)
        self.conv3 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv4 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.conv5 = ConvLayer(nb_filter[3], output_nc, kernel_size, stride)

    def encoder(self, input):
        x1 = self.conv1(input)
        x_DB = self.DB1(x1)
        return [x_DB],x1

    def fusion(self, en1, en2):
        fusion_function = fusion_strategy.attention_fusion_weight
        f_0 = fusion_function(en1, en2)
        return [f_0]

    def decoder(self, f_en, fe_conv2):
        x2 = self.conv2(f_en[0])
        x3 = self.conv3(x2)
        output_feature = self.conv4(x3)
        output_fe = fe_conv2 + output_feature

        return [output_fe], x2

    def result(self, out_fe):
        output = self.conv5(out_fe[0])
        return [output]

## Double Frequency Attention (FDB)
class DFA(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(DFA, self).__init__()

        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(2)]
        self.body = nn.Sequential(*modules_body)

        self.conv1 = nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=False)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        x = self.body(x)
        query = self.conv1(x)
        key = self.conv2(x)
        energy = query.mul(key)
        attention = self.softmax(energy)

        out = self.conv3(key)
        out = out.mul(attention)
        out = self.gamma * out
        return out

## DFA_Group (DFAG)
class DFAG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(DFAG, self).__init__()
        modules_body = [
            DFA(conv, n_feat, kernel_size, reduction) for _ in range(6)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Corner Embedding Attention (CEA)
class CEA(nn.Module):
    def __init__(self, n_feat):
        super(CEA, self).__init__()
        self.poa = POA(n_feat)
        self.cha = CHA(n_feat)
        self.coa = COA(n_feat)

        self.convp1 = nn.Sequential(nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(n_feat),
                                    nn.ReLU())
        self.convp2 = nn.Sequential(nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(n_feat),
                                    nn.ReLU())
        self.convp3 = nn.Sequential(nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(n_feat),
                                    nn.ReLU())
        self.convp4 = nn.Sequential(nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(n_feat),
                                    nn.ReLU())
        self.convp5 = nn.Sequential(nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(n_feat),
                                    nn.ReLU())
        self.convp6 = nn.Sequential(nn.Conv2d(n_feat, n_feat, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(n_feat),
                                    nn.ReLU())

        self.conv5 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(n_feat, n_feat, 1))
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(n_feat, n_feat, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(n_feat, n_feat, 1))
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(n_feat, n_feat, 1))

    def forward(self, x):
        feat1 = self.convp1(x)
        poa_feat = self.poa(feat1)
        poa_conv = self.convp2(poa_feat)
        poa_output = self.conv5(poa_conv)

        feat2 = self.convp3(x)
        cha_feat = self.cha(feat2)
        cha_conv = self.convp4(cha_feat)
        cha_output = self.conv6(cha_conv)

        feat3 = self.convp5(x)
        coa_feat = self.coa(feat3)
        coa_conv = self.convp6(coa_feat)
        coa_output = self.conv7(coa_conv)

        out = poa_output + cha_output + coa_output
        out = self.conv8(out)

        return out

# Student fusion part
class Student_fu(nn.Module):
    def __init__(self, args, input_nc=1, output_nc=1, conv=common.default_conv):
        super(Student_fu, self).__init__()
        nb_filter = [16, 64, 32, 16]
        stride = 1
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        act = nn.ReLU(True)

        # encoder
        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.conv2 = ConvLayer(nb_filter[0], nb_filter[1], kernel_size, stride)

        modules_body_i = [ResidualGroup(conv, n_feats, kernel_size, reduction, n_resblocks=9)]
        modules_body_i.append(DFAG(conv, n_feats, kernel_size, reduction))
        modules_body_i.append(conv(n_feats, n_feats, kernel_size))
        self.body_i = nn.Sequential(*modules_body_i)

        self.conv = conv(n_feats, n_feats, kernel_size)

        modules_body_v = [
            ResidualGroup(conv, n_feats, kernel_size, reduction, n_resblocks=9)]
        modules_body_v.append(CEA(n_feats))
        modules_body_v.append(conv(n_feats, n_feats, kernel_size))
        self.body_v = nn.Sequential(*modules_body_v)

        # decoder
        self.conv5 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)
        self.conv6 = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)
        self.conv7 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv8 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        # self.conv9 = ConvLayer(nb_filter[3], output_nc, kernel_size, stride)

    def encoder_i(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        res = self.body_i(x)
        res += x
        return res

    def encoder_v(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        res = self.body_v(x)
        out = x + res
        return out

    def fusion(self, en1, en2):
        fusion_function = fusion_strategy.attention_fusion_weight
        f_0 = fusion_function(en1, en2)
        return [f_0]

    def decoder(self, f_en):
        x5 = self.conv5(f_en[0])
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        output_feature = self.conv8(x7)
        return [output_feature]

    # def result(self, out_fe):
    #     output = self.conv9(out_fe)
    #     return [output]

