import cv2
import torch
from torch.nn import Module, Conv2d, Parameter, Softmax
import torch.nn as nn

class POA(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(POA, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CHA(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CHA, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class COA(nn.Module):
    def __init__(self, channel):
        super(COA, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
        )
    def forward(self, x):
        y = self.conv(x)
        att = harris(x)
        att_map = att.cuda()
        out = att_map * y + x
        return out

def harris(img):
    img = img.float()
    att_0 = torch.tensor([])
    for a in range(img.shape[0]):
        att_1 = torch.tensor([])
        for b in range(img.shape[1]):
            img_c = img[a, b, :, :]
            img_c = img_c.cuda().data.cpu().detach().numpy()
            dst = cv2.cornerHarris(img_c, 3, 3, 0.04)
            dst = cv2.dilate(dst, None)
            img_c[dst > 0.7 * dst.max()] = [1]
            img_c[dst < 0.7 * dst.max()] = [0]
            img_c = torch.from_numpy(img_c)
            img_c = torch.unsqueeze(img_c, dim=0)
            att_1 = torch.cat((att_1, img_c), 0)
        att_2 = torch.unsqueeze(att_1, dim=0)
        att_0 = torch.cat((att_0, att_2), 0)

    return att_0