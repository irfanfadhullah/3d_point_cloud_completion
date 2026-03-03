import torch
import torch.nn as nn
import torch.nn.functional as F

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if num_points < k:
        k = num_points
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x_center = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    # 计算邻居点与中心点的特征差异
    diff = feature - x_center

    attention_net = nn.Sequential(
        nn.Linear(num_dims, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 128)
    ).to(device)

    attention_scores = attention_net(diff)  # (batch_size, num_points, k, 1)
    attention_weights = F.softmax(attention_scores, dim=2)  # (batch_size, num_points, k, 1)
    weighted_feature = feature * attention_weights  # (batch_size, num_points, k, num_dims)
    feature = torch.cat((weighted_feature - x_center, x_center), dim=3).permute(0, 3, 1, 2).contiguous() #(B, C+C, N, K)

    return feature

class HDC_ConvModule_2D(nn.Module):
    def __init__(self):
        super(HDC_ConvModule_2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class EdgeConv(nn.Module):
    def __init__(self, k, dim):
        super(EdgeConv, self).__init__()
        self.k = k
        self.dim = dim
        self.bn1 = nn.BatchNorm2d(128)
        self.conv1 = nn.Sequential(nn.Conv2d(dim+dim, 128, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.HDC_ConvModule_2D = HDC_ConvModule_2D()

    def forward(self, x):
        # batch_size = x.size(0)
        x0 = get_graph_feature(x, k=self.k)
        x1 = self.HDC_ConvModule_2D(x0)
        x2 = x1.max(dim=-1, keepdim=False)[0]
        return x2

class EDformer(nn.Module):
    def __init__(self, dims, num_head=8):
        super(EDformer, self).__init__()
        self.dims = dims
        self.num_head = num_head
        self.indices = dims // num_head

        self.conv_q = EdgeConv(k=16, dim=self.dims)
        self.conv_k = EdgeConv(k=16, dim=self.dims)
        self.conv_v = EdgeConv(k=16, dim=self.dims)

    def forward(self, inputs):

        q = self.conv_q(inputs)
        k = self.conv_k(inputs)
        v = self.conv_v(inputs)

        outputs = []
        for i in range(self.num_head):
            query = q[:, :, i * self.indices:(i + 1) * self.indices]
            key = k[:, :, i * self.indices:(i + 1) * self.indices]
            value = v[:, :, i * self.indices:(i + 1) * self.indices]

            matmul_qk = torch.matmul(query, key.transpose(-2, -1))

            attention_weights = F.softmax(matmul_qk, dim=-1)

            output = torch.matmul(attention_weights, value)
            if i == 0:
                net = output
            else:
                net = torch.cat([net, output], dim=-1)

        return net


class MSMHA(nn.Module):
    def __init__(self, input_dim, num_heads=8, dropout=0.1, scales=[1]):
        super(MSMHA, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert self.head_dim * num_heads == input_dim
        self.scales = scales

        self.scale = self.head_dim ** -0.5

        # 定义权重矩阵
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        # dropout层
        self.dropout = nn.Dropout(dropout)

    def get_scaled_dot_product_attention(self, q, k, v, scale):
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        return torch.matmul(attention_weights, v)

    def forward(self, x):
        batch_size, seq_length, _ = x.shape

        # 动态生成位置编码
        position_encoding = torch.arange(seq_length, dtype=torch.float32, device=x.device).unsqueeze(1)
        position_encoding = position_encoding / (
                    10000 ** (torch.arange(0, x.shape[-1], 2, device=x.device).float() / x.shape[-1]))
        position_encoding = torch.cat([torch.sin(position_encoding), torch.cos(position_encoding)], dim=-1)
        position_encoding = position_encoding.unsqueeze(0).expand(batch_size, -1, -1)

        # 添加位置编码
        x = x + position_encoding

        # 计算 query、key 和 value
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # 拆分为多头
        query_head = query.view(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attention_outputs = []
        for scale in self.scales:
            # 根据尺度动态调整key和value的序列长度
            scaled_key = key[:, ::scale]
            scaled_value = value[:, ::scale]

            key_head = scaled_key.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            value_head = scaled_value.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            scaled_attention = self.get_scaled_dot_product_attention(query_head, key_head, value_head, self.scale)
            attention_outputs.append(scaled_attention)

        if len(self.scales) > 1:
            attention_output = torch.mean(torch.stack(attention_outputs, dim=-2), dim=-2)
        else:
            attention_output = attention_outputs[0]

        attention_output = attention_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, -1)

        return attention_output



