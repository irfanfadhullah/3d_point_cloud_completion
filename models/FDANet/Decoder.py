import torch
from models.FDANet.Encoder import TransformerEncoder
import torch.nn as nn
from models.FDANet.Encoder import PositionEncoder, PoswiseFeedForwardLayer


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, in_chans, N_Head):
        super().__init__()
        self.N_Head = N_Head
        self.chans_Head = in_chans // N_Head
        self.W_QKV = nn.Linear(in_chans, in_chans)
        self.linear = nn.Linear(in_chans, in_chans)
        self.layer_norm = nn.LayerNorm(in_chans)

    def forward(self, DecoderInput, EncOutput):
        Res = DecoderInput
        B, N, C = DecoderInput.size()
        Q = self.W_QKV(DecoderInput).view(B, -1, self.N_Head, C // self.N_Head).transpose(1, 2)
        K = self.W_QKV(EncOutput).view(B, -1, self.N_Head, C // self.N_Head).transpose(1, 2)
        V = self.W_QKV(EncOutput).view(B, -1, self.N_Head, C // self.N_Head).transpose(1, 2)
        attn = (Q @ K.transpose(-2, -1)) / self.chans_Head ** 0.5
        attn = attn.softmax(dim=-1)
        DecFea_Attention = torch.matmul(attn, V).reshape(B, N, C)
        DecFea_Attention = self.linear(DecFea_Attention)
        DecFea_Attention = DecFea_Attention #+ Res#self.layer_norm()
        return DecFea_Attention


class Decoder(nn.Module):
    def __init__(self, in_chans, N_Head):
        super().__init__()
        self.attn = MultiHeadCrossAttention(in_chans, N_Head)
        self.ffl = PoswiseFeedForwardLayer(in_chans)

    def forward(self, DecoderInput, EncOutput):
        DecFea_attn = self.attn(DecoderInput, EncOutput)
        Dec_ffl = self.ffl(DecFea_attn)
        Dec_out = Dec_ffl
        return Dec_out


class TransformerDecoder(nn.Module):
    '''
    Input: [B,N,C]
    Output: [B,N,C]
    '''

    def __init__(self, NumEncoder, in_chans, pos_chans, N_Head):
        super().__init__()
        self.DecoderInput = nn.Sequential(
            nn.Conv1d(in_chans, in_chans * 2, 1),
            nn.BatchNorm1d(in_chans * 2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(in_chans * 2, in_chans, 1)
        )
        self.PositionEncoder = PositionEncoder(in_chans, pos_chans)
        self.Decoder = nn.ModuleList([Decoder(in_chans=in_chans, N_Head=N_Head) for _ in range(NumEncoder)])

    def forward(self, FeaExtra, EncOutput):
        position = self.PositionEncoder(FeaExtra)
        FeaExtra = self.DecoderInput(FeaExtra)
        DecInput = FeaExtra + position
        DecInput = DecInput.transpose(1, 2).contiguous()  # [B,N,C]
        for decoderlayer in self.Decoder:
            DecFea = decoderlayer(DecInput, EncOutput)
        return DecFea  # [B,N,C]


if __name__ == '__main__':
    FeaExtra = torch.randn(4, 512, 128).cuda()
    EncOutput = torch.randn(4, 128, 512).cuda()
    model = TransformerDecoder(NumEncoder=6, in_chans=512, pos_chans=512, N_Head=8).to('cuda')
    Fea = model(FeaExtra, EncOutput)
    print(Fea.shape)
