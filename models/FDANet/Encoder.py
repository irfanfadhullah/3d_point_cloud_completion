import numpy as np
import torch
import torch.nn as nn


class PoswiseFeedForwardLayer(nn.Module):
    def __init__(self,in_chans):
        super().__init__()
        self.fc1 = nn.Linear(in_chans, in_chans)
        self.fc2 = nn.Linear(in_chans,in_chans)
        self.norm = nn.LayerNorm(in_chans)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
    def forward(self, x):
        #residual = x # inputs : [batch_size, len_q, d_model]
        output = self.fc1(x)
        #output = self.norm(output)
        output = self.gelu(output)
        output = self.drop(output)
        output = self.fc2(output)
        #output = self.relu(output)
        output = self.drop(output)
        return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,in_chans,N_Head):
        super().__init__()
        self.N_Head=N_Head
        self.chans_Head=in_chans//N_Head
        self.W_QKV = nn.Linear(in_chans, in_chans*3)
        self.linear = nn.Linear(in_chans, in_chans)
        self.drop_attn=nn.Dropout(0.5)
        self.drop_linear = nn.Dropout(0.3)
    def forward(self,EncFea):
        B,N,C=EncFea.shape
        #print(EncFea.shape)
        QKV = self.W_QKV(EncFea).view(B, -1,3, self.N_Head, C // self.N_Head).permute(2,0,3,1,4)
        Q = QKV[0]
        K = QKV[1]
        V = QKV[2]
        attn=(Q@K.transpose(-2,-1))/self.chans_Head**0.5
        attn = attn.softmax(dim=-1)
        attn = self.drop_attn(attn)
        EncFea_Attention = torch.matmul(attn, V).transpose(1,2).reshape(B, N, -1)
        #print('EncFea_Attention',EncFea_Attention.shape)
        EncFea_Attention =self.linear(EncFea_Attention)
        EncFea_Attention = self.drop_linear(EncFea_Attention)
        return EncFea_Attention

class GlobalAttention(nn.Module):
    '''
    Input:EncFea [B,N,C]
          SectionFea [B,1,C]
    Output: EncFea_glob_Attention [B,N,C]
    '''
    def __init__(self,in_chans,N_Head,N_Points):
        super().__init__()
        self.N_Head = N_Head
        self.chans_Head = in_chans // N_Head
        self.W_Q = nn.Linear(in_chans, in_chans)
        self.W_K = nn.Conv1d(in_chans, in_chans,N_Points)
        self.W_V = nn.Linear(in_chans, in_chans)
        self.linear = nn.Linear(in_chans,192)
        self.relu=nn.ReLU()
        self.drop_attn=nn.Dropout(0.5)
        self.drop_linear = nn.Dropout(0.5)
    def forward(self,EncFea,SectionFea):
        B, N, C = EncFea.size()
        #print('EncFea',EncFea.shape)
        Q = self.W_Q(EncFea).view(B, -1, self.N_Head, C // self.N_Head).transpose(1,2)
        #print('Q', Q.shape)
        K = self.W_K(EncFea.transpose(1,2).contiguous()).transpose(1,2).contiguous().view(B, -1, self.N_Head, C // self.N_Head).transpose(1,2)
        #print('K', K.shape)
        #print('SectionFea',SectionFea.shape)
        V = self.W_V(SectionFea).view(B, -1, self.N_Head, C // self.N_Head).transpose(1,2)
        #print('V', V.shape)
        glob_attn=(Q@K.transpose(-2,-1))/self.chans_Head**-0.5
        glob_attn = glob_attn.softmax(dim=-1)
        glob_attn=self.drop_attn(glob_attn)
        #print('glob_attn',glob_attn.shape)
        EncFea_glob_Attention = torch.matmul(glob_attn, V).reshape(B, N, C)
        #print('EncFea_glob_Attention', EncFea_glob_Attention.shape)
        EncFea_glob_Attention = self.linear(EncFea_glob_Attention)
        EncFea_glob_Attention = self.drop_linear(EncFea_glob_Attention)
        return EncFea_glob_Attention

class PositionEncoder(nn.Module):
    def __init__(self,in_chans,out_chans):
        super().__init__()
        self.in_chans=in_chans
        self.out_chans=out_chans
        self.pos_embed = nn.Sequential(
            nn.Conv1d(in_chans, out_chans, 1),
            nn.GroupNorm(4,out_chans),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(out_chans, out_chans, 1)
        )
    def forward(self,x):
        #print('x',x.shape)
        position=self.pos_embed(x)
        return position

class Encoder(nn.Module):
    def __init__(self,in_chans,N_Head,N_Points):
        super().__init__()
        self.norm=nn.LayerNorm(in_chans)
        self.attn=MultiHeadSelfAttention(in_chans,N_Head)
        self.glob_attn=GlobalAttention(in_chans,N_Head,N_Points)
        self.layer_norm = nn.LayerNorm(in_chans)
        self.linear = nn.Linear(in_chans, in_chans)
        self.ffl=PoswiseFeedForwardLayer(in_chans)
        self.relu=nn.ReLU()
        self.drop=nn.Dropout(0.5)
        self.N_points=N_Points
    def forward(self,EncFea_in,SectionFea_in=None):
        Res=EncFea_in
        #print(EncFea_in.shape)
        #print(SectionFea_in.shape)
        EncFea=self.norm(EncFea_in)
        #SectionFea=self.norm2(SectionFea_in)
        EncFea_attn = self.attn(EncFea)
        if SectionFea_in!=None:
            SectionFea = SectionFea_in
            SectionFea_attn = self.glob_attn(EncFea, SectionFea)
            EncFea_sec=self.linear(EncFea_attn+SectionFea_attn)
            EncFea_sec = self.norm(EncFea_sec)
            EncFea_sec = self.relu(EncFea_sec)
        else:
            EncFea_sec=EncFea_attn
            #print('EncFea_sec', EncFea_se+c.shape)

        EncFea_out =Res+ self.drop (EncFea_sec)
        EncFea_out=EncFea_out+self.drop(self.ffl(self.layer_norm(EncFea_out)))
        return EncFea_out

class TransformerEncoder(nn.Module):
    '''
    Input: [B,C,N]
    Output: [B,N,C]
    '''
    def __init__(self,NumEncoder,in_chans,pos_chans,N_Head,N_Points):
        super().__init__()
        self.in_chans=in_chans
        self.pos_chans = pos_chans
        self.N_Head = N_Head
        self.N_Points = N_Points
        self.positionencoder=PositionEncoder(in_chans=self.in_chans,out_chans=self.pos_chans)
        self.NumEncoder=NumEncoder
        self.encoder=nn.ModuleList([Encoder(in_chans=self.pos_chans,N_Head=self.N_Head,N_Points=self.N_Points) for _ in range(NumEncoder)])
        self.increase_dim=nn.Sequential(
            nn.Linear(in_chans, 1920),
            nn.ReLU(),
        )
    def forward(self,x,S):
        B=x.shape[0]
        #print('x',x.shape)
        #position=self.positionencoder(x).transpose(1,2).contiguous()
        #print('position',position.shape)
        EncFea=x.transpose(1,2).contiguous()# [B,N,C].
        #print('EncFea',EncFea.shape)
        SectionFea =S
        if S != None:
            SectionFea=S.transpose(1,2).contiguous()
        #print('SectionFea', SectionFea.shape)
        for encoderlayer in self.encoder:
            EncFea=encoderlayer(EncFea,SectionFea)#+position
        glb_Encfea=self.increase_dim(EncFea).transpose(1,2)
        #print(glb_Encfea.shape)
        glb_Encfea=glb_Encfea.max(-1,keepdim=False)[0]
        #print(glb_Encfea.shape)
        return glb_Encfea# [B,N,C]



if __name__ == '__main__':
    a = torch.randn(4, 192,128).cuda()
    b = torch.randn(4, 512,1).cuda()
    model = TransformerEncoder(NumEncoder=6,in_chans=192,pos_chans=192,N_Head=6,N_Points=128).to('cuda')
    Fea = model(a,None)
    print(Fea.shape)

