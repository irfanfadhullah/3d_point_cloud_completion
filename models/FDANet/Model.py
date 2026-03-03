import torch
import torch.nn as nn
from models.FDANet import FeatureExtract,PointGenerator,Encoder
from models.FDANet.Model_utils import Mlp


class mymodel(nn.Module):
    '''
    Input: x [B,N,3]    s[B,37,3]
    Output: Feature_Extract:[]
    '''

    def __init__(self, point_resolution, k, FE_inchans,p_num,Sec_chans=[3, 128, 192]):
        super().__init__()
        self.feature_extract = FeatureExtract.FeatureExtract(point_resolution, k, FE_inchans)
        self.encoder=Encoder.TransformerEncoder(NumEncoder=6,in_chans=192,pos_chans=192,N_Head=6,N_Points=128)
        self.point_generator = PointGenerator.PointGenerator(p_num)
        self.mlp = Mlp(Sec_chans[0], layer_dims=[Sec_chans[1], Sec_chans[2]])
    def forward(self, x,S):
        global_code,global_feature = self.feature_extract(x)

        s_Fea=None
        if S!=None:
            s_Fea = self.mlp(S).transpose(1, 2).max(dim=-1, keepdim=True)[0]
        #print('s_Fea', s_Fea.shape)
        point_code=self.encoder(global_code,s_Fea)
        CPoint = self.point_generator(point_code,global_feature)
        return CPoint


if __name__ == '__main__':
    tubepoint = torch.randn(1, 3, 2048).cuda()
    secpoint = torch.randn(1, 36, 3).cuda()
    _, _ , N= tubepoint.size()
    N_x = [N, N // 2, N // 4]
    model = mymodel(point_resolution=N_x, k=8,FE_inchans=[32,64,128,256],p_num=1024).cuda()
    CP = model(tubepoint, secpoint)
    print(CP.shape)
