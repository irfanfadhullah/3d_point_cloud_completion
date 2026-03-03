import torch
import torch.nn as nn
from models.FDANet.Model_utils import Mlp,Mlp_Res,Mlp_Point

class PointGenerator(nn.Module):
    '''
    Input: FatureExtract[B,C,N]  DecOutput[B,N,C]
    Output: Point[B,3,N]
    '''

    def __init__(self,num_query):
        super().__init__()

        self.coarse_pred = nn.Sequential(
            nn.Linear(1920, 768),
            nn.ReLU(),
            nn.Linear(768, 3*num_query)
        )
        self.displace = nn.Sequential(
            nn.Linear(1920,  64 * num_query // 4),
            nn.ReLU(),
        )

        self.code_pred = nn.Sequential(
            nn.Linear(1920, 1920),
            nn.ReLU(),
            #nn.Linear(768, 3 * num_query)
        )
        self.num_query=num_query
        self.conv=nn.Conv1d(64,3,1)
    def forward(self, point_code,global_feature):
        #print('global_feature', global_feature.shape)

        bs=global_feature.shape[0]
        Point = self.coarse_pred(global_feature)

        Code = self.code_pred(point_code).reshape(-1, 1920)
        #print('Code', Code.shape)
        global_feature=global_feature+Code
        Dis = self.displace(global_feature).reshape(bs, 64, self.num_query//4)
        #print('Dis',Dis.shape)
        Point=Point.reshape(bs, -1,4, 3)
        #print('Point',Point.shape)
        Dis=self.conv(Dis).reshape(bs, -1,1, 3)
        #print('Dis',Dis.shape)
        Point=(Point+Dis).reshape(bs, -1, 3)

        return Point


if __name__ == '__main__':
    FeaExtra = torch.randn(4,1920).cuda()
    model = PointGenerator(1024).to('cuda')
    Point = model(FeaExtra,FeaExtra)
    print(Point.shape)