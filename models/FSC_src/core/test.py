import argparse
import numpy as np
import torch
import torch.utils.data as Data
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))

from models.FSCSVD import Model
from datasets import ShapeNet
from utils.loss_utils import *

from models.model_utils import cross_attention, furthest_point_sample
from pointnet2_ops.pointnet2_utils import gather_operation as gather_points
from config_pcn import cfg
import pdb

CATEGORIES_PCN       = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'vessel']
CATEGORIES_PCN_NOVEL = ['bus', 'bed', 'bookshelf', 'bench', 'guitar', 'motorbike', 'skateboard', 'pistol']


def random_sample(pc, n):
    idx = np.random.permutation(pc.shape[1])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pc.shape[1], size=n-pc.shape[1])])
    return pc[:, idx[:n], :]


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def export_ply(filename, points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pc, write_ascii=True)


def test_single_category(category, model, params):

    test_dataset = ShapeNet(params.test_dataset_path, 'test_novel' if params.novel else 'test', category)
    test_dataloader = Data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    index = 1
    total_l1_cd, total_l2_cd, total_f_score = 0.0, 0.0, 0.0
    with torch.no_grad():
        for p, c in test_dataloader:
            p = p.to(params.device)
            c = c.to(params.device)
            # pdb.set_trace()
            if params.test_point_num != 2048:
                partial = gather_points(c.transpose(1, 2).contiguous(), furthest_point_sample(c, params.test_point_num))
                p = random_sample(partial.transpose(1, 2).contiguous(), 2048)
            _, _, c_ = model(p) # SVD decoder architecture
            # _, c_ = model(p)
            cdl1, cdl2, f1 = calc_cd(c_, c, calc_f1=True)
            total_l1_cd += cdl1.mean().item()
            total_l2_cd += cdl2.mean().item()
            for i in range(len(c)):
                total_f_score += f1.mean().item()
    
    avg_l1_cd = total_l1_cd / len(test_dataset)
    avg_l2_cd = total_l2_cd / len(test_dataset)
    avg_f_score = total_f_score / len(test_dataset)

    return avg_l1_cd, avg_l2_cd, avg_f_score


def test(params):

    print(params.exp_name)

    # load pretrained model
    model = Model(cfg)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    model.load_state_dict(torch.load(params.ckpt_path)['model'])
    model.eval()

    print('\033[33m{:20s}{:20s}{:20s}{:20s}\033[0m'.format('Category', 'L1_CD(1e-3)', 'L2_CD(1e-4)', 'FScore-0.01(%)'))
    print('\033[33m{:20s}{:20s}{:20s}{:20s}\033[0m'.format('--------', '-----------', '-----------', '--------------'))

    if params.category == 'all':
        if params.novel:
            categories = CATEGORIES_PCN_NOVEL
        else:
            categories = CATEGORIES_PCN
        
        l1_cds, l2_cds, fscores = list(), list(), list()
        for category in categories:
            avg_l1_cd, avg_l2_cd, avg_f_score = test_single_category(category, model, params)
            print('{:20s}{:<20.4f}{:<20.4f}{:<20.4f}'.format(category.title(), 1e3 * avg_l1_cd, 1e4 * avg_l2_cd, 1e2 * avg_f_score))
            l1_cds.append(avg_l1_cd)
            l2_cds.append(avg_l2_cd)
            fscores.append(avg_f_score)
        
        print('\033[33m{:20s}{:20s}{:20s}{:20s}\033[0m'.format('--------', '-----------', '-----------', '--------------'))
        print('\033[32m{:20s}{:<20.4f}{:<20.4f}{:<20.4f}\033[0m'.format('Average', np.mean(l1_cds) * 1e3, np.mean(l2_cds) * 1e4, np.mean(fscores) * 1e2))
    else:
        avg_l1_cd, avg_l2_cd, avg_f_score = test_single_category(params.category, model, params)
        print('{:20s}{:<20.4f}{:<20.4f}{:<20.4f}'.format(params.category.title(), 1e3 * avg_l1_cd, 1e4 * avg_l2_cd, 1e2 * avg_f_score))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Point Cloud Completion Testing')
    parser.add_argument('--exp_name', type=str, help='Tag of experiment')
    parser.add_argument('--result_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--ckpt_path', type=str, help='The path of pretrained model.')
    parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loader')
    parser.add_argument('--num_workers', type=int, default=8, help='Num workers for data loader')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for testing')
    parser.add_argument('--novel', type=bool, default=False, help='unseen categories for testing')
    parser.add_argument('--test_point_num', type=int, default=2048, help='number of point cloud when testing')
    params = parser.parse_args()

    if not params.emd:
        test(params)
