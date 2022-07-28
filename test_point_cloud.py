from re import L
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from dataloader.mvs_dataset import MVSTestSet
from networks.ucsnet import UCSNet
from utils.utils import dict2cuda, dict2numpy, mkdir_p, save_cameras, write_pfm
from Depth_Fusion.warp_func import *

import argparse, time, cv2
import numpy as np
from collections import *

cudnn.benchmark = True



def filter_depth(ref_depth, src_depths, ref_proj, src_projs):
	'''
	
	:param ref_depth: (1, 1, H, W)
	:param src_depths: (B, 1, H, W)
	:param ref_proj: (1, 4, 4)
	:param src_proj: (B, 4, 4)
	:return: ref_pc: (1, 3, H, W), aligned_pcs: (B, 3, H, W), dist: (B, 1, H, W)
	'''

	ref_pc = generate_points_from_depth(ref_depth, ref_proj)
	src_pcs = generate_points_from_depth(src_depths, src_projs)

	aligned_pcs = homo_warping(src_pcs, src_projs, ref_proj, ref_depth)

	# _, axs = plt.subplots(3, 4)
	# for i in range(3):
	# 	axs[i, 0].imshow(src_pcs[0, i], vmin=0, vmax=1)
	# 	axs[i, 1].imshow(aligned_pcs[0, i],  vmin=0, vmax=1)
	# 	axs[i, 2].imshow(ref_pc[0, i],  vmin=0, vmax=1)
	# 	axs[i, 3].imshow(ref_pc[0, i] - aligned_pcs[0, i], vmin=-0.5, vmax=0.5, cmap='coolwarm')
	# plt.show()

	x_2 = (ref_pc[:, 0] - aligned_pcs[:, 0])**2
	y_2 = (ref_pc[:, 1] - aligned_pcs[:, 1])**2
	z_2 = (ref_pc[:, 2] - aligned_pcs[:, 2])**2
	dist = torch.sqrt(x_2 + y_2 + z_2).unsqueeze(1)

	return ref_pc, aligned_pcs, dist
	
def extract_points(pc, mask, rgb, prob, feature):
	pc = pc.cpu().numpy()
	mask = mask.cpu().numpy()

	mask = np.reshape(mask, (-1,))
	pc = np.reshape(pc, (-1, 3))
	rgb = np.reshape(rgb, (-1, 3))
	points_prob = np.reshape(prob, (-1,))[np.where(mask)]
	points_feature = np.reshape(feature, (-1, feature.shape[-1]))[np.where(mask)]

	points = pc[np.where(mask)]
	colors = rgb[np.where(mask)]

	points_with_color = np.concatenate([points, colors], axis=1)

	return points_with_color, points_prob, points_feature

def process_data(args, scene_results, thresh):
    depths = scene_results['depth']
    cams = scene_results['cam']
    rgbs = scene_results['rgb']
    probs = scene_results['confidence']
    features = scene_results['feature']

    depths = depths[probs>thresh].float()
    projs = torch.cat([torch.eye(4).unsqueeze(0) for _ in range(cams.shape[0])], 0)
    projs[:, :3, :4] = cams[:, 1, :3, :3] @ cams[:, 0, :3, :4]
    projs = projs.float()

    if args.device == 'cuda' and torch.cuda.is_available():
        depths = depths.cuda()
        projs = projs.cuda()

    return depths, projs, rgbs, probs, features

def test_main(args):
    # dataset, dataloader
    testset = MVSTestSet(root_dir=args.root_path, data_list=args.test_list,
                         max_h=args.max_h, max_w=args.max_w, num_views=args.num_views)
    test_loader = DataLoader(testset, 1, shuffle=False, num_workers=4, drop_last=False)

    # build model
    model = UCSNet(stage_configs=list(map(int, args.net_configs.split(","))),
                   lamb=args.lamb)

    # load checkpoint file specified by args.loadckpt
    print("Loading model {} ...".format(args.ckpt))
    state_dict = torch.load(args.ckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    print('Success!')

    model = nn.DataParallel(model)
    model.cuda()
    model.eval()

    tim_cnt = 0

    results = {}
    # results[scene_name]['cam', 'feature', 'rgb', 'confidence', 'depth'] = list of tensor (len = frame)
    last_name = ''

    for batch_idx, sample in enumerate(test_loader):
        scene_name = sample["scene_name"][0]
        frame_idx = sample["frame_idx"][0][0]

        if frame_idx == 0:
            if last_name != '':
                for key in scene_results.keys():
                    scene_results[key] = torch.cat(scene_results[key], 0)
                results[last_name] = scene_results

            last_name = scene_name
            scene_results = {}
            scene_results['rgb'] = []
            scene_results['cam'] = []
            scene_results['depth'] = []
            scene_results['confidence'] = []
            scene_results['feature'] = []

        print('Process data ...')
        sample_cuda = dict2cuda(sample)

        print('Testing {} frame {} ...'.format(scene_name, frame_idx))
        start_time = time.time()
        outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
        end_time = time.time()

        # outputs = dict2numpy(outputs)
        del sample_cuda

        tim_cnt += (end_time - start_time)

        print('Finished {}/{}, time: {:.2f}s ({:.2f}s/frame).'.format(batch_idx+1, len(test_loader), end_time-start_time,
                                                               tim_cnt / (batch_idx + 1.)))

        scene_results['rgb'].append(torch.clamp(sample['imgs'][0, 0]*255, 0, 255).permute(1, 2, 0).unsqueeze(0)) # not multiplied by 255 yet
        scene_results['cam'].append(sample['proj_matrices']['stage3'][0, 0].unsqueeze(0))

        # ref_img = sample["imgs"][0, 0].numpy().transpose(1, 2, 0) * 255
        # ref_img = np.clip(ref_img, 0, 255).astype(np.uint8)
        # Image.fromarray(ref_img).save(rgb_path+'/{:08d}.png'.format(frame_idx))

        # cam = sample["proj_matrices"]["stage3"][0, 0].numpy()
        # save_cameras(cam, cam_path+'/cam_{:08d}.txt'.format(frame_idx))

        res = outputs['stage3']
        h, w = res['depth'][0].shape
        scene_results['depth'].append(res['depth'][0].unsqueeze(0))
        confidence = torch.tensor(cv2.resize(outputs['stage1']['confidence'][0].numpy(), (w, h), interpolation=cv2.INTER_LINEAR))
        scene_results['confidence'].append(confidence.unsqueeze(0))
        scene_results['feature'].append(res['feature'][0].permute((1, 2, 0)).unsqueeze(0))

        print('Saved results for {}/{} (resolution: {})'.format(scene_name, frame_idx, res['depth'][0].shape))

    results[last_name] = scene_results

    torch.cuda.empty_cache()

    return results


def point_cloud_main(args, results):
    # results[scene_name]['cam', 'feature', 'rgb', 'confidence', 'depth'] = list of tensor (len = frame)

    for scene in results.keys():
        scene_results = results[scene]
        depths, projs, rgbs, probs, features = process_data(args, scene_results, args.prob_thresh)
        tot_frame = depths.shape[0]
        height, width = depths.shape[2], depths.shape[3]
        points = []
        points_prob = []
        points_feature = []

        print('Scene: {} total: {} frames'.format(scene, tot_frame))
        for i in range(tot_frame):
            pc_buff = torch.zeros((3, height, width), device=depths.device, dtype=depths.dtype)
            val_cnt = torch.zeros((1, height, width), device=depths.device, dtype=depths.dtype)
            j = 0
            batch_size = 20

            while True:
                ref_pc, pcs, dist = filter_depth(ref_depth=depths[i:i+1], src_depths=depths[j:min(j+batch_size, tot_frame)],
                                                    ref_proj=projs[i:i+1], src_projs=projs[j:min(j+batch_size, tot_frame)])
                masks = (dist < args.dist_thresh).float()
                masked_pc = pcs * masks
                pc_buff += masked_pc.sum(dim=0, keepdim=False)
                val_cnt += masks.sum(dim=0, keepdim=False)

                j += batch_size
                if j >= tot_frame:
                    break

            final_mask = (val_cnt >= args.num_consist).squeeze(0)
            avg_points = torch.div(pc_buff, val_cnt).permute(1, 2, 0)

            final_pc, pc_prob, pc_feature = extract_points(avg_points, final_mask, rgbs[i], probs[i], features[i])
            points.append(final_pc)
            points_prob.append(pc_prob)
            points_feature.append(pc_feature)
            print('Processing {} {}/{} ...'.format(scene, i+1, tot_frame))
            # np.save('{}/{}/{:08d}.npy'.format(args.save_path, scene, i+1), final_pc)

        point_cloud = {}
        point_cloud['points_with_color'] = np.concatenate(points, axis=0)
        point_cloud['confidence'] = np.concatenate(points_prob)
        point_cloud['feature'] = np.concatenate(points_feature)

        del points, depths, rgbs, projs

        print('Save {} successful.'.format(scene))

        return point_cloud

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test UCSNet.')

    parser.add_argument('--root_path', type=str, help='path to root directory.')
    parser.add_argument('--test_list', type=str, help='testing scene list.')
    parser.add_argument('--save_path', type=str, help='path to save depth maps.')

    #test parameters
    parser.add_argument('--max_h', type=int, help='image height', default=1080)
    parser.add_argument('--max_w', type=int, help='image width', default=1920)
    parser.add_argument('--num_views', type=int, help='num of candidate views', default=3)
    parser.add_argument('--lamb', type=float, help='the interval coefficient.', default=1.5)
    parser.add_argument('--net_configs', type=str, help='number of samples for each stage.', default='64,32,8')
    parser.add_argument('--ckpt', type=str, help='the path for pre-trained model.', default='./checkpoints/model.ckpt')

    parser.add_argument('--dist_thresh', type=float, default=0.001)
    parser.add_argument('--prob_thresh', type=float, default=0.6)
    parser.add_argument('--num_consist', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args([])

    with torch.no_grad():
        results = test_main(args)
        point_cloud = point_cloud_main(args, results)
