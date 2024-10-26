from train_utils import *
from transform import *
from LoG import *
from model import *
from dataset import MonteCarloTestset
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_msssim import ssim
import os
import logging
import datetime
import argparse

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="Custom identifier prepending checkpoint, log, plot, etc. names for a given run")
    parser.add_argument("--output-dir", help="Directory containing render/ and log/", required=True)
    parser.add_argument("--testset-dir", help="Directory containing albedo/, depth/, normal/ and rgb/ folders", required=True)
    parser.add_argument("--save-renders", help="Save every frame produced by the model", action="store_true")
    parser.add_argument("--ckpt-path", help="Path specifying model checkpoint to be tested", required=True)
    parser.add_argument("--width", help="Width to crop model input and output to", type=int, default=1920)
    parser.add_argument("--height", help="Height to crop model input and output to", type=int, default=1024)
    parser.add_argument("--kernel-size", help="Set the kernel size to be used in every convolutional layer", type=int, default=3)
    parser.add_argument("--stateless", help="Disable model state retention", action="store_true")
    parser.add_argument("--flow-out", help="Flow warp model output at t-1 and average with model output at t", action="store_true")
    return parser.parse_args()

def test(
        model,
        device,
        checkpoint_path,
        testset_dir,
        render_dir,
        log_dir,
        identifier,
        width,
        height,
        save_renders,
        stateless,
        flow_out
):
    start_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') 

    logging.info('Initializing Test dataset...')
    test_set = MonteCarloTestset(root_dir=testset_dir,
                                 width=width,
                                 height=height,
                                 transform=transforms.Compose([
                                     AlbedoDemodulateBoring(),
                                     TonemapIsik(),
                                     ToTensor()
                                 ]))
    
    logging.info('Initializing Test dataloader...')
    test_loader = DataLoader(test_set, shuffle=False, batch_size=1, num_workers=8, pin_memory=True)
    
    logging.info(f'Loading Autoencoder from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if identifier is not None:
        log_filename = f'{identifier}-test-{start_time}.txt'
    else:
        log_filename = f'test-{start_time}.txt'
    log_filepath = os.path.join(log_dir, 'test', log_filename)
    
    ckpt_info = f'Loaded Autoencoder: Mean average validation loss: {checkpoint["loss"]}, epoch: {checkpoint["epoch"]}'
    logging.info(ckpt_info)
    log_to_file(os.path.join(log_dir, 'test', log_filename), ckpt_info)

    logging.info('Initializing Criteria...')
    mse = nn.MSELoss()

    num_scenes = 0
    scene_frame = 0
    last_scene = None
    output_history = None
    target_history = None

    with torch.inference_mode():
        model.eval()

        # Using PSNR, TRMAE, SSIM, RMSE on tonemapped output

        psnr_all = trmae_all = ssim_all = rmse_all = 0
        psnr_scene = trmae_scene = ssim_scene = rmse_scene = 0

        for i, batch in enumerate(test_loader):
            inputs, targets, albedo, motion, rgb_in, normal_vanilla, depth = (batch[x].to(device) for x in ['inputs', 'targets', 'albedo', 'motion', 'rgb_in', 'normal_vanilla', 'depth'])

            # Initialise 2-frame buffers for output and target images based on batch size
            if output_history is None:
                N, _, C, H, W = targets.shape
                output_history = torch.zeros(N, 2, C, H, W, device=device)
            if target_history is None:
                N, _, C, H, W = targets.shape
                target_history = torch.zeros(N, 2, C, H, W, device=device)
            
            # Check if test scene changed
            scene = batch['scene'][0]
            if last_scene is not None and scene != last_scene: # Dataloader advanced to next scene
                num_scenes += 1
                
                model.clear_state()
                output_history.zero_() # 1 = this output, 0 = last output
                target_history.zero_() 

                psnr_all += psnr_scene_avg
                trmae_all += trmae_scene_avg
                ssim_all += ssim_scene_avg
                rmse_all += rmse_scene_avg
                
                scene_info = f'{last_scene} Metrics | PSNR: {psnr_scene_avg}, TRMAE: {trmae_scene_avg}, SSIM: {ssim_scene_avg}, RMSE: {rmse_scene_avg}'
                logging.info(scene_info)
                log_to_file(log_filepath, scene_info)

                psnr_scene = trmae_scene = ssim_scene = rmse_scene = 0
                scene_frame = 0
            last_scene = scene

            if scene_frame >= 200: 
                # Keep continuing until the scene changes
                continue

            # Forward
            target = targets[:, 0]
            gbuffers, motion_vec, rgb = inputs[:, 0], motion[:, 0], rgb_in[:, 0]
            output, _, _ = model(gbuffers, motion_vec, rgb, normal_vanilla[:, 0], depth[:, 0]) 
            output *= (albedo[:, 0] + 1e-2)

            if flow_out:
                if scene_frame == 0:
                    prev_normal = ((normal_vanilla[:, 0].clone() - 127.0) / 127.0) 
                    prev_normal = prev_normal / torch.sqrt((prev_normal ** 2).sum(dim=1))
                    prev_depth = depth[:, 0].clone()
                    prev_out = output.clone()
                elif scene_frame > 0:
                    fw_prev_out = flow_warp_utils(prev_out, motion[:, 0], 'nearest', device)
                    fw_prev_normal = flow_warp_utils(prev_normal, motion[:, 0], 'nearest', device)
                    fw_prev_depth = flow_warp_utils(prev_depth, motion[:, 0], 'nearest', device)

                    # Disocclusion mask
                    cur_depth = depth[:, 0].clone()
                    mask_depth = (torch.abs(cur_depth - fw_prev_depth) > 0.1)

                    cur_normal = (normal_vanilla[:, 0].clone() - 127.0) / 127.0
                    cur_normal = cur_normal / torch.sqrt((cur_normal ** 2).sum(dim=1))
                    dot_norm = (fw_prev_normal * cur_normal).sum(dim=1)
                    mask_normals = (dot_norm < 0.9).unsqueeze(1)

                    mask_nan = torch.isnan(cur_normal).any(dim=1).unsqueeze(1)

                    mask_disocclusion = (mask_normals | mask_depth | mask_nan)
                    fw_prev_out[mask_disocclusion.expand_as(fw_prev_out)] = 0.0

                    # Panning mask
                    mask = (fw_prev_out == 0.0).all(dim=1)
                    mask_unsqueezed = mask.unsqueeze(1)
                    fw_prev_out[mask_unsqueezed.expand_as(fw_prev_out)] = output[mask_unsqueezed.expand_as(output)]

                    alpha = 0.5
                    taa_out = (alpha * fw_prev_out) + ((1.0 - alpha) * output)

                    prev_depth = cur_depth.clone()
                    prev_normal = cur_normal.clone()
                    output = taa_out.clone()
                    prev_out = output.clone()

            if stateless:
                model.clear_state()

            # Tonemap
            tonemapped_output = (output / (1 + output)) ** (1 / 2.4)
            tonemapped_target = (target / (1 + target)) ** (1 / 2.4)

            norm_output = ((tonemapped_output - tonemapped_output.min()) / (tonemapped_output.max() - tonemapped_output.min()))
            norm_target = ((tonemapped_target - tonemapped_target.min()) / (tonemapped_target.max() - tonemapped_target.min()))

            # Store (non-tonemapped) output and target in 2-frame buffers
            output_history[:, 0] = output_history[:, 1]
            output_history[:, 1] = norm_output
            target_history[:, 0] = target_history[:, 1]
            target_history[:, 1] = norm_target

            # Temporal differencing images. Computed from raw HDR, not tonemapped
            dt_outputs, dt_targets = finite_differencing(output_history, target_history)

            # Compute test metrics
            psnr_scene += psnr(norm_output, norm_target)
            trmae_scene += trmae(dt_outputs, dt_targets)
            ssim_scene += ssim(norm_output, norm_target, data_range=1, size_average=True)
            rmse_scene += torch.sqrt(torch.mean((norm_output - norm_target) ** 2))

            if save_renders:
                if identifier is not None:
                    render_filpath = os.path.join(render_dir, 'test', f'{identifier}-test-{scene}-{scene_frame}.png')
                else:
                    render_filpath = os.path.join(render_dir, 'test', f'test-{scene}-{scene_frame}.png')

                top = (rgb_in[:, 0]).unsqueeze(1) ** (1 / 2.2)
                middle = output.unsqueeze(1) ** (1 / 2.2)
                bottom = target.unsqueeze(1) ** (1 / 2.2)

                save_sequence_test(top, bottom, middle, f'Frame {i}', render_filpath)

            frame_str = str(scene_frame).zfill(3)
            psnr_scene_avg = psnr_scene / (scene_frame+1)
            trmae_scene_avg = trmae_scene / (scene_frame+1)
            ssim_scene_avg = ssim_scene / (scene_frame+1)
            rmse_scene_avg = rmse_scene / (scene_frame+1)
            logging.info(f'{scene}, Frame {frame_str} | PSNR: {psnr_scene_avg:.5f}, TRMAE: {trmae_scene_avg:.5f}, SSIM: {ssim_scene_avg:.5f}, RMSE: {rmse_scene_avg:.5f}')

            scene_frame += 1

        # Have to do this for the last scene
        num_scenes += 1
        psnr_all += psnr_scene_avg
        trmae_all += trmae_scene_avg
        ssim_all += ssim_scene_avg
        rmse_all += rmse_scene_avg
        scene_info = f'{last_scene} Metrics | PSNR: {psnr_scene_avg}, TRMAE: {trmae_scene_avg}, SSIM: {ssim_scene_avg}, RMSE: {rmse_scene_avg}'
        logging.info(scene_info)
        log_to_file(log_filepath, scene_info)

        test_info = f'Test Metrics | PSNR: {psnr_all / num_scenes}, TRMAE: {trmae_all / num_scenes}, SSIM: {ssim_all / num_scenes}, RMSE: {rmse_all / num_scenes}'
        logging.info(test_info)
        log_to_file(log_filepath, test_info)
        

def main():
    args = parse_cli_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%H:%M:%S')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    logging.info('Initializing Autoencoder...')
    model = IsikNet(in_channel=9, embedding_dims=32, kernel_size=args.kernel_size, device=device)
    
    logging.info('Transferring Autoencoder to GPU...')
    model.to(device=device)

    if not args.save_renders:
        logging.warning('Command line arguments "save-renders" not specified - test renders will not be saved!')
    
    test(
        model=model,
        device=device,
        checkpoint_path=args.ckpt_path,
        testset_dir=args.testset_dir,
        render_dir=os.path.join(args.output_dir, 'render'),
        log_dir=os.path.join(args.output_dir, 'log'),
        identifier=args.id,
        width=args.width,
        height=args.height,
        save_renders=args.save_renders,
        stateless=args.stateless,
        flow_out=args.flow_out
    )

    logging.shutdown()

if __name__ == '__main__':
    main()