import os
from train_utils import *
from transform import *
from LoG import *
from model import *
from dataset import MonteCarloDataset
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, random_split
from pytorch_msssim import ssim
import logging
import argparse
from early_stopping import EarlyStopper
from math import log10, floor

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="Custom identifier prepending checkpoint, log, plot, etc. names for a given run", required=True)
    parser.add_argument("--output-dir", help="Directory containing render/, plot/, checkpoint/ and log/", required=True)
    parser.add_argument("--dataset-dir", help="Directory containing albedo/, depth/, normal/ and rgb/ folders", required=True)
    parser.add_argument("--kernel-size", help="Set the kernel size to be used in every convolutional layer", type=int, default=3)
    parser.add_argument("--save-renders", help="Save renders for every K-th of the epoch completed", type=int)
    parser.add_argument("--num-epochs", help="Force training to stop after a certain number of epochs", type=int)
    parser.add_argument("--patience", help="The number of epochs with (loss > min_val_loss + min_delta) not interuppted by (loss < min_val_loss) needed to stop training early", type=int, required=True)
    parser.add_argument("--min-delta", help="If (loss > min_val_loss + min_delta) the early stopping counter increments. If loss exceeds min_val_loss by less than least min_delta, the counter is sustained. If loss is less than min_val_loss, the counter is reset to 0", type=float, required=True)
    parser.add_argument("--seq-len", help="Number of frames in each minibatch element", type=int)
    parser.add_argument("--batch-size", help="Number of sequences in each minibatch", type=int)
    return parser.parse_args()

def train_model(
        model,
        device,
        num_epochs,
        batch_size,
        learning_rate,
        decay_rates,
        val_frac,
        save_checkpoint,
        crop_size,
        mod_interval,
        sequence_len,
        checkpoint_dir,
        plot_dir,
        render_dir,
        log_dir,
        identifier,
        dataset_dir,
        save_renders,
        patience,
        min_delta
):
    logging.info('Initializing dataset...')
    dataset = MonteCarloDataset(root_dir=dataset_dir,
                                subseq_len=sequence_len,
                                transform=transforms.Compose([
                                    RandomCrop(crop_size),
                                    AlbedoDemodulateBoring(),
                                    TonemapIsik(),
                                    ToTensor()
                                ]))
    
    logging.info('Splitting dataset...')
    n_val = int(len(dataset) * val_frac)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    logging.info('Initializing training dataloader...')
    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True)
    train_len = len(train_loader)

    logging.info('Initializing validation dataloader...')
    val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size, num_workers=8, pin_memory=True)
    val_len = len(val_loader)

    logging.info('Initializing optimizer...')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=decay_rates)

    logging.info('Initializing scheduler...')
    lrs = lambda epoch: 0.75 ** (epoch // 150)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lrs])

    logging.info('Initializing loss curve array...')
    global_train_loss_curve = np.zeros(num_epochs)
    global_val_loss_curve = np.zeros(num_epochs)

    logging.info('Initialising Early Stopper...')
    log_filename = f'{identifier}-train.txt'
    log_filepath = os.path.join(log_dir, 'train', log_filename)

    early_stopper = EarlyStopper(log_filepath=log_filepath, patience=patience, min_delta=min_delta)

    l1_loss = nn.L1Loss()

    regular_ckpt_filename = f'{identifier}-regular.pt'
    best_ckpt_filename = f'{identifier}-best.pt'
    ckpt_probe = os.path.join(checkpoint_dir, regular_ckpt_filename)
    logging.info(f'Testing for existence of regular model checkpoint at {ckpt_probe}')
    if os.path.isfile(ckpt_probe):
        logging.info(f'\tCheckpoint found, resuming in-progress training run')
        checkpoint = torch.load(ckpt_probe)

        start_epoch = checkpoint['epoch']
        ckpt_loss = checkpoint['loss']
        early_stopper = checkpoint['early_stopper']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        np.random.set_state(checkpoint['random_state_np'])
        torch.set_rng_state(checkpoint['random_state_torch'])
        torch.cuda.set_rng_state_all(checkpoint['random_state_torch_cuda'])

        min_val_loss = early_stopper.min_validation_loss

        logging.info(f'''Loaded checkpoint:
            Loss: {ckpt_loss}
            Epoch: {start_epoch}
            Early stopper min: {min_val_loss}
            Early stopper counter: {early_stopper.counter}''')
    else:
        start_epoch = 0
        min_val_loss = float('inf')


    init_info = f'''Starting training:
        Epochs:          {num_epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Betas:           {decay_rates}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images Cropping: {crop_size}
        Mod. Interval:   {mod_interval}
        Sequence Length: {sequence_len}
        Identifier:      {identifier}
        Patience:        {patience}
        Min Delta:       {min_delta}'''

    log_to_file(log_filepath, init_info)
    logging.info(init_info)

    zfill_batch = floor(log10(len(train_loader))) + 1
    zfill_batch_val = floor(log10(len(val_loader))) + 1
    zfill_epoch = floor(log10(num_epochs)) + 1

    # Train. Batch tensor = N x M x C x H x W   
    for epoch in range(start_epoch, num_epochs):
        epoch_str = str(epoch+1).zfill(zfill_epoch)
        model.train()
        loss_epoch = l_recons_epoch = l_temporal_epoch = l_reg_epoch = 0
        for i, batch in enumerate(train_loader):
            inputs, targets, albedo, motion, rgb_in, normal_vanilla, depth = (batch[x].to(device) for x in ['inputs', 'targets', 'albedo', 'motion', 'rgb_in', 'normal_vanilla', 'depth'])
            N, M, C, H, W = targets.shape
            outputs = torch.zeros(N, M, C, H, W, device=device)
            params_a = torch.zeros(N, M, 3, H, W, device=device)
            params_b = torch.zeros(N, M, H, W, device=device)

            for j in range(M):
                output, params_a[:, j], params_b[:, j] = model(inputs[:, j], motion[:, j], rgb_in[:, j], normal_vanilla[:, j], depth[:, j]) 
                outputs[:, j] = output * (albedo[:, j] + 1e-2)

            l_recons = SMAPE(outputs, targets)
            l_temporal = 0.25 * SMAPE(*finite_differencing(outputs, targets))
            l_reg = 1e-5 * reg_loss(params_a, params_b)
            loss = l_recons + l_temporal + l_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            model.clear_state()
            
            with torch.no_grad():
                if save_renders is not None and i % ((train_len - 1) // save_renders) == 0:
                    render_filepath = os.path.join(render_dir, 'train', f'{identifier}-train-{epoch}-{i}.png')
                    save_sequence(inputs, targets, outputs, f'Epoch {epoch+1}/{num_epochs}, Training Batch {i+1}/{train_len}', render_filepath)
                    logging.info(f"Saved training render to {render_filepath}")

                batch_str = str(i+1).zfill(zfill_batch)
                loss_epoch += loss
                l_recons_epoch += l_recons
                l_temporal_epoch += l_temporal
                l_reg_epoch += l_reg
                logging.info(f'Epoch {epoch_str}/{num_epochs} Train Batch {batch_str}/{train_len} | L: {loss_epoch/(i+1):.5f} LR: {l_recons_epoch/(i+1):.5f} LT: {l_temporal_epoch/(i+1):.5f} Lreg: {l_reg_epoch/(i+1):.5f}')

        # Validation
        with torch.inference_mode():
            model.eval()
            loss_epoch_val = l_recons_epoch_val = l_temporal_epoch_val = l_reg_epoch_val = 0
            for i, batch in enumerate(val_loader):
                loss = 0
                inputs, targets, albedo, motion, rgb_in, normal_vanilla, depth = (batch[x].to(device) for x in ['inputs', 'targets', 'albedo', 'motion', 'rgb_in', 'normal_vanilla', 'depth'])
                N, M, C, H, W = targets.shape
                outputs = torch.zeros(N, M, C, H, W, device=device)
                params_a = torch.zeros(N, M, 3, H, W, device=device)
                params_b = torch.zeros(N, M, H, W, device=device)

                for j in range(M):
                    output, params_a[:, j], params_b[:, j] = model(inputs[:, j], motion[:, j], rgb_in[:, j], normal_vanilla[:, j], depth[:, j]) 
                    outputs[:, j] = output * (albedo[:, j] + 1e-2)

                l_recons = SMAPE(outputs, targets)
                l_temporal = 0.25 * SMAPE(*finite_differencing(outputs, targets))
                l_reg = 1e-5 * reg_loss(params_a, params_b)
                loss = l_recons + l_temporal + l_reg
                
                model.clear_state()

                if save_renders is not None and i % ((val_len - 1) // save_renders) == 0:
                    render_filepath = os.path.join(render_dir, 'val', f'{identifier}-val-{epoch}-{i}.png')
                    save_sequence(inputs, targets, outputs, f'Epoch {epoch+1}/{num_epochs}, Val. Batch {i+1}/{train_len}', render_filepath)
                    logging.info(f"Saved val. render to {render_filepath}")

                batch_str = str(i+1).zfill(zfill_batch_val)
                loss_epoch_val += loss
                l_recons_epoch_val += l_recons
                l_temporal_epoch_val += l_temporal
                l_reg_epoch_val += l_reg
                logging.info(f'Epoch {epoch_str}/{num_epochs} Val. Batch {batch_str}/{val_len} | L: {loss_epoch_val/(i+1):.5f} LR: {l_recons_epoch_val/(i+1):.5f} LT: {l_temporal_epoch_val/(i+1):.5f} Lreg: {l_reg_epoch_val/(i+1):.5f}')

        scheduler.step()
                
        mean_train_loss = loss_epoch/train_len
        mean_val_loss = loss_epoch_val/val_len
        global_train_loss_curve[epoch] = mean_train_loss
        global_val_loss_curve[epoch] = mean_val_loss
        epoch_info = f'Epoch {epoch_str}/{num_epochs} | Mean Train Loss: {mean_train_loss} Mean Val. Loss: {mean_val_loss}'
        logging.info(epoch_info) 
        log_to_file(log_filepath, epoch_info)

        if epoch > 0:
            save_global_loss_plot(global_train_loss_curve[:epoch+1], global_val_loss_curve[:epoch+1], os.path.join(plot_dir, f'{identifier}-{epoch}.png'))

        if mean_val_loss < min_val_loss:
            min_val_loss = mean_val_loss
            new_best = True
        else:
            new_best = False

        if early_stopper.early_stop(mean_val_loss):
            termination_info = f'Early stopping criterion met at epoch {epoch}'
            logging.info(termination_info)
            log_to_file(log_filepath, termination_info)
            break

        if save_checkpoint:
            state_dict = create_state_dict(epoch, mean_val_loss, early_stopper, model, optimizer, scheduler)
            save_checkpoint_func(state_dict, checkpoint_dir, regular_ckpt_filename)
            if new_best:
                save_checkpoint_func(state_dict, checkpoint_dir, best_ckpt_filename)        

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
    
    train_model(        
        model=model,
        device=device,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=1e-4,
        decay_rates=(0.9,0.999),
        val_frac=0.2,
        save_checkpoint=True,
        crop_size=128,
        mod_interval=(0,2),
        sequence_len=args.seq_len,
        checkpoint_dir=os.path.join(args.output_dir, 'checkpoint'),
        plot_dir=os.path.join(args.output_dir, 'plot'),
        render_dir=os.path.join(args.output_dir, 'render'),
        log_dir=os.path.join(args.output_dir, 'log'),
        identifier=args.id,
        dataset_dir=args.dataset_dir,
        save_renders=args.save_renders,
        patience=args.patience,
        min_delta=args.min_delta
    )

    logging.shutdown()

if __name__ == '__main__':
    main()