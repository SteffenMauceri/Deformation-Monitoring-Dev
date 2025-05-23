import glob
import os, sys
import time
import torch
from torch import optim
import torch.nn as nn
import timeit
import math
import numpy as np

import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_train
from utils.utils import setup_seed, init_weight, netParams
from torch.utils.tensorboard import SummaryWriter
# Import the denormalization function
from dataset.phaseUnwrapping import denormalize_label


sys.setrecursionlimit(1000000)  # solve problem 'maximum recursion depth exceeded'

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable
print(torch_ver)

GLOBAL_SEED = 1234


def parse_args():
    parser = ArgumentParser(description='PUNet and DDNet')
    # model and dataset
    parser.add_argument('--model', type=str, default="PUNet", help="model name: (default PUNet)")
    parser.add_argument('--dataRootDir', type=str, default=r"dataset",
                        help="dataset dir")
    parser.add_argument('--dataset', type=str, default="phaseUnwrapping", help="dataset")
    parser.add_argument('--input_size', type=str, default="256,256", help="input size of model")
    parser.add_argument('--num_workers', type=int, default=4, help=" the number of parallel threads")
    parser.add_argument('--num_channels', type=int, default=1,
                        help="the num_channels ")
    # training hyper params
    parser.add_argument('--max_epochs', type=int, default=500,
                        help="the number of epochs")
    parser.add_argument('--random_mirror', type=bool, default=True, help="input image random mirror")
    parser.add_argument('--lr', type=float, default=1e-4, help="initial learning rate")
    parser.add_argument('--batch_size', type=int, default=5, help="the batch size is set to 16 for 2 GPUs")
    parser.add_argument('--optim', type=str.lower, default='adam', choices=['sgd', 'adam'],
                        help="select optimizer")
    parser.add_argument('--poly_exp', type=float, default=0.95, help='polynomial LR exponent')
    # cuda setting
    parser.add_argument('--cuda', type=bool, default=True, help="running on CPU or GPU")
    parser.add_argument('--gpus', type=str, default="0", help="default GPU devices (0,1)")
    # checkpoint and log
    parser.add_argument('--resume', type=str, default="",
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument('--savedir', default="./checkpoint/", help="directory to save the model snapshot")
    parser.add_argument('--logFile', default="log.txt", help="storing the training and validation logs")
    args = parser.parse_args()

    return args


def train_model(args):
    """
    args:
       args: global arguments
    """
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    print("=====> input size:{}".format(input_size))

    print(args)

    device = "cpu" # Default to CPU
    if args.cuda:
        if torch.cuda.is_available():
            print("=====> use gpu id: '{}'".format(args.gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            device = "cuda"
        elif torch.backends.mps.is_available(): # Check for MPS
            print("=====> using Apple Metal Performance Shaders (MPS)")
            device = "mps"
        else:
            print("=====> No GPU (CUDA or MPS) available/selected, running on CPU")
            # Optionally raise an exception if GPU was explicitly requested but none found
            # raise Exception("GPU specified but no CUDA or MPS found, please run without --cuda or check setup")
    else:
        print("=====> Running on CPU")

    # set the seed
    setup_seed(GLOBAL_SEED)
    print("=====> set Global Seed: ", GLOBAL_SEED)

    cudnn.enabled = True
    print("=====> building network")

    # build the model and initialization
    model = build_model(args.model, num_channels=args.num_channels)
    init_weight(model, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-3, 0.1,
                mode='fan_in')

    print("=====> computing network parameters and FLOPs")
    total_paramters = netParams(model)
    print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))

    # load data and data augmentation
    datas, trainLoader, valLoader = build_dataset_train(args.dataRootDir, args.dataset, input_size, args.batch_size,
                                                        args.random_mirror, args.num_workers)

    args.per_iter = len(trainLoader)
    args.max_iter = args.max_epochs * args.per_iter

    if args.dataset == 'phaseUnwrapping':
        criteria = nn.MSELoss(reduction='mean')
    else:
        raise NotImplementedError(
            "not support dataset: %s" % args.dataset)

    # Move model and criteria to the selected device
    model = model.to(device)
    criteria = criteria.to(device)

    if device == "cuda" and torch.cuda.device_count() > 1: # Specific multi-GPU handling for CUDA
        print("torch.cuda.device_count()=", torch.cuda.device_count())
        args.gpu_nums = torch.cuda.device_count()
        model = nn.DataParallel(model) # .cuda() is implicitly handled by .to(device) now
        args.savedir = (args.savedir + args.dataset + '/' + args.model + 'bs'
                    + str(args.batch_size) + 'gpu' + str(args.gpu_nums) + '/')
    elif device != "cpu": # Single GPU (CUDA or MPS) or MPS
        args.gpu_nums = 1
        print(f"=====> Using single {device.upper()} device for training")
        args.savedir = (args.savedir + args.dataset + '/' + args.model + 'bs'
                    + str(args.batch_size) + device.lower() + str(args.gpu_nums) + '/')
    else: # CPU
        args.savedir = (args.savedir + args.dataset + '/' + args.model + 'bs'
                    + str(args.batch_size) + 'cpu/')
    os.makedirs(args.savedir,exist_ok=True)

    # TensorBoard Writer
    tb_log_dir = os.path.join(args.savedir, 'runs') # Define a subdirectory for TensorBoard logs
    writer = SummaryWriter(log_dir=tb_log_dir)

    start_epoch = 0

    # continue training
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            print("=====> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=====> no checkpoint found at '{}'".format(args.resume))
    else:
        try:
            p = sorted(glob.glob(os.path.join('checkpoint', args.dataset, args.model + 'bs*', '*.pth')))[-1]
            print("=====> loading checkpoint '{}'".format(p))
            checkpoint = torch.load(p)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            print("=====> loaded checkpoint '{}' (epoch {})".format(p, checkpoint['epoch']))
        except:
            print("=====> no checkpoint found at '{}'".format(args.resume))

    model.train()
    cudnn.benchmark = True
    # cudnn.deterministic = True ## my add

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s Seed: %s" % (str(total_paramters), GLOBAL_SEED))
        logger.write("\n%s\t%s\t\t%s\t%s\t\t%s\t%s" % ('Epoch', 'lr', 'Loss(Tr)', 'Loss(Val)', 'RMSE (val)', 'MAE(val)'))
    logger.flush()

    # define optimization strategy
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
            weight_decay=1e-4)
    else:
        raise NotImplementedError(
            "not supported: %s" % args.optim)

    lossTr_list = []
    epoches = []
    mRMSE_val_list = []

    # Initialize the learning rate scheduler once
    lambda1 = lambda epoch: math.pow((1 - (epoch / args.max_epochs)), args.poly_exp)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    print('=====> beginning training')
    for epoch in range(start_epoch, args.max_epochs):
        # training
        lossTr, lr = train(args, trainLoader, model, criteria, optimizer, epoch, device)
        lossTr_list.append(lossTr)

        # validation
        if epoch % 1 == 0 or epoch == (args.max_epochs - 1):
            epoches.append(epoch)
            rmse, mae, val_loss = val(args, valLoader, model, device, criteria)
            mRMSE_val_list.append(rmse)
            # record train information
            logger.write("\n%d\t%.7f\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (epoch, lr, lossTr, val_loss, rmse, mae))
            logger.flush()
            print("Epoch : " + str(epoch) + ' Details')
            print("Epoch No.: %d\tTrain Loss = %.4f\tVal Loss = %.4f\tmRMSE(val) = %.4f\t lr= %.6f\n" % (epoch,
                                                                                        lossTr,
                                                                                        val_loss,
                                                                                        rmse, lr))
            # TensorBoard logging
            writer.add_scalar('Loss/train', lossTr, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('RMSE/validation', rmse, epoch)
            writer.add_scalar('MAE/validation', mae, epoch)
            writer.add_scalar('Learning_Rate', lr, epoch)
        else:
            # record train information
            logger.write("\n%d\t%.7f\t%.4f" % (epoch,lr, lossTr))
            logger.flush()
            print("Epoch : " + str(epoch) + ' Details')
            print("Epoch No.: %d\tTrain Loss = %.4f\t lr= %.6f\n" % (epoch, lossTr, lr))
            # TensorBoard logging (for epochs where validation isn't run)
            writer.add_scalar('Loss/train', lossTr, epoch)
            writer.add_scalar('Learning_Rate', lr, epoch)

        # Step the scheduler once per epoch, after validation
        scheduler.step()

        # save the model
        if epoch % 1 == 0 or epoch == (args.max_epochs - 1):
            model_file_name = args.savedir + '/model_' + '%04d'%(epoch + 1) + '.pth'
            state = {"epoch": epoch + 1, "model": model.state_dict()}
            torch.save(state, model_file_name)
            print("Model saved: %s" % model_file_name)

    logger.close()
    writer.close() # Close the TensorBoard writer


def train(args, train_loader, model, criterion, optimizer, epoch, device):
    """
    args:
       train_loader: loaded for training dataset
       model: model
       criterion: loss function
       optimizer: optimization algorithm, such as ADAM or SGD
       epoch: epoch number
    return: average loss
    """

    model.train()
    epoch_loss = []

    total_batches = len(train_loader)
    print("=====> the number of iterations per epoch: ", total_batches)
    st = time.time()
    for iteration, batch in enumerate(train_loader, 0):

        args.per_iter = total_batches
        args.max_iter = args.max_epochs * args.per_iter
        args.cur_iter = epoch * args.per_iter + iteration
        # learming scheduling
        # lambda1 = lambda epoch: math.pow((1 - (args.cur_iter / args.max_iter)), args.poly_exp)
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

        lr = optimizer.param_groups[0]['lr']

        start_time = time.time()
        images, labels, _, names = batch # Unpack names

        # Move data to the selected device
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)

        loss = criterion(output, labels)

        if loss.item() > 1000000: # Add this check
            print(f"High loss detected: {loss.item()} in batch with files: {names}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()  # In pytorch 1.1.0 and later, should call 'optimizer.step()' before 'lr_scheduler.step()'
        epoch_loss.append(loss.item())
        time_taken = time.time() - start_time

        print('=====> epoch[%d/%d] iter: (%d/%d) \tcur_lr: %.6f loss: %.3f time:%.2f' % (epoch + 1, args.max_epochs,
                                                                                         iteration + 1, total_batches,
                                                                                         lr, loss.item(), time_taken))

    time_taken_epoch = time.time() - st
    remain_time = time_taken_epoch * (args.max_epochs - 1 - epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    print("Remaining training time = %d hour %d minutes %d seconds" % (h, m, s))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_train, lr


def val(args, val_loader, model, device, criterion):
    """
    args:
      val_loader: loaded for validation dataset
      model: model
      device: device to run on
      criterion: loss function
    return: mean rmses, mean maes, average validation loss
    """
    # evaluation mode
    model.eval()
    total_batches = len(val_loader)
    epoch_val_loss = [] # To store batch losses

    rmses = 0
    maes = 0
    for i, (input, label, size, name) in enumerate(val_loader):
        start_time = time.time()
        with torch.no_grad():
            # Move data to the selected device
            input_var = input.to(device)
            # label_var is already normalized by the DataLoader
            label_var = label.to(device) 
            output = model(input_var)
            
            loss = criterion(output, label_var) # Loss is calculated on normalized output vs normalized label
            epoch_val_loss.append(loss.item())

        time_taken = time.time() - start_time
        print("[%d/%d]  time: %.2f" % (i + 1, total_batches, time_taken))
        
        # Denormalize output and label for metrics calculation
        output_denorm = denormalize_label(output.cpu().numpy())
        # The label from val_loader is already normalized, so we need to get the original gt for metrics.
        # This is tricky as the original label isn't directly passed. 
        # For now, we'll denormalize the normalized label_var for metric calculation.
        # This assumes criterion works on normalized values, and metrics on denormalized.
        gt_denorm = denormalize_label(label_var.cpu().numpy())

        rmses += np.sqrt(np.mean((output_denorm - gt_denorm) ** 2))
        maes += np.mean(np.abs(output_denorm - gt_denorm))
    
    average_val_loss = sum(epoch_val_loss) / len(epoch_val_loss) # Calculate average loss
    return rmses/(i+1), maes/(i+1), average_val_loss


if __name__ == '__main__':
    start = timeit.default_timer()
    args = parse_args()

    train_model(args)
    end = timeit.default_timer()
    hour = 1.0 * (end - start) / 3600
    minute = (hour - int(hour)) * 60
    print("training time: %d hour %d minutes" % (int(hour), int(minute)))
