import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets/test-100',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='test-100',
                    help='dataset name', choices=['stl10', 'cifar10', 'test-100'])
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    dataset = ContrastiveLearningDataset()

    train_dataset = dataset.get_dataset(args.data, args.dataset_name, args.n_views)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        if args.checkpoint:
            if os.path.isfile(args.checkpoint):
                print("=> loading checkpoint '{}'".format(args.checkpoint))
                checkpoint = torch.load(args.checkpoint)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
                test_loader = get_test_loader(args)
                #criterion = torch.nn.MSELoss().to(args.device)
                test(model, test_loader, args)
            else:
                print("=> no checkpoint found at '{}'".format(args.checkpoint))
        else:
            simclr.train(train_loader)


if __name__ == "__main__":
    import os
    import sys
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    def test(model, test_loader, args):
        model.eval()
        all_embeddings = []
        with torch.no_grad():
            for i, (images, _) in enumerate(test_loader):
                images = torch.cat(images, dim=0)
                if args.device == torch.device('cuda'):
                    images = images.cuda(args.gpu_index, non_blocking=True)
                
                # get the output from the model
                output = model(images)
                
                # calculate the loss
                all_embeddings.append(output.cpu())
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        print(f"Extracted embeddings shape: {all_embeddings.shape}")
        # Run t-SNE on the embeddings
        tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
        tsne_results = tsne.fit_transform(all_embeddings.numpy())

        # Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=5, alpha=0.6)
        plt.title('t-SNE of SimCLR Embeddings (Unlabeled)')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.tight_layout()
        plt.savefig("tsne_unlabeled.png", dpi=300)
        plt.show()
    
    def get_test_loader(args):
        dataset = ContrastiveLearningDataset()
        test_dataset = dataset.get_dataset(args.data, args.dataset_name, args.n_views)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False)
        return test_loader
    
    main()
