import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR', default='./datasets/test-100',
                    help='path to dataset')
parser.add_argument('--dataset-name', default='test-100',
                    help='dataset name', choices=['stl10', 'cifar10', 'test-100', 'single_organoids', 'train-100', 'test-unlabeled'])
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
    args.n_views = 1
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
                #model.load_state_dict(checkpoint)
                #optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
                #test_loader = get_test_loader(args)
                test_loader, ground_truth_ids = get_test_loader_with_targets(args)
                test(model, test_loader, args, ground_truth_ids=ground_truth_ids)
                #criterion = torch.nn.MSELoss().to(args.device)
                #test(model, test_loader, args)
            else:
                print("=> no checkpoint found at '{}'".format(args.checkpoint))
        else:
            simclr.train(train_loader)


if __name__ == "__main__":
    import os
    import sys
    import matplotlib.pyplot as plt
    
    def plot_tsne(embeddings, cluster_ids=None, save_path="tsne.png", title="t-SNE of SimCLR Embeddings"):
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
        tsne_results = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        if cluster_ids is not None:
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_ids, cmap='tab10', s=5)
            plt.colorbar(label='Cluster ID')
        else:
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=5, alpha=0.6)
        plt.title(title)
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        return tsne_results

    def plot_tsne_3d(embeddings, cluster_ids=None, save_path="tsne_3d.png", title="3D t-SNE of SimCLR Embeddings"):
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        tsne = TSNE(n_components=3, perplexity=30, init='pca', random_state=42)
        tsne_results = tsne.fit_transform(embeddings)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if cluster_ids is not None:
            scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2],
                                c=cluster_ids, cmap='tab10', s=10, alpha=0.7)
            legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend1)
        else:
            ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2],
                    s=10, alpha=0.7)

        ax.set_title(title)
        ax.set_xlabel("TSNE 1")
        ax.set_ylabel("TSNE 2")
        ax.set_zlabel("TSNE 3")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.show()
        return tsne_results


    def test(model, test_loader, args, ground_truth_ids=None):
        model.eval()
        all_embeddings = []
        with torch.no_grad():
            for i, (images, _) in enumerate(test_loader):
                images = torch.cat(images, dim=0)
                if args.device == torch.device('cuda'):
                    images = images.cuda(args.gpu_index, non_blocking=True)
                output = model(images)
                all_embeddings.append(output.cpu())
        all_embeddings = torch.cat(all_embeddings, dim=0)
        print(f"Extracted embeddings shape: {all_embeddings.shape}")

        # Plot t-SNE with ground truth clusters if available
        if ground_truth_ids is not None:
            plot_tsne(
                all_embeddings.numpy(),
                cluster_ids=ground_truth_ids,
                save_path="tsne_ground_truth.png",
                title="t-SNE of SimCLR Embeddings (Ground Truth Clusters)"
            )
            plot_tsne_3d(
                all_embeddings.numpy(),
                cluster_ids=ground_truth_ids,  # or ground_truth_ids
                save_path="tsne_3d_kmeans.png",
                title="3D t-SNE with KMeans"
            )


        # Plot t-SNE without clusters
        tsne_results = plot_tsne(
            all_embeddings.numpy(),
            save_path="tsne_unlabeled.png",
            title="t-SNE of SimCLR Embeddings (Unlabeled)"
        )

        from sklearn.cluster import KMeans
        k = 10# try 3–10 and see which gives meaningful clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_ids = kmeans.fit_predict(all_embeddings.numpy())

        # Plot t-SNE with cluster coloring
        plot_tsne(
            all_embeddings.numpy(),
            cluster_ids=cluster_ids,
            save_path="tsne_kmeans.png",
            title=f"t-SNE of SimCLR Embeddings with KMeans (k={k})"
        )

        print(f"Number of embeddings: {all_embeddings.shape[0]}")
        plot_cosine_similarity_matrix(all_embeddings.numpy())


    def plot_cosine_similarity_matrix(embeddings, save_path="cosine_similarity_matrix.png"):
        # Compute cosine similarity
        sim_matrix = cosine_similarity(embeddings)

        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(sim_matrix, cmap="viridis", xticklabels=False, yticklabels=False)
        plt.title("Cosine Similarity Matrix of SimCLR Embeddings")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Cosine similarity matrix saved to: {save_path}")


    def get_test_loader(args):
        dataset = ContrastiveLearningDataset()
        test_dataset = dataset.get_dataset(args.data, args.dataset_name, args.n_views)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False)
        return test_loader

    def get_test_loader_with_targets(args):
        dataset = ContrastiveLearningDataset()
        test_dataset = dataset.get_dataset(args.data, args.dataset_name, args.n_views)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False)
        # Extract ground truth cluster IDs
        ground_truth_ids = test_dataset.targets if hasattr(test_dataset, 'targets') else None
        return test_loader, ground_truth_ids
    
    def finetune_supervised(checkpoint_path, data_dir, num_classes=10, epochs=10, device='cuda'):
        from torchvision import datasets, transforms
        import torch
        import torch.nn as nn
        import torch.optim as optim

        # Data
        supervised_transform = transforms.Compose([
            transforms.Resize(96),
            transforms.CenterCrop(96),
            transforms.ToTensor(),
        ])
        train_dataset = datasets.ImageFolder(data_dir, transform=supervised_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

        # Model
        model = ResNetSimCLR(base_model='resnet18', out_dim=128)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        encoder = model.backbone

        # Linear head
        classifier = nn.Linear(128, num_classes)
        model_with_head = nn.Sequential(encoder, classifier)
        model_with_head = model_with_head.to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model_with_head.parameters(), lr=1e-4)

        # Training loop
        model_with_head.train()
        for epoch in range(epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model_with_head(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        print("Fine-tuning complete.")

        # Save the fine-tuned model
        torch.save({ 'epoch': epochs,
            'arch': 'resnet18',
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()}, "finetuned_model.pth")
        print("Fine-tuned model saved to finetuned_model.pth")

    checkpoint_path = "runs/May27_08-44-15_kirk/checkpoint_0200.pth.tar"
    data_dir = "./datasets/train-100"
    num_classes = 10  # Set to the number of classes in your dataset
    epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #finetune_supervised(
    #    checkpoint_path=checkpoint_path,
    #    data_dir=data_dir,
    #    num_classes=num_classes,
    #    epochs=epochs,
    #    device=device
    #)

    main()
