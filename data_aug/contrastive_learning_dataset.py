from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
import os


class ContrastiveLearningDataset:

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, data_dir, name, n_views):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(data_dir, train=True,
                                                transform=ContrastiveLearningViewGenerator(
                                                    self.get_simclr_pipeline_transform(32),
                                                    n_views),
                                                download=True),

            'stl10': lambda: datasets.STL10(data_dir, split='unlabeled',
                                            transform=ContrastiveLearningViewGenerator(
                                                self.get_simclr_pipeline_transform(96),
                                                n_views),
                                            download=True),
            'test-100': lambda: datasets.ImageFolder(data_dir,
                                                    transform=ContrastiveLearningViewGenerator(
                                                        self.get_simclr_pipeline_transform(96),
                                                        n_views)),
            'test-unlabeled': lambda: datasets.ImageFolder(data_dir,
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views))
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
