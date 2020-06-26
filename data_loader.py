import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data_utils


def get_data_loader(args):

    if args.dataset == 'cifar':
        # We can use an image folder dataset the way we have it setup.
        # Create the dataset
        dataset = dset.CIFAR10(root=args.dataroot, download=args.download,
                               transform=transforms.Compose([
                                   transforms.Resize(args.image_size),
                                   transforms.CenterCrop(args.image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                               ]))
        # Create the dataloader
        dataloader = data_utils.DataLoader(dataset, batch_size=args.batch_size,
                                           shuffle=True, num_workers=args.workers)

    return dataloader
