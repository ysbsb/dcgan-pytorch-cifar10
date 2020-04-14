import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pytorch implementation of GAN models."
    )

    parser.add_argument("--model", type=str, default="dcgan", choices=["dcgan"])
    parser.add_argument("--is_train", type=str, default="True")
    parser.add_argument(
        "--dataroot", type=str, default="dataset/cifar", help="path to dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar",
        choices=["cifar"],
        help="The name of dataset",
    )
    parser.add_argument("--download", type=str, default="True")
    parser.add_argument(
        "--epochs", type=int, default=25, help="The number of epochs to run"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="The size of batch")
    parser.add_argument(
        "--image_size", type=int, default=64, help="Spatial size of training images."
    )
    parser.add_argument(
        "--nc", type=int, default=3, help="Number of channels in the training images"
    )
    parser.add_argument("--nz", type=int, default=100, help="Size of z latent vector")
    parser.add_argument(
        "--ngf", type=int, default=64, help="Size of feature maps in generator"
    )
    parser.add_argument(
        "--ndf", type=int, default=64, help="Size of feature maps in discriminator"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0002,
        help="Learning rate for optimizers",
    )
    parser.add_argument(
        "--beta1", type=float, default=0.5, help="Beta1 hyperparam for Adam optimizers"
    )
    parser.add_argument("--ngpu", type=int, default=1, help="Number of GPUs available.")
    parser.add_argument("--cuda", type=str, default="True", help="Availability of cuda")
    parser.add_argument(
        "--load_D",
        type=str,
        default="False",
        help="Path for loading Discriminator network",
    )
    parser.add_argument(
        "--load_G", type=str, default="False", help="Path for loading Generator network"
    )
    parser.add_argument("--workers", type=int, default=2, help="The number of workers.")
    parser.add_argument(
        "--generator_iters",
        type=int,
        default=10000,
        help="The number of iterations for generator in WGAN model.",
    )
    parser.add_argument("--gpuids", default=[0], help="GPU ids for using (Default: 0)")

    cfg = parser.parse_args()
    cfg.gpuids = list(map(int, cfg.gpuids))

    return cfg
