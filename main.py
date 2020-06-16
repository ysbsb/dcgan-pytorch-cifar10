from utils.config import parse_args
from utils.data_loader import get_data_loader
from model.dcgan import DCGAN


def main(args):
    model = None
    if args.model == 'dcgan':
        model = DCGAN(args)
    else:
        print("Model type non-existing. Try again.")
        exit(-1)

    print('----------------- configuration -----------------')
    for k, v in vars(args).items():
        print('  {}: {}'.format(k, v))
    print('-------------------------------------------------')

    data_loader = get_data_loader(args)

    model.train(data_loader)


if __name__ == '__main__':
    args = parse_args()
    main(args)
