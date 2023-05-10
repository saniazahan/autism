import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_info',
                        type=str,
                        help='info that will be displayed when logging'
                        )

    parser.add_argument('--std',
                        type=float,
                        help='standard deviation for gaussian distribution learning',
                        default=5)

    parser.add_argument('--save',
                        action='store_true',
                        help='if set true, save the best model',
                        default=True)

    parser.add_argument('--lr',
                        type=float,
                        help='learning rate',
                        default=1e-5)

    parser.add_argument('--weight_decay',
                        type=float,
                        help='L2 weight decay',
                        default=1e-5)

    parser.add_argument('--temporal_aug',
                        type=int,
                        help='the maximum of random temporal shift, ranges from 0 to 6',
                        default=6)

    parser.add_argument('--seed',
                        type=int,
                        help='manual seed',
                        default=0)
    
    parser.add_argument('--phase',
                        type=int,
                        help='train or test phase',
                        default='train')

    parser.add_argument('--num_workers',
                        type=int,
                        help='number of subprocesses for dataloader',
                        default=1)

    parser.add_argument('--gpu',
                        type=str,
                        help='id of gpu device(s) to be used',
                        default='0')

    parser.add_argument('--train_batch_size',
                        type=int,
                        help='batch size for training phase',
                        default=64)

    parser.add_argument('--test_batch_size',
                        type=int,
                        help='batch size for test phase',
                        default=128)

    parser.add_argument('--num_epochs',
                        type=int,
                        help='number of training epochs',
                        default=100)

    return parser


