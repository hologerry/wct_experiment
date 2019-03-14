import argparse


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='WCT Pytorch')
        subparser = self.parser.add_subparsers(title="subcommands", dest="subcommand")

        #####
        # training arguments
        train_arg = subparser.add_parser("train", help="Parser for training arguments")
        # Dataset directory
        train_arg.add_argument('--run_id', type=int, required=True)
        # train_arg.add_argument('--dataset_dir', type=str, default='/S1/CSCL/gaoy/mscoco/',
        #                        help='root directory of mscoco dataset')
        train_arg.add_argument('--dataset_dir', type=str, default='/data/gaoy/coco/',
                               help='root directory of mscoco dataset')
        train_arg.add_argument('--train_img_dir', type=str, default='train2014',
                               help='Path to train image')
        train_arg.add_argument('--val_img_dir', type=str, default='val2014',
                               help='Path to validate image')
        train_arg.add_argument('--relu_target', type=str, required=True,
                               help='Target VGG19 relu layer to decode from, e.g. relu4_1')
        # Loss weight
        train_arg.add_argument('--pixel_weight', type=float, default=1.0,
                               help='Pixel reconstruction loss weight')
        train_arg.add_argument('--feature_weight', type=float, default=1.0,
                               help='Feature loss weight')
        train_arg.add_argument('--tv_weight', type=float, default=1.0,
                               help='Total variation loss weight')
        # hyperparameter
        train_arg.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
        train_arg.add_argument('--lr_decay_fr', type=float, default=10, help='Learning rate decay frequency')
        train_arg.add_argument("--log_fr", type=int, default=50, help="frequency of saving log")
        train_arg.add_argument('--epochs', type=int, default=15, help='Number of epochs')
        train_arg.add_argument('--d1_epochs', type=int, default=20, help='Number of epochs on training decoder1')
        train_arg.add_argument('--d2_epochs', type=int, default=32, help='Number of epochs on training decoder2')
        train_arg.add_argument('--d3_epochs', type=int, default=64, help='Number of epochs on training decoder3')
        train_arg.add_argument('--d4_epochs', type=int, default=100, help='Number of epochs on training decoder4')
        train_arg.add_argument('--d5_epochs', type=int, default=128, help='Number of epochs on traiing decoder5')
        train_arg.add_argument('--d1_batch_size', type=int, default=32, help='batch size')
        train_arg.add_argument('--d2_batch_size', type=int, default=16, help='batch size')
        train_arg.add_argument('--d3_batch_size', type=int, default=8, help='batch size')
        train_arg.add_argument('--d4_batch_size', type=int, default=4, help='batch size')
        train_arg.add_argument('--d5_batch_size', type=int, default=2, help='batch size')
        train_arg.add_argument('--img_size', type=int, default=256, help='Train image size')

        train_arg.add_argument('--workers', default=2, type=int, metavar='N',
                               help='Number of data loading workers (default: 4)')
        train_arg.add_argument('--vgg1', default='models/vgg_normalised_conv1_1.t7',
                               help='Path to the VGG conv1_1')
        train_arg.add_argument('--vgg2', default='models/vgg_normalised_conv2_1.t7',
                               help='Path to the VGG conv2_1')
        train_arg.add_argument('--vgg3', default='models/vgg_normalised_conv3_1.t7',
                               help='Path to the VGG conv3_1')
        train_arg.add_argument('--vgg4', default='models/vgg_normalised_conv4_1.t7',
                               help='Path to the VGG conv4_1')
        train_arg.add_argument('--vgg5', default='models/vgg_normalised_conv5_1.t7',
                               help='Path to the VGG conv5_1')
        train_arg.add_argument('--decoder5', default='models/feature_invertor_conv5_1.t7',
                               help='Path to the decoder5')
        train_arg.add_argument('--decoder4', default='models/feature_invertor_conv4_1.t7',
                               help='Path to the decoder4')
        train_arg.add_argument('--decoder3', default='models/feature_invertor_conv3_1.t7',
                               help='Path to the decoder3')
        train_arg.add_argument('--decoder2', default='models/feature_invertor_conv2_1.t7',
                               help='Path to the decoder2')
        train_arg.add_argument('--decoder1', default='models/feature_invertor_conv1_1.t7',
                               help='Path to the decoder1')
        train_arg.add_argument('--cuda', type=int, default=1, help='enables cuda')
        train_arg.add_argument('--gpu', type=int, default=0, help='gpu id')
        train_arg.add_argument('--fineSize', type=int, default=512,
                               help='resize image to fineSize x fineSize,leave it to 0 if not resize')
        train_arg.add_argument('--output_dir', default='samples/', help='folder to output images')
        train_arg.add_argument('--alpha', type=float, default=1,
                               help='hyperparameter to blend wct feature and content feature')

        ###
        # test arguments
        test_arg = subparser.add_parser("test", help="parser for training arguments")
        test_arg.add_argument('--content_path', default='images/content', help='path to content')
        test_arg.add_argument('--style_path', default='images/style', help='path to style')

    def parse(self):
        return self.parser.parse_args()
