
class InitParser(object):
    def __init__(self):

        self.do_you_wanna_train = True                                              # 'Training will start'
        self.do_you_wanna_load_weights = False                                        # 'Load weights'
        self.do_you_wanna_check_accuracy = False                                      # 'Model will be tested after the training or only this is done'

        # gpu setting
        self.multi_gpu = False                                                       # 'Decide to use one or more GPUs'
        self.gpu_id = 0                                                              # 'Select the GPUs for training and testing'
        # optimizer setting
        self.lr = 0.001                                                              # 'Learning rate'
        self.weight_decay = 1e-4                                                     # 'Weight decay'
        # train setting
        self.increase_factor_data = 0                                                # 'Increase the data number passed each epoch'
        # self.resample = True                                                       # 'Decide or not to rescale the images to a new resolution'
        # self.new_resolution = (2, 2, 8)   # 2.5 before                             # 'New resolution'
        self.resize = True
        self.new_size = [128, 128, 32]
        # self.patch_size = [192, 192, 32]                                           # "Input dimension for the Unet3D"
        # self.drop_ratio = 0                                                        # "Probability to drop a cropped area if the label is empty. All empty patches will be dropped for 0 and accept all cropped patches if set to 1"
        # self.min_pixel = 0.1                                                       # "Percentage of minimum non-zero pixels in the cropped label"
        self.batch_size = 1                                                         # 'Batch size: the greater more GPUs are used and faster is the training'
        self.num_works = 1
        self.num_epoch = 200                                                         # "Number of epochs"
        self.init_epoch = 1
        self.stride_inplane = 64                                                     # "Stride size in 2D plane"
        self.stride_layer = 16                                                       # "Stride size in z direction"

        # path setting
        self.data_path = './data/raw_dataset2/train'                           # Training data folder
        self.val_path = './data/raw_dataset2/val'                       # Validation data folder
        self.test_path = './data/raw_dataset2/test'                            # Testing data folder
        self.history_dir = './History/20220718'
        self.output_path = "./History/20220718"

class MyParser(object):
    def __init__(self):
        self.do_you_wanna_train = False
        self.do_you_wanna_load_weights = True
        self.do_you_wanna_check_accuracy = True

        self.multi_gpu = False
        self.gpu_id = 0
        # optimizer setting
        self.lr = 0.001
        self.weight_decay = 0.001
        # train setting
        self.increase_factor_data = 1
        self.resize = True
        self.new_size = [128, 128, 8]
        self.batch_size = 50
        self.num_works = 10
        self.num_epoch = 300
        self.init_epoch = 1

        self.data_path = './Data_folder/train_set/'
        self.val_path = './Data_folder/validation_set/'
        self.test_path = './Data_folder/test_set/'
        self.label_path = './Data_folder/FS-NPC-NUM-PFS-LEVEL.xlsx'
        self.history_dir = './History/20210408'
        self.output_path = self.history_dir

