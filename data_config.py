
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            # self.root_dir = 'path to the root of LEVIR-CD dataset'
            # self.root_dir = './LEVIR-CD/'
            self.root_dir = '../../Datasets/LEVIR-CD-256-v1/'
        elif data_name == 'quick_start':
            self.root_dir = './samples/'
        elif data_name == 'DSIFN':
            self.root_dir = '../../Datasets/DSIFN-CD-256/' 
        elif data_name == 'INRIA':
            self.root_dir = '../../Datasets/INRIA256/'     
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

