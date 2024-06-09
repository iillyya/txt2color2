class Args():
    def __init__(self, args):
        self.hidden_size = args['hidden_size']
        self.n_layers = args['n_layers']
        self.always_give_global_hint = args['always_give_global_hint']
        self.add_L = args['add_L']
        self.mode = args['mode']
        self.dataset = args['dataset']
        self.lr = args['lr']
        self.num_epochs = args['num_epochs']
        self.resume_epoch = args['resume_epoch']
        self.batch_size = args['batch_size']
        self.dropout_p = args['dropout_p']
        self.weight_decay = args['weight_decay']
        self.beta1 = args['beta1']
        self.beta2 = args['beta2']
        self.lambda_sL1 = args['lambda_sL1']
        self.lambda_KL = args['lambda_KL']
        self.lambda_GAN = args['lambda_GAN']
        self.text2pal_dir = args['text2pal_dir']
        self.pal2color_dir = args['pal2color_dir']
        self.train_sample_dir = args['train_sample_dir']
        self.log_interval = args['log_interval']
        self.sample_interval = args['sample_interval']
        self.save_interval = args['save_interval']
        self.test_sample_dir = args['test_sample_dir']
