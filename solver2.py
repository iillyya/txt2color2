import torch
import torch.nn as nn
import os
import time
import numpy as np
import pickle
#import matplotlib.pyplot as plt
#plt.switch_backend('agg')
from skimage.color import lab2rgb
from utils import Dictionary
from model import CA_NET, EncoderRNN, AttnDecoderRNN, Attn, Discriminator

class Solver(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = args.device
        # Build the model.
        self.build_model(args.mode)

    def prepare_dict(self):
        input_dict = Dictionary()
        src_path = os.path.join('СЮДА ВСТАВЬ ССЫЛКУ НА ФАЙЛ all_names.pkl')
        with open(src_path, 'rb') as f:
            text_data = pickle.load(f)
            f.close()



        for i in range(len(text_data)):
            input_dict.index_elements(text_data[i])
        return input_dict



    def build_model(self, mode):
            
		# Data loader.
		self.input_dict = self.prepare_dict()

		# Load pre-trained GloVe embeddings.
		emb_file = os.path.join('ПУТЬ ДО ПАПКИ В КОТОРОЙ ЛЕЖИТ Color-Hex-vf.pth', 'Color-Hex-vf.pth')
		if os.path.isfile(emb_file):
		    W_emb = torch.load(emb_file)
		else:
		    W_emb = load_pretrained_embedding(self.input_dict.word2index,
		                                      'ПУТЬ ДО ФАЙЛА glove.840B.300d.txt',
		                                      300)
		    W_emb = torch.from_numpy(W_emb)
		    torch.save(W_emb, emb_file)
		W_emb = W_emb.to(self.device)

		# Data loader.
		self.test_loader, self.imsize = test_loader(self.args.dataset, self.args.batch_size, self.input_dict)

		# Load the trained generators.
		self.encoder = EncoderRNN(self.input_dict.n_words, self.args.hidden_size,
		                              self.args.n_layers, self.args.dropout_p, W_emb).to(self.device)
		self.G_TPN = AttnDecoderRNN(self.input_dict, self.args.hidden_size,
		                            self.args.n_layers, self.args.dropout_p).to(self.device)
    # Load model.
    if self.args.resume_epoch:
        self.load_model(self.args.mode, self.args.resume_epoch)

    def load_model(self, mode, resume_epoch):
        encoder_path = os.path.join(self.args.text2pal_dir, '{}_G_encoder.ckpt'.format(resume_epoch))
        G_TPN_path = os.path.join(self.args.text2pal_dir, '{}_G_decoder.ckpt'.format(resume_epoch))
        self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
        self.G_TPN.load_state_dict(torch.load(G_TPN_path, map_location=lambda storage, loc: storage))

