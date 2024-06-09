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

        if mode == 'train_TPN':
            # Data loader.
            self.input_dict = self.prepare_dict()
            self.train_loader, _ = t2p_loader(self.args.batch_size, self.input_dict)

            # Load pre-trained GloVe embeddings.
            emb_file = os.path.join('СЮДА ВСТАВЬ ПУТЬ НА ФАЙЛ Color-Hex-vf.pth')
            if os.path.isfile(emb_file):
                W_emb = torch.load(emb_file)
            else:
                W_emb = load_pretrained_embedding(self.input_dict.word2index,
                                                  'СЮДА ВСТАВЬ ПУТЬ НА ФАЙЛ glove.840B.300d.txt',
                                                  300)
                W_emb = torch.from_numpy(W_emb)
                torch.save(W_emb, emb_file)
            W_emb = W_emb.to(self.device)

            # Generator and discriminator.
            self.encoder = EncoderRNN(self.input_dict.n_words, self.args.hidden_size,
                                      self.args.n_layers, self.args.dropout_p, W_emb).to(self.device)
            self.G = AttnDecoderRNN(self.input_dict, self.args.hidden_size,
                                    self.args.n_layers, self.args.dropout_p).to(self.device)
            self.D = Discriminator(15, self.args.hidden_size).to(self.device)

            # Initialize weights.
            self.encoder.apply(init_weights_normal)
            self.G.apply(init_weights_normal)
            self.D.apply(init_weights_normal)

            # Optimizer.
            self.G_parameters = list(self.encoder.parameters()) + list(self.G.parameters())
            self.g_optimizer = torch.optim.Adam(self.G_parameters,
                                                lr=self.args.lr, weight_decay=self.args.weight_decay)
            self.d_optimizer = torch.optim.Adam(self.D.parameters(),
                                                lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))

        elif mode == 'test_TPN' or 'test_text2colors' or 'sample_TPN':
            
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
        if mode == 'train_TPN':
            encoder_path = os.path.join(self.args.text2pal_dir, '{}_G_encoder.ckpt'.format(resume_epoch))
            G_path = os.path.join(self.args.text2pal_dir, '{}_G_decoder.ckpt'.format(resume_epoch))
            D_path = os.path.join(self.args.text2pal_dir, '{}_D.ckpt'.format(resume_epoch))
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
            self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
            self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
            
        elif mode == 'test_TPN' or 'sample_TPN':
            encoder_path = os.path.join(self.args.text2pal_dir, '{}_G_encoder.ckpt'.format(resume_epoch))
            G_TPN_path = os.path.join(self.args.text2pal_dir, '{}_G_decoder.ckpt'.format(resume_epoch))
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
            self.G_TPN.load_state_dict(torch.load(G_TPN_path, map_location=lambda storage, loc: storage))

        elif mode == 'test_text2colors':
            encoder_path = os.path.join(self.args.text2pal_dir, '{}_G_encoder.ckpt'.format(resume_epoch))
            G_TPN_path = os.path.join(self.args.text2pal_dir, '{}_G_decoder.ckpt'.format(resume_epoch))
            G_PCN_path = os.path.join(self.args.pal2color_dir, '{}_G.ckpt'.format(resume_epoch))
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=lambda storage, loc: storage))
            self.G_TPN.load_state_dict(torch.load(G_TPN_path, map_location=lambda storage, loc: storage))
            self.G_PCN.load_state_dict(torch.load(G_PCN_path, map_location=lambda storage, loc: storage))


    



    def test_TPN(self):

        for batch_idx, (txt_embeddings, real_palettes, _) in enumerate(self.test_loader):
            if txt_embeddings.size(0) != self.args.batch_size:
                break

            # Compute text input size (without zero padding).
            batch_size = txt_embeddings.size(0)
            nonzero_indices = list(torch.nonzero(txt_embeddings)[:, 0])
            each_input_size = [nonzero_indices.count(j) for j in range(batch_size)]

            # Prepare test data.
            txt_embeddings = txt_embeddings.to(self.device)
            real_palettes = real_palettes.to(self.device).float()

            # Generate multiple palettes from same text input.
            for num_gen in range(10):

                # Prepare input and output variables.
                palette = torch.FloatTensor(batch_size, 3).zero_().to(self.device)
                fake_palettes = torch.FloatTensor(batch_size, 15).zero_().to(self.device)

                # ============================== Text-to-Palette ==============================#
                # Condition for the generator.
                encoder_hidden = self.encoder.init_hidden(batch_size).to(self.device)
                #encoder_outputs, decoder_hidden, mu, logvar = self.encoder(txt_embeddings, encoder_hidden)
                encoder_outputs, decoder_hidden, mu, logvar = self.encoder(txt_embeddings.to(self.device), encoder_hidden.to(self.device))

                # Generate color palette.
                for i in range(5):
                    palette, decoder_context, decoder_hidden, _ = self.G_TPN(palette,
                                                                             decoder_hidden.squeeze(0),
                                                                             encoder_outputs,
                                                                             each_input_size,
                                                                             i)
                    fake_palettes[:, 3 * i:3 * (i + 1)] = palette

                # ================================ Save Results ================================#
                for x in range(self.args.batch_size):
                    # Input text.
                    input_text = ''
                    for idx in txt_embeddings[x]:
                        if idx.item() == 0: break
                        input_text += self.input_dict.index2word[idx.item()] + ' '

                    # Save palette generation results.
                    fig1, axs1 = plt.subplots(nrows=2, ncols=5)
                    axs1[0][0].set_title(input_text + 'fake {}'.format(num_gen + 1))
                    for k in range(5):
                        lab = np.array([fake_palettes.data[x][3 * k].cpu().numpy(),
                                        fake_palettes.data[x][3 * k + 1].cpu().numpy(),
                                        fake_palettes.data[x][3 * k + 2].cpu().numpy()], dtype='float64')
                        rgb = lab2rgb_1d(lab)
                        axs1[0][k].imshow([[rgb]])
                        axs1[0][k].axis('off')
                    axs1[1][0].set_title(input_text + 'real')
                    for k in range(5):
                        lab = np.array([real_palettes.data[x][3 * k].cpu().numpy(),
                                        real_palettes.data[x][3 * k + 1].cpu().numpy(),
                                        real_palettes.data[x][3 * k + 2].cpu().numpy()], dtype='float64')
                        rgb = lab2rgb_1d(lab)
                        axs1[1][k].imshow([[rgb]])
                        axs1[1][k].axis('off')

                    fig1.savefig(os.path.join(self.args.test_sample_dir, self.args.mode,
                                              '{}_palette{}.jpg'.format(self.args.batch_size*batch_idx+x+1,
                                                                        num_gen+1)))
                    



    def sample_TPN(self, queryStrings, numPalettesPerQuery=1):

        # ==================== Preprocessing src_seqs ====================#
        # Return a list of indexes, one for each word in the sentence.
        txt_embeddings = []
        for index, queryString in enumerate(queryStrings):
            # Set list size to the longest palette name.
            temp = [0] * self.input_dict.max_len
            for i, word in enumerate(queryString.split(' ')):
                temp[i] = self.input_dict.word2index[word]
            txt_embeddings.append(temp)

        # Convert to tensor
        txt_embeddings = torch.LongTensor(txt_embeddings).to(self.device)

        # ==== END ====#

        # Compute text input size (without zero padding).
        batch_size = txt_embeddings.size(0)
        nonzero_indices = list(torch.nonzero(txt_embeddings)[:, 0])
        each_input_size = [nonzero_indices.count(j) for j in range(batch_size)]

        # Placeholder for final palettes
        palettes = [{'queryString': q, 'samples': []} for q in queryStrings]

        # Generate multiple palettes from same text input.
        for num_gen in range(numPalettesPerQuery):

            # Prepare input and output variables.
            palette = torch.FloatTensor(batch_size, 3).zero_().to(self.device)
            fake_palettes = torch.FloatTensor(batch_size, 15).zero_().to(self.device)

            # ============================== Text-to-Palette ==============================#
            # Condition for the generator.
            encoder_hidden = self.encoder.init_hidden(batch_size).to(self.device)
            encoder_outputs, decoder_hidden, mu, logvar = self.encoder(txt_embeddings.to(self.device), encoder_hidden.to(self.device))

            # Generate color palette.
            for i in range(4):
                palette, _, decoder_hidden, _ = self.G_TPN(palette,
                                                                decoder_hidden.squeeze(0),
                                                                encoder_outputs,
                                                                each_input_size,
                                                                i)

                

        return palettes
