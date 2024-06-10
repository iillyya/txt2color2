from args import Args
from solver import Solver
import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

def get_ind(solver, src_seqs):
    input_dict = solver.input_dict
    
    

    words_index = []
    for index, palette_name in enumerate(src_seqs):
        temp = [0] * input_dict.max_len

        for i, word in enumerate(palette_name):
            word = word if word in input_dict.word2index else '<unk>'
            temp[i] = input_dict.word2index[word]
        words_index.append(temp)
    src_seqs = torch.LongTensor(words_index).to(device)
    
    return src_seqs

args_d = {
    'hidden_size':150,
    'n_layers':1,
    'always_give_global_hint':1,
    'add_L':1,
    'mode':'sample_TPN',
    'dataset':'bird256',
    'lr':5e-4,
    'num_epochs':1000,
    'resume_epoch':200,
    'batch_size':1,
    'dropout_p':0.2,
    'weight_decay':5e-5,
    'beta1':0.5,
    'beta2':0.99,
    'lambda_sL1':100.0,
    'lambda_KL':0.5,
    'lambda_GAN':0.1,
    'text2pal_dir':'ПУТЬ ДО ВЕСОВ checkpoints', 
    'train_sample_dir':'ПУТЬ ДО ФАЙЛА С ТРЕНИРОВОЧНОЙ ВЫБОРКОЙ',
    'test_sample_dir':'ПУТЬ ДО ФАЙЛА С ТЕСТОВОЙ ВЫБОРКОЙ',
    'log_interval':1,
    'sample_interval':20,
    'save_interval':50
}
args = Args(args_d)
solver = Solver(args)

device = torch.device('cpu')

    
    
@app.route('/generate_palette', methods=['GET'])
def generate_palette(solver, txt):
    solver.encoder.eval()
    solver.G_TPN.eval()
    
    encoder_hidden = solver.encoder.init_hidden(batch_size=1).to(device)
    palette = torch.FloatTensor(1, 3).zero_().to(device)
    input_size = len(txt[0])

    encoder_outputs, decoder_hidden, mu, logvar = solver.encoder(get_ind(solver, txt), encoder_hidden)
    
    with torch.no_grad():
        colors = []

        for i in range(4):
	    palette, _ , decoder_hidden, _ = solver.G_TPN(
	        palette, decoder_hidden.squeeze(0), encoder_outputs, input_size, i
	    )
	    palette_np = palette.cpu().detach().numpy()
	    palette_rgb = lab2rgb(palette.cpu().detach().numpy()) * 255.0
	    colors.append(palette_rgb)
    
    return jsonify(colors)
    
    
if __name__ == '__main__':
    app.run(debug=True)


