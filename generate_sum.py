import logging
import argparse
import os

import torch
import torch.nn as nn

from data_utils import get_minibatch, read_config, hyperparam_string, read_summarization_data
from model import Seq2SeqAttentionSharedEmbedding

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config",
    required=True
)
args = parser.parse_args()
config_file_path = args.config
config = read_config(config_file_path)
experiment_name = hyperparam_string(config)
save_dir = config['data']['save_dir']
load_dir = config['data']['load_dir']
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/%s' % (experiment_name),
    filemode='w'
)

# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


print 'Reading data ...'

src, trg = read_summarization_data(
    src=config['data']['src'],
    trg=config['data']['trg']
)

src_test, trg_test = read_summarization_data(
    src=config['data']['test_src'],
    trg=config['data']['test_trg']
)

output_path = config['data']['output']

batch_size = config['data']['batch_size']
max_length = config['data']['max_src_length']
vocab_size = len(src['word2id'])

logging.info('Model Parameters : ')
logging.info('Task : %s ' % (config['data']['task']))
logging.info('Model : %s ' % (config['model']['seq2seq']))
logging.info('Language : %s ' % (config['model']['src_lang']))
logging.info('Embedding Dim  : %s' % (config['model']['dim_word_src']))
logging.info('Source RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Target RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Source RNN Depth : %d ' % (config['model']['n_layers_src']))
logging.info('Target RNN Depth : %d ' % (1))
logging.info('Source RNN Bidirectional  : %s' % (config['model']['bidirectional']))
logging.info('Batch Size : %d ' % (config['model']['n_layers_trg']))
logging.info('Optimizer : %s ' % (config['training']['optimizer']))
logging.info('Learning Rate : %f ' % (config['training']['lrate']))

logging.info('Found %d words ' % (vocab_size))

weight_mask = torch.ones(vocab_size).cuda()
weight_mask[trg['word2id']['<pad>']] = 0
loss_criterion = nn.CrossEntropyLoss(weight=weight_mask).cuda()

model = Seq2SeqAttentionSharedEmbedding(
    emb_dim=config['model']['dim_word_src'],
    vocab_size=vocab_size,
    src_hidden_dim=config['model']['dim'],
    trg_hidden_dim=config['model']['dim'],
    ctx_hidden_dim=config['model']['dim'],
    attention_mode='dot',
    batch_size=batch_size,
    bidirectional=config['model']['bidirectional'],
    pad_token_src=src['word2id']['<pad>'],
    pad_token_trg=trg['word2id']['<pad>'],
    nlayers=config['model']['n_layers_src'],
    nlayers_trg=config['model']['n_layers_trg'],
    dropout=0.,
).cuda()

model_path = os.path.join(load_dir, "epoch_0.model")
if os.path.exists(model_path):
    model.load_state_dict(torch.load(open(model_path)))

test_trg_h = open(output_path, 'w')
test_sys_out = open('sumdata/Giga/systems/task1_ref0.txt', 'w')

for j in xrange(0, len(src_test['data']), batch_size):
    input_lines_src, _, lens_src, mask_src = get_minibatch(
        src_test['data'], src_test['word2id'], j,
        batch_size, max_length, add_start=True, add_end=True
    )
    input_lines_trg, output_lines_trg, lens_trg, mask_trg = get_minibatch(
        trg_test['data'], trg_test['word2id'], j,
        batch_size, max_length, add_start=True, add_end=True
    )
    decoder_logit = model(input_lines_src, input_lines_trg)

    word_probs = model.decode(
        decoder_logit
    ).data.cpu().numpy().argmax(axis=-1)

    output_lines_trg = output_lines_trg.data.cpu().numpy()

    for sentence_pred, sentence_real in zip(word_probs, output_lines_trg):
        sentence_pred = [trg['id2word'][x] for x in sentence_pred]
        sentence_real = [trg['id2word'][x] for x in sentence_real]
        if '</s>' in sentence_pred:
            index = sentence_pred.index('</s>')
            sentence_pred = sentence_pred[:index]
        if '</s>' in sentence_real:
            index = sentence_real.index('</s>')
            sentence_real = sentence_real[:index]
        test_trg_h.write(' '.join(sentence_pred) + '\n')
        test_sys_out.write(' '.join(sentence_real) + '\n')

test_trg_h.close()
test_sys_out.close()
