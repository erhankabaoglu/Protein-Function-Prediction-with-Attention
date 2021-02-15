import torch
import torch.nn as nn
from torch.nn import functional as F


class PFPModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, layer_size, drop_out=0.5):
        super(PFPModel, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.layer_size = layer_size
        self.drop_out = drop_out

        self.bilstm = nn.LSTM(vocab_size, hidden_size, layer_size, dropout=drop_out, bidirectional=True)
        # We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
        self.W_s1 = nn.Linear(2 * hidden_size, 350)
        self.W_s2 = nn.Linear(350, 30)
        self.fc_layer = nn.Linear(30 * 2 * hidden_size, 2000)
        self.label = nn.Linear(2000, output_size)

    def attention_net(self, lstm_output):

        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, input_sentences):

        output, (h_n, c_n) = self.bilstm(input_sentences)
        output = output.permute(1, 0, 2)

        attn_weight_matrix = self.attention_net(output)

        hidden_matrix = torch.bmm(attn_weight_matrix, output)

        fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2]))
        logits = self.label(fc_out)

        return logits
