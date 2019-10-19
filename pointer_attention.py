import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import random

import time

# If we have a GPU available, we'll set our device to GPU.
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

def data_generator(size=20000, min_length=10, max_lenght=50, element_bound=1e+5):
    data = []
    targets = []
    labels = []
    lengths = []
    max_lenght = 0
    for _ in range(size):
        length = random.randint(min_length, max_lenght)
        if length > max_lenght:
            max_lenght = length
        sequence = random.sample(range(element_bound), length)
        target = sorted(sequence)
        label = []
        data.append(sequence)
        targets.append(target)
        for i in range(length):
            label.append(data.index(target[i]))

        labels.append(label)
        lengths.append(length)

    data = sorted(data, key=len, reverse=True)
    labels = sorted(labels, key=len, reverse=True)
    targets = sorted(targets, key=len, reverse=True)
    lengths = sorted(lengths, reverse=True)

    # Pad data with 0
    for seq in data:
        if len(seq) < max_lenght:
            seq.extend([0]*(max_lenght-len(seq)))

    return torch.Tensor(data).to(device), targets, labels, lengths.to(device)

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True )

    def forward(self, input, hidden, lengths):
        input = nn.utils.rnn.pack_padded_sequence(input, lengths)
        output, hidden = self.gru(input, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, padding_value=0)
        # output = torch.Tensor(output)
        # output.view(output.size(0), self.batch_size, 2, self.hidden_size)
        return output, hidden

    def initHidden(self):
        return torch.zeros(2, self.batch_size, self.hidden_size, device=device)

# A sequence is a batch for this attendtion layer
class Pointer_Attention(nn.Module):
    def __init__(self, encoder_context_size, output_context_size):
        super(Pointer_Attention, self).__init__()
        self.encoder_context_size = encoder_context_size
        self.output_context_size = output_context_size
        self.out = nn.Linear(encoder_context_size*2 + output_context_size, 1)
        # self.softmax = nn.LogSoftmax(dim=0)
    # input is a sequence with shape (length * encoder_context_size+output_context_size)
    def forward(self, encoder_context, output_context):
        output_context = output_context.repeat(encoder_context.size(0),1)
        full_context = torch.cat([encoder_context,output_context], dim=1)
        output = self.out(full_context)
        # return self.softmax(output)
        return output

class Output_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Output_Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.gru = nn.GRU(input_size, hidden_size)

    # def forward(self, input, hidden, lengths):
    #     input = nn.utils.rnn.pack_padded_sequence(input, lengths)
    #     output, hidden = self.gru(input, hidden)
    #     output, _ = nn.utils.rnn.pad_packed_sequence(output, padding_value=0)
    #     output.view(output.size(0), self.batch_size, self.hidden_size)
    #     return output, hidden

    def forward(self, input, hidden):
        # input = nn.utils.rnn.pack_padded_sequence(input, lengths)
        output, hidden = self.gru(input, hidden)
        # output, _ = nn.utils.rnn.pad_packed_sequence(output, padding_value=0)
        # output.view(output.size(0), self.batch_size, self.hidden_size)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=device)

def train(epoch_num=100, data_size=20000, batch_size=10):
    # epoch_num = 100
    # data_size = 20000
    # batch_size = 10
    data, targets, labels, lengths = data_generator(data_size)
    input_size = 1 
    hidden_size = 20

    encoder = Encoder(input_size, hidden_size, batch_size).to(device)
    pointer_attention = Pointer_Attention(hidden_size, hidden_size).to(device)
    output_encoder = Output_Encoder(input_size, hidden_size, batch_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(pointer_attention.parameters()) + list(output_encoder.parameters()), lr=5e-4)

    batch_num = data_size//batch_size
    losses = []
    for epoch in range(epoch_num):
        epoch_loss = []
        for i in range(batch_num):
            batch_loss =[]
            batch_length = lengths[i*batch_size : i*batch_size+batch_size]
            max_length = batch_length[0]

            batch = data[i*batch_size : i*batch_size+batch_size, :max_length, :] # batch_size * length * hidden_size
            batch = torch.transpose(batch,0,1) # length * batch_size * hidden_size
            batch_target = targets[i*batch_size : i*batch_size+batch_size]
            batch_label = labels[i*batch_size : i*batch_size+batch_size]

            pred_label = torch.zeros(max_length, batch_size)
            preditions = torch.zeros(max_length, batch_size)

            hidden = encoder.initHidden()
            output_hidden = output_encoder.initHidden()

            encoded_context, hidden = encoder.forward(batch, hidden, batch_length) # length * batch_size * (2 * hidden_size)

            for t in range(max_length):

                output_context, output_hidden = output_encoder.forward(preditions, output_hidden) # length * batch_size * hidden_size
                output_context = output_context[-1] # batch_size * hidden_size

                for b in range(batch_size):
                    if batch_length[b] <= t:
                        continue
                    sequence_context = encoded_context[:,b,:] # length * (2 * hidden_size)
                    output_c = output_context[b,:] # hidden_size
                    attention_weight = pointer_attention.forward(sequence_context, output_c) # length * 1
                    _, predition = torch.max(attention_weight)
                    pred_label[t,b] = predition
                    preditions[t,b] = batch[t,b]
                    loss = criterion([attention_weight], [batch_label[b,t]])
                    loss.backward()
                    batch_loss.append(loss.values())
            
            epoch_loss.append(np.mean(batch_loss))

            optimizer.step()
            optimizer.zero_grad()
        mean_loss = np.mean(epoch_loss)
        losses.append(mean_loss)
        print(time.strftime("%H:%M:%S", time.localtime()) + 'epoch {} finished, with mean loss: {}; loss for each batch: {}'.format(epoch, mean_loss, epoch_loss))
        if epoch % 20 == 0:
            torch.save(encoder.state_dict(), './encoder_{}_epoch'.format(epoch))
            torch.save(pointer_attention.state_dict(), './pointer_attention_{}_epoch'.format(epoch))
            torch.save(output_encoder.state_dict(), './output_encoder_{}_epoch'.format(epoch))

    torch.save(encoder.state_dict(), './encoder_{}_epoch'.format(epoch_num-1))
    torch.save(pointer_attention.state_dict(), './pointer_attention_{}_epoch'.format(epoch_num-1))
    torch.save(output_encoder.state_dict(), './output_encoder_{}_epoch'.format(epoch_num-1))
    return losses

if __name__ == '__main__':
    losses = train(epoch_num=100, data_size=20000, batch_size=10)

    




            



