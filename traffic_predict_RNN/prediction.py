# import traci
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

edge_list = []
queue_7to8_list = []
queue_8to7_list = []
step_list = []
phase_list7 = []
phase_list8 = []

data_frame = pd.read_csv("../sumo/7to8_data.csv", index_col=0)

scaler = MinMaxScaler()
data_frame[['phase_7']] = scaler.fit_transform(data_frame[['phase_7']])
data_frame[['phase_8']] = scaler.fit_transform(data_frame[['phase_8']])
data_frame[['7to8']] = scaler.fit_transform(data_frame[['7to8']])
data_frame[['8to7']] = scaler.fit_transform(data_frame[['8to7']])
data_frame[['7in']] = scaler.fit_transform(data_frame[['7in']])
data_frame[['8in']] = scaler.fit_transform(data_frame[['8in']])

scaler_data_frame_phase_7 = scaler.inverse_transform(data_frame[['phase_7']])
scaler_data_frame_phase_8 = scaler.inverse_transform(data_frame[['phase_8']])
scaler_data_frame_7to8 = scaler.inverse_transform(data_frame[['7to8']])
scaler_data_frame_8to7 = scaler.inverse_transform(data_frame[['8to7']])
scaler_data_frame_7in = scaler.inverse_transform(data_frame[['7in']])
scaler_data_frame_8in = scaler.inverse_transform(data_frame[['8in']])


# x1 = data_frame[['step', 'phase_7', 'phase_8', '7to8', '7in', '8in']].values
# x2 = data_frame[['step', 'phase_7', 'phase_8', '8to7', '7in', '8in']].values
x1 = data_frame[['phase_7', 'phase_8', '7to8', '7in', '8in']].values
x2 = data_frame[['phase_7', 'phase_8', '8to7', '7in', '8in']].values
# print(x1)
y1 = data_frame['7to8'].values
y2 = data_frame['8to7'].values

# print(scaler_data_frame_7to8)

sequence_length = 5
num_layers = 2
hidden_size = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 20

def seq_data(x, y, sequence_length):
    x_seq = []
    y_seq = []
    
    for i in range(len(y)-sequence_length):
        x_seq.append(x[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])

    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device).view([-1,1])

x1_seq, y1_seq = seq_data(x1, y1, sequence_length)
x2_seq, y2_seq = seq_data(x2, y2, sequence_length)
input_size = x1_seq.size(2)
input_size = x2_seq.size(2)

train1 = torch.utils.data.TensorDataset(x1_seq, y1_seq)
train2 = torch.utils.data.TensorDataset(x2_seq, y2_seq)
train_loader1 = torch.utils.data.DataLoader(dataset=train1, batch_size=batch_size, shuffle=False)
train_loader2= torch.utils.data.DataLoader(dataset=train2, batch_size=batch_size, shuffle=False)

class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
        super(VanillaRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc = nn.Sequential(nn.Linear(hidden_size * sequence_length,1),nn.Sigmoid())

    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        # print(h0)
        # print(x)
        out, _ = self.rnn(x, h0)
        # print(out)
        # print(out.shape)
        out = out.reshape(out.shape[0],-1)
        # print(out)
        # out = self.fc(out)
        return out


model = VanillaRNN(input_size=input_size, hidden_size=hidden_size, sequence_length=sequence_length, num_layers=num_layers, device=device)

criterion = nn.MSELoss()
learning_rate = 0.00007
num_epochs = 10000
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

loss_graph = []
loss_graph2 = []
output_7to8 = []
output_8to7 = []
n = len(train_loader1)
# print(n)

for epoch in range(num_epochs):
    running_loss = 0.0
    running_loss2 = 0.0
    for data in train_loader1:
        seq, target  = data    
        out = model(seq)
        loss = criterion(out, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        loss_graph.append(running_loss/n)
    
    for data in train_loader2:
        seq2, target2 = data
        out2 = model(seq2)
        loss2 = criterion(out2, target2)
        
        optimizer.zero_grad()
        # loss.backward()
        optimizer.step()
        running_loss2 += loss2.item()
        loss_graph2.append(running_loss2/n)


    if epoch % 1000 == 0:
        # print('epoch: %d, loss_7to8: %.4f, loss_8to7: %.4f'%(epoch, running_loss/n, running_loss2/n))
        print('epoch: %d, loss_7to8: %.4f'%(epoch, running_loss/n))
        # print(out)
        # learning_rate *= 0.8
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)

for i in range(sequence_length):
    output_7to8.append(0)


for input in range(len(x1_seq)):
    x1_test = x1_seq[input].view(1, 5, 5)
    # print(x1_test)
    # print(model(x1_test))
    # x2_test = x2_seq[input].view(1, 5, 6)
    model_num = pd.DataFrame(model(x1_test))
    output_7to8.append(scaler.inverse_transform(model_num)[0][-1])
# print(output_7to8)


plt.plot(scaler.inverse_transform(data_frame[['7to8']]))
# plt.plot(y2)
plt.plot(output_7to8)
# plt.plot(output_8to7)
plt.show()

accuracy_sum1 = 0
accuracy_sum2 = 0
sub1= []
sub2 = []
model_accuracy_sum1 = 0
# print('length : ',len(output_7to8)) 
# print(output_7to8)

for i in range(len(output_7to8)):
    if i == len(output_7to8) - 1:
        break
    # print(output_7to8[-1])
    accuracy1 = abs((output_7to8[i] - output_7to8[i+1])) 
    model_accuracy1 = abs((output_7to8[i] - scaler_data_frame_7to8[i])) 

    # accuracy2 = abs((output_8to7[i] - output_8to7[i+1])) / max(output_8to7) * 100
    accuracy_sum1 += accuracy1
    model_accuracy_sum1 += model_accuracy1
    # accuracy_sum2 += accuracy2

    sub1.append(output_7to8[i] - output_7to8[i+1])
    # sub2.append(output_8to7[i] - output_8to7[i+1])
print(accuracy_sum1 / len(output_7to8))
print(max(sub1))
print('model accuracy : ', model_accuracy_sum1 / len(output_7to8))
# print(accuracy_sum2 / len(output_8to7))
# print(max(sub2))

