import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv('../sumo/7to8_data.csv', index_col=0)

x = df[['phase_7', 'phase_8', '7to8','7in', '8in']].values
y = df[['7to8']].values

scaler = MinMaxScaler()

scaler_x = scaler.fit_transform(x)
scaler_y = scaler.fit_transform(y)
# print(scaler_x)

sequence_length = 5
num_layers = 5
hidden_size = 1
num_classes = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 20

def seq_data(x, y, sequence_length):
    x_seq = []
    y_seq = []

    for i in range(len(y) - sequence_length):
        x_seq.append(x[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])

    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device).view([-1,1])

x_seq, y_seq = seq_data(scaler_x, scaler_y, sequence_length) 
# print(x_seq)
# print('-----------')
# print(y_seq)
input_size = x_seq.size(2)

train = torch.utils.data.TensorDataset(x_seq, y_seq)
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, sequence_length, device):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.device = device
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
    
        out, (h0, c0) = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        return out

model = LSTM(num_classes=num_classes, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, sequence_length=sequence_length, device=device)

loss_list = []
output = []

learning_rate = 0.0001
num_epochs = 10000
n = len(train_loader)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    running_loss = 0.0

    for data in train_loader:
        seq, target = data
        out = model(seq)
        loss = criterion(out, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        loss_list.append(running_loss/n)

    if epoch % 1000 == 0:
        print('epoch : %d , loss : %.4f'%(epoch, running_loss/n))

for i in range(len(x_seq)):
    x_test = x_seq[i].view(1,5,5)
    model_numpy = pd.DataFrame(model(x_test))
    output.append(scaler.inverse_transform(model_numpy)[0][-1])

error_sum = 0
model_error_sum = 0

for i in range(len(output)):
    if i == len(output)-1:
        break
    error = abs(output[i] - output[i+1])
    model_error = abs(output[i] - scaler_y[i])

    error_sum += error
    model_error_sum += model_error
    
print(error_sum / len(output))
print(model_error_sum / len(output))

plt.plot(y)
plt.plot(output)
plt.show()