import torch
import torch.nn as nn

class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(xLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )
        self.conv1 = nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size // 2, kernel_size=7, padding=3)
        self.skip_gate = nn.Sequential(
            nn.Linear(hidden_size + (hidden_size // 2) * 3, hidden_size),
            nn.Sigmoid()
        )
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))
        attention_weights = self.attention(lstm_out)
        attended_out = torch.sum(lstm_out * attention_weights, dim=1)

        lstm_out_transposed = lstm_out.transpose(1, 2)
        conv1_out = self.conv1(lstm_out_transposed)
        conv2_out = self.conv2(lstm_out_transposed)
        conv3_out = self.conv3(lstm_out_transposed)

        multi_scale = torch.cat([conv1_out, conv2_out, conv3_out], dim=1)
        multi_scale = multi_scale.transpose(1, 2)
        pooled = torch.max(multi_scale, dim=1)[0]

        combined = torch.cat([attended_out, pooled], dim=1)
        skip = self.skip_gate(combined)
        # Apply skip gate to modulate attended_out
        gated = attended_out * skip

        gated = self.batch_norm1(gated)
        out = self.dropout(gated)
        out = self.relu(self.fc1(torch.cat([out, attended_out], dim=1)))
        out = self.batch_norm2(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
