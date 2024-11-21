import torch
import h5py
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
import gc

class HDF5DataGenerator:
    def __init__(self, hdf5_file_iq, hdf5_file_ap, batch_size, shuffle=True, augment=False, indices=None):
        self.hdf5_file_iq = hdf5_file_iq
        self.hdf5_file_ap = hdf5_file_ap
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.data_iq = h5py.File(hdf5_file_iq, 'r')
        self.data_ap = h5py.File(hdf5_file_ap, 'r')
        self.num_samples = len(indices) if indices is not None else self.data_iq['X'].shape[0]
        self.indices = indices if indices is not None else np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(self.num_samples / float(self.batch_size)))

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_index >= self.num_samples:
            raise StopIteration
        
        start = self.current_index
        end = min(start + self.batch_size, self.num_samples)
        batch_indices = self.indices[start:end]
        sorted_indices = np.sort(batch_indices)
        X_iq_batch = self.data_iq['X'][sorted_indices]
        X_ap_batch = self.data_ap['X'][sorted_indices]
        # X_iq_batch = X_iq_batch.transpose(0, 2, 1)  # shape: (batch_size, channels, sequence_length)
        # X_ap_batch = X_ap_batch.transpose(0, 2, 1)
        y_batch = self.data_iq['Y'][sorted_indices]
        snr_batch = self.data_iq['Z'][sorted_indices]
        reorder_indices = np.argsort(batch_indices)
        X_iq_batch = torch.tensor(X_iq_batch[reorder_indices], dtype=torch.float32)
        X_ap_batch = torch.tensor(X_ap_batch[reorder_indices], dtype=torch.float32)
        y_batch = torch.tensor(y_batch[reorder_indices], dtype=torch.float32)
        snr_batch = torch.tensor(snr_batch[reorder_indices], dtype=torch.float32)
        X_iq_batch = self.normalize_data(X_iq_batch)
        X_ap_batch = self.normalize_data(X_ap_batch)
        
        self.current_index = end
    
        return X_iq_batch, X_ap_batch, y_batch, snr_batch

    def close(self):
        self.data_iq.close()
        self.data_ap.close()

    def normalize_data(self, data):
        data_min = data.min(dim=-1, keepdim=True)[0]
        data_max = data.max(dim=-1, keepdim=True)[0]
        normalized_data = (data - data_min) / (data_max - data_min + 1e-8)
        return normalized_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VGG19FeatureExtractor(nn.Module):
    def __init__(self, in_channels=2):
        super(VGG19FeatureExtractor, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Conv1d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Conv1d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Dropout(0.5),

            nn.Conv1d(192, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Dropout(0.7),

            nn.AdaptiveAvgPool1d(1)
        )
 
    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return x

class LSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = attn_weights * lstm_out
        context = torch.sum(context, dim=1)
        out = self.classifier(context)
        return out

class MultiModalNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiModalNet, self).__init__()
        self.vgg19_iq = VGG19FeatureExtractor(in_channels=2)
        self.vgg19_ap = VGG19FeatureExtractor(in_channels=2)
        self.lstm_attention = LSTMAttention(512, 128, 1, num_classes)

    def forward(self, x_iq, x_ap):
        x_iq = self.vgg19_iq(x_iq)
        x_ap = self.vgg19_ap(x_ap)
        x = torch.cat((x_iq, x_ap), dim=1)
        x = x.unsqueeze(1)
        x = self.lstm_attention(x)
        return x
    
model = MultiModalNet(num_classes=53).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Train the model
def train_model(num_epochs, train_generator):
    start_time = time.time()
    model.train()
    for epoch in range(num_epochs):
        for i, (inputs_iq, inputs_ap, labels, _) in enumerate(train_generator):
            inputs_iq, inputs_ap, labels = inputs_iq.to(device), inputs_ap.to(device), labels.to(device)
            labels = torch.argmax(labels, dim=1)
            outputs = model(inputs_iq, inputs_ap)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 500 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(train_generator)}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        print(f'Learning rate: {scheduler.get_last_lr()[0]}')

    end_time = time.time()
    print(f'Training completed in {end_time - start_time:.2f} seconds.')

# Evaluate the model
def evaluate_model(dataloader, dataset_name):
    start_time = time.time()
    model.eval()
    total = correct = 0
    snr_accuracy = {}
    with torch.no_grad():
        for inputs_iq, inputs_ap, labels, snrs in dataloader:
            inputs_iq, inputs_ap, labels = inputs_iq.to(device), inputs_ap.to(device), labels.to(device)
            labels = torch.argmax(labels, dim=1)
            outputs = model(inputs_iq, inputs_ap)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            for idx, snr in enumerate(snrs):
                snr_value = snr.item()
                if snr_value not in snr_accuracy:
                    snr_accuracy[snr_value] = {'correct': 0, 'total': 0}
                snr_accuracy[snr_value]['correct'] += (predicted[idx] == labels[idx]).item()
                snr_accuracy[snr_value]['total'] += 1

    overall_accuracy = correct / total
    print(f'{dataset_name} Accuracy: {overall_accuracy:.4f}')
    
    for snr_value in sorted(snr_accuracy.keys()):
        acc = snr_accuracy[snr_value]['correct'] / snr_accuracy[snr_value]['total']
        print(f'{dataset_name} Accuracy at SNR {snr_value}: {acc:.4f}')

    end_time = time.time()
    print(f'Evaluation completed in {end_time - start_time:.2f} seconds.')

def train_model(num_epochs, train_generator):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        with tqdm(total=len(train_generator), desc=f"Epoch {epoch+1}/{num_epochs}", unit='batch') as pbar:
            for inputs_iq, inputs_ap, labels, _ in train_generator:
                inputs_iq, inputs_ap, labels = inputs_iq.to(device), inputs_ap.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs_iq, inputs_ap)
                loss = criterion(outputs, torch.argmax(labels, dim=1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inputs_iq.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
                total += labels.size(0)
                pbar.set_postfix(loss=epoch_loss / total, accuracy=correct / total)
                pbar.update(1)
        
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / total:.4f}, Accuracy: {correct / total:.4f}')
        gc.collect()
        torch.cuda.empty_cache()

# File path
hdf5_file_iq = 'E:/sig53.hdf5'
hdf5_file_ap = 'E:/sig53_ap.hdf5'
batch_size = 32

f_tem = h5py.File(hdf5_file_iq, 'r')
total_samples = f_tem['X'].shape[0]
f_tem.close()
indices = np.arange(total_samples)

train_size = int(0.98 * total_samples)
test_size = total_samples - train_size
train_indices, test_indices = indices[:train_size], indices[train_size:]
train_generator = HDF5DataGenerator(hdf5_file_iq, hdf5_file_ap, batch_size, shuffle=True, augment=True, indices=train_indices)
test_generator = HDF5DataGenerator(hdf5_file_iq, hdf5_file_ap, batch_size, shuffle=False, indices=test_indices)

train_model(num_epochs=200, train_generator=train_generator)
train_generator.close()
evaluate_model(test_generator, 'Test Set')
test_generator.close()