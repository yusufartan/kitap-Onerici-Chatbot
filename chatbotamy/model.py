import torch
import torch.nn as nn

# Sinir ağı modeli sınıfı tanımlanıyor
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # Tam bağlı (fully connected) katmanlar oluşturuluyor
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        # ReLU aktivasyon fonksiyonu tanımlanıyor
        self.relu = nn.ReLU()
    
    # İleri geçiş fonksiyonu tanımlanıyor
    def forward(self, x):
        # Giriş verisi ilk gizli katmana uygulanıyor
        out = self.l1(x)
        out = self.relu(out)  # ReLU aktivasyon fonksiyonu uygulanıyor
        out = self.l2(out)
        out = self.relu(out)  # ReLU aktivasyon fonksiyonu uygulanıyor
        out = self.l3(out)
        # Son çıkış dönüştürülmeden döndürülüyor (aktivasyon veya softmax yok)
        return out
