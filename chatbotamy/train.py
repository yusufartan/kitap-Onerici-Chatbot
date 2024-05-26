import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem  # nltk_utils dosyasından gerekli fonksiyonları ve modülleri alıyoruz
from model import NeuralNet  # model dosyasından yapay sinir ağı modelini alıyoruz

# intents.json dosyasını okuyarak intents adlı bir değişkene yüklüyoruz
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []  # tüm kelimeleri içerecek bir liste oluşturuyoruz
tags = []  # etiketleri içerecek bir liste oluşturuyoruz
xy = []  # modelin eğitimi için kullanılacak kelime ve etiket çiftlerini içerecek bir liste oluşturuyoruz

# intents içindeki her bir pattern için işlem yapacağız
for intent in intents['intents']:
    tag = intent['tag']  # pattern'in etiketini alıyoruz
    tags.append(tag)  # etiketi etiket listesine ekliyoruz
    for pattern in intent['patterns']:
        # pattern'daki her bir kelimeyi tokenize ediyoruz
        w = tokenize(pattern)
        all_words.extend(w)  # tüm kelimeleri birleştiriyoruz
        xy.append((w, tag))  # kelime-etiket çiftini xy listesine ekliyoruz

ignore_words = ['?', '.', '!']  # göz ardı edilecek karakterler
all_words = [stem(w) for w in all_words if w not in ignore_words]  # kelimeleri köklerine ayırıyoruz
all_words = sorted(set(all_words))  # tekrar eden kelimeleri kaldırıyoruz ve sıralıyoruz
tags = sorted(set(tags))  # tekrar eden etiketleri kaldırıyoruz ve sıralıyoruz

# Eğitim verisi oluşturuyoruz
X_train = []  # eğitim verisi
y_train = []  # etiketler
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)  # kelime torbası oluşturuyoruz
    X_train.append(bag)  # eğitim verisine ekliyoruz
    label = tags.index(tag)  # etiketleri sayısal değerlere dönüştürüyoruz
    y_train.append(label)  # etiket listesine ekliyoruz

X_train = np.array(X_train)  # Numpy dizisine dönüştürüyoruz
y_train = np.array(y_train)  # Numpy dizisine dönüştürüyoruz

# Hyper-parametreler
num_epochs = 500  # eğitim döngüsü sayısı
batch_size = 64  # her adımda kullanılacak veri miktarı
learning_rate = 0.001  # öğrenme katsayısı
input_size = len(X_train[0])  # girdi boyutu
hidden_size = 16  # gizli katman boyutu
output_size = len(tags)  # çıkış boyutu

# PyTorch'un veri kümesi sınıfını kullanarak veri kümesini oluşturuyoruz
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

dataset = ChatDataset()  # veri kümesi örneği oluşturuyoruz
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # CUDA'nın kullanılabilir olup olmadığını kontrol ediyoruz

model = NeuralNet(input_size, hidden_size, output_size).to(device)  # modeli tanımlıyoruz ve CUDA'ya gönderiyoruz (varsa)

# Loss ve optimizer
criterion = nn.CrossEntropyLoss()  # kayıp fonksiyonu
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # optimize edici

# Modeli eğitiyoruz
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # İleri yayılım
        outputs = model(words)
        loss = criterion(outputs, labels)  # kaybı hesaplıyoruz
        
        # Geri yayılım ve optimizasyon
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

# Modelin durumunu ve diğer gerekli bilgileri kaydediyoruz
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"  # kaydedilecek dosya adı
torch.save(data, FILE)  # modeli ve diğer bilgileri kaydediyoruz

print(f'Eğitim tamamlandı. Dosya {FILE} olarak kaydedildi.')
