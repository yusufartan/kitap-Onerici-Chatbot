import random  # Rastgele cevaplar için gereken kütüphane
import json  # JSON dosyasını işlemek için gereken kütüphane
import mysql.connector  # MySQL veritabanına bağlanmak için gereken kütüphane

import torch  # PyTorch kütüphanesi
from model import NeuralNet  # Önceden tanımlanan modeli içeri aktarıyoruz
from nltk_utils import bag_of_words, tokenize  # Bag of words oluşturmak ve tokenize işlemi için gerekli fonksiyonlar

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # CUDA'nın kullanılabilir olup olmadığını kontrol ediyoruz
INSERT_SQL = "INSERT INTO conversations (epostaId, birim, tag, message, response) VALUES (%s, %s, %s, %s, %s)"  # SQL sorgusu

with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)  # intents.json dosyasını yüklüyoruz

FILE = "data.pth"  # Eğitilmiş modelin kaydedildiği dosyanın adı
data = torch.load(FILE)  # Eğitilmiş modeli yüklüyoruz

input_size = data["input_size"]  # Giriş boyutu
hidden_size = data["hidden_size"]  # Gizli katman boyutu
output_size = data["output_size"]  # Çıkış boyutu
all_words = data['all_words']  # Tüm kelimeler
tags = data['tags']  # Etiketler
model_state = data["model_state"]  # Modelin durumu

model = NeuralNet(input_size, hidden_size, output_size).to(device)  # Modeli tanımlıyoruz ve CUDA'ya gönderiyoruz (varsa)
model.load_state_dict(model_state)  # Modelin durumunu yüklüyoruz
model.eval()  # Modeli değerlendirme modunda ayarlıyoruz

# MySQL veritabanı bağlantısı oluşturuyoruz ve tabloyu oluşturuyoruz (varsa)
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="chatbotamy")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS conversations
             (id INT AUTO_INCREMENT PRIMARY KEY,
             epostaId VARCHAR(255),
             birim VARCHAR(255),
             tag VARCHAR(255),
             message TEXT,
             response TEXT);''')
conn.commit()
conn.close()

bot_name = "Amy"  # Chatbot'un adı

# Rastgele cevap almak için bir fonksiyon tanımlıyoruz
def get_random_response():
    responses = [
        "Dediğinizi anlamadım ben sadece kitap öneri sistem chatbotuyum. Sevdiğiniz türden kitaplar önerebilirim.",
        "Ben sadece kitap öneri sistemiyim. Üzgünüm sorunuzu anlamadım sevdiğiniz kitap türleri var mıdır?",
        "Hmm, sanırım şu anda o konuyu anlamıyorum.",
        "Sizin için kitaplar önerebilirim sadece sevdiğiniz türlerden konuşalım.",
        "Kitap öneri sistemi için geliştirilmiş bir yapay zekayım. Sevdiğiniz tür kitapları sorabilirseniz yardımcı olabilirim."
    ]
    return random.choice(responses)

# Modelden bir yanıt almak için bir fonksiyon tanımlıyoruz
def get_response(msg, eposta, birim):
    sentence = tokenize(msg)  # Gelen mesajı tokenize ediyoruz
    X = bag_of_words(sentence, all_words)  # Bag of words vektörünü oluşturuyoruz
    X = X.reshape(1, X.shape[0])  # Boyutları yeniden şekillendiriyoruz
    X = torch.from_numpy(X).to(device)  # PyTorch tensorüne dönüştürüyoruz

    output = model(X)  # Modelden bir çıktı alıyoruz
    _, predicted = torch.max(output, dim=1)  # En yüksek olasılıklı tahmini alıyoruz

    tag = tags[predicted.item()]  # Etiketi alıyoruz

    probs = torch.softmax(output, dim=1)  # Olasılıkları hesaplıyoruz
    prob = probs[0][predicted.item()]  # İlgili etiketin olasılığını alıyoruz

    # Olasılık belirli bir eşik değerinden yüksekse
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # Kullanıcının sorduğu soruyu kaydediyoruz
                conn = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="1234",
                    database="chatbotamy")
                c = conn.cursor()
                c.execute(INSERT_SQL,
                          (eposta, birim, tag, msg, random.choice(intent['responses'])))
                conn.commit()
                conn.close()
                return random.choice(intent['responses'])  # Yanıtı döndürüyoruz

        # Kullanıcının sorduğu soru hiçbir etikete uymuyorsa
        conn = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="1234",
                    database="chatbotamy")
        c = conn.cursor()
        c.execute(INSERT_SQL,
                  (eposta, birim, "none", msg, "Bilinmeyen."))
        conn.commit()
        conn.close()
        return get_random_response()  # Rastgele bir yanıt döndürüyoruz
    else:
        # Kullanıcının sorduğu soru hiçbir etikete uymuyorsa
        conn = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="1234",
                    database="chatbotamy")
        c = conn.cursor()
        c.execute(INSERT_SQL,
                  (eposta, birim, "none", msg, "bilinmeyen"))
        conn.commit()
        conn.close()
        return get_random_response()  # Rastgele bir yanıt döndürüyoruz

# Ana döngü
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")  # Kullanıcıdan bir giriş alıyoruz
        if sentence == "quit":
            break

        resp = get_response(sentence)  # Yanıt alıyoruz
        print(resp)  # Yanıtı yazdırıyoruz
