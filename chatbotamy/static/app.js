let countMsg = 0;  // Mesaj sayacı

// Mesaj sayacını artıran ve HTML içeriğini güncelleyen bir fonksiyon
function myFunction() {
    countMsg++;
    document.getElementById("countMsg").innerHTML = countMsg;
    return countMsg;
}

// E-posta kimliğini ayarlayan bir fonksiyon
function setId(id) {
    document.getElementById("epostaId").innerHTML = id;
}

// Chatbox sınıfı
class Chatbox {
    constructor() {
        this.args = {
            openButton: document.querySelector('.chatbox__button'),  // Chatbox'ı açan düğme
            chatBox: document.querySelector('.chatbox__support'),    // Chatbox
            sendButton: document.querySelector('.send__button')      // Gönderme düğmesi
        };

        this.state = false;  // Chatbox durumu
        this.messages = [];  // Mesajlar dizisi
    }

    // Chatbox'ı görüntüleyen fonksiyon
    display() {
        const { openButton, chatBox, sendButton } = this.args;

        // Açma düğmesine tıklama olayı
        openButton.addEventListener('click', () => this.toggleState(chatBox));

        // Otomatik olarak bir başlangıç mesajı gönderme
        this.sendInitialMessage(chatBox);

        // Gönderme düğmesine tıklama olayı
        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        // Klavyeden Enter tuşuna basıldığında gönderme işlemi
        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox);
            }
        });
    }

    // Chatbox durumunu değiştiren fonksiyon
    toggleState(chatbox) {
        if (document.getElementById("countMsg").innerHTML != 0) {
            this.state = !this.state;

            // Kutuyu göster veya gizle
            if (this.state) {
                chatbox.classList.add('chatbox--active');
            } else {
                chatbox.classList.remove('chatbox--active');
            }
        } else {
            document.getElementById("loginDiv").style.display = "block";
            document.getElementById("loginDivUpper").style.display = "block";
            this.state = !this.state;

            // Kutuyu göster veya gizle
            if (this.state) {
                chatbox.classList.add('chatbox--active');
            } else {
                chatbox.classList.remove('chatbox--active');
            }
        }
    }

    // Gönderme düğmesine tıklama işlemi
    onSendButton(chatbox) {
        var numPlaceholder = myFunction();
        var birim = document.getElementById("birimAdı").innerHTML;
        var msgCount = document.getElementById("countMsg").innerHTML;
        var textField = chatbox.querySelector('input');
        let text1 = textField.value;
        if (text1 === "") {
            return;
        }
        var id_ = document.getElementById("epostaId").innerHTML;
        let msg1 = { name: "User", message: text1 };
        this.messages.push(msg1);

        // Kullanıcının mesajını güncelleme
        this.updateChatText(chatbox);
        textField.value = '';

        // Botun yanıtı için yazma etkisi simülasyonu
        const typingTime = 2000; // 2 saniye
        const partialResponse = "Bot is typing..."; // Yazma işlemi sırasında gösterilecek mesaj
        const msg2 = { name: "BOT", message: partialResponse };
        this.messages.push(msg2);
        this.updateChatText(chatbox);

        // Yazma etkisi süresini bekleyin
        setTimeout(() => {
            // Sunucudan gerçek yanıtı alın
            fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                body: JSON.stringify({ 'message': text1, 'count': msgCount, 'id': id_, 'birim': birim }),
                mode: 'cors',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': '*/*',
                    'charset': 'utf8'
                },
            })
                .then(r => r.json())
                .then(r => {
                    // Yazma mesajını kaldır
                    const typingMessageIndex = this.messages.findIndex(msg => msg.name === "BOT" && msg.message === partialResponse);
                    if (typingMessageIndex !== -1) {
                        this.messages.splice(typingMessageIndex, 1);
                    }

                    // Yanıtı ekranda göster
                    let msg3 = { name: "BOT", message: r.answer };
                    this.messages.push(msg3);
                    this.updateChatText(chatbox);
                })
                .catch((error) => {
                    console.error('Error:', error);
                    this.updateChatText(chatbox);
                });
        }, typingTime);
    }

    // Başlangıç mesajını gönderen fonksiyon
    sendInitialMessage(chatbox) {
        const initialMessage = "Merhaba, kitap öneri sistemine hoşgeldiniz size nasıl yardımcı olabilirim?";
        const msg = { name: "BOT", message: initialMessage };
        this.messages.push(msg);
        this.updateChatText(chatbox);
    }

    // Chat metnini güncelleyen fonksiyon
    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function (item, index) {
            if (item.name === "BOT") {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>';
            } else {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>';
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox__messages');
        chatmessage.innerHTML = html;
    }
}

// Chatbox örneği oluşturuluyor ve görüntüleniyor
const chatbox = new Chatbox();
chatbox.display();

// Giriş formuna tıklama olayı
let loginForm = document.getElementById("loginForm");
loginForm.addEventListener("submit", (e) => {
    e.preventDefault();

    let eposta_form = document.getElementById("eposta_form");
    let birim_form = document.getElementById("birim_form");
    setId(eposta_form.value);
    document.getElementById("birimAdı").innerHTML = birim_form.value;
    document.getElementById("loginDiv").style.display = "none";
    document.getElementById("loginDivUpper").style.display = "none";

    document.getElementById("countMsg").innerHTML = 1;
    countMsg++;
});
