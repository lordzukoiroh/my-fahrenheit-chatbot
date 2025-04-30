
import torch
import gradio as gr
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from google.colab import drive
from datasets import Dataset
import json
import os
from typing import List, Tuple
import shutil
from functools import partial

# === CSS ve Emoji Fonksiyonu ===
current_css = """
#chatbot { height: 500px; overflow-y: auto; }
"""
def add_emojis(text: str) -> str:
    emoji_mapping = {
        "kitap": "📚", "kitaplar": "📚",
        "bilgi": "🧠", "öğrenmek": "🧠",
        "özgürlük": "🕊️", "özgür": "🕊️",
        "düşünce": "💭", "düşünmek": "💭",
        "ateş": "🔥", "yanmak": "🔥",
        "yasak": "🚫", "yasaklamak": "🚫",
        "tehlike": "⚠️", "tehlikeli": "⚠️",
        "devlet": "🏛️", "hükümet": "🏛️",
        "soru": "❓", "cevap": "✅",
        "okumak": "👁️", "oku": "👁️",
        "itfaiye": "🚒", "itfaiyeci": "🚒",
        "değişim": "🔄", "değişmek": "🔄",
        "isyan": "✊", "başkaldırı": "✊",
        "uyuşturucu": "💊", "hap": "💊",
        "televizyon": "📺", "tv": "📺",
        "mutlu": "😊", "mutluluk": "😊",
        "üzgün": "😞", "korku": "😨",
        "merak": "🤔", "meraklı": "🤔"
    }
    
    found_emojis = []
    words = text.split()
    for word in words:
        clean_word = word.lower().strip(".,!?")
        if clean_word in emoji_mapping:
            found_emojis.append(emoji_mapping[clean_word])
    
    unique_emojis = list(set(found_emojis))
    if unique_emojis:
        return f"{text} {' '.join(unique_emojis)}"
    return text

# === SABİTLER ===
MODEL_PATH = "/mnt/data/gpt2_finetuned_lora"
BOOK_PATH = "/mnt/data/fahrenheittt451.txt"
QA_PATH = "/mnt/data/qa_dataset.jsonl"
EMBEDDER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === GOOGLE DRIVE ===
drive.mount('/content/drive')

# === DOSYALARI GOOGLE DRIVE'DAN /mnt/data'ya KOPYALAMA ===
if os.path.exists(MODEL_PATH):
    shutil.rmtree(MODEL_PATH)
shutil.copytree('/content/drive/MyDrive/gpt2_finetuned_lora', MODEL_PATH)
shutil.copy('/content/drive/MyDrive/fahrenheittt451.txt', BOOK_PATH)

# === MODEL VE VERİYİ YÜKLE ===
def initialize_components():
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()

    with open(BOOK_PATH, "r", encoding="utf-8") as f:
        book_text = f.read()
    paragraphs = [p.strip() for p in book_text.split("\n") if len(p.strip()) > 100]
    embedder = SentenceTransformer(EMBEDDER_NAME)
    paragraph_embeddings = embedder.encode(paragraphs, convert_to_numpy=True)
    index = faiss.IndexFlatL2(paragraph_embeddings.shape[1])
    index.add(paragraph_embeddings)

    return model, tokenizer, embedder, paragraphs, index

model, tokenizer, embedder, paragraphs, index = initialize_components()

# === QA KAYIT ===
def save_feedback(question: str, answer: str, liked: bool, filepath: str = QA_PATH):
    qa_pair = {"question": question, "answer": answer, "liked": liked}
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")

def count_qa_examples(filepath: str = QA_PATH) -> int:
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

# === LoRA Fine-tuning ===
def lora_finetune(filepath: str = QA_PATH):
    print("⚙️ LoRA fine-tuning başlıyor...")
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["liked"]:  # Beğenilenleri al
                prompt = f"Kullanıcı: {item['question']}\nMontag:"
                data.append({"prompt": prompt, "output": item["answer"]})

    dataset = Dataset.from_list(data)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    base_model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        logging_steps=10,
        save_strategy="no",
        learning_rate=2e-4,
    )

    def tokenize(example):
        inputs = tokenizer(example["prompt"], truncation=True, padding="max_length", max_length=256)
        labels = tokenizer(example["output"], truncation=True, padding="max_length", max_length=256)
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_dataset = dataset.map(tokenize)
    from transformers import Trainer
    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    peft_model.save_pretrained(MODEL_PATH)
    print("✅ Model başarıyla güncellendi.")


# === CEVAP ÜRET ===
def retrieve_context(question: str, top_k: int = 1) -> str:
    try:
        question_vec = embedder.encode([question])
        _, indices = index.search(np.array(question_vec).astype("float32"), top_k)
        return "\n".join([paragraphs[i] for i in indices[0]])
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ""

def generate_answer(question: str, history: List[dict] = []) -> str:
    try:
        context = retrieve_context(question)
        prompt = (
            f"Kullanıcı: {question}\n\n"
            f"Bağlam:\n{context}\n\n"
            "Montag, kendisi bir itfaiyeci olarak, her zaman bilgiye ve fikir özgürlüğüne değer verir. "
            "Onun için kitaplar ve düşünceler, sistemin dayattığı kurallardan çok daha önemlidir. "
            "Montag'ın perspektifinden cevap ver:"
        )

        inputs = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(
            inputs,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.95,
            temperature=0.85,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Montag:" in response:
            response = response.split("Montag:")[-1].strip()
        else:
            response = response.replace(prompt, "").strip()

        # Emojilerle cevap ekle (Cümlenin sonuna)
        response = add_emojis(response)  # Emojiler cevabın sonunda olacak şekilde
        return response
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin."


# --- Yeni regenerate fonksiyonu (👎 basınca alternatif cevap üretir) ---
def regenerate_answer(chat_history: List[dict]) -> Tuple[str, List[dict]]:
    if not chat_history:
        return "", chat_history
    # Son soruyu ve cevabı al
    last_question = chat_history[-1]["content"]
    # Aynı soruya yeni cevap üret
    new_answer = generate_answer(last_question)
    # Son cevabı yeni cevapla güncelle
    chat_history[-1] = {"role": "user", "content": last_question}
    chat_history.append({"role": "assistant", "content": new_answer})
    return "", chat_history


# Save liked answers to the feedback file
def save_liked(chatbot_history):
    # Sadece son kullanıcı sorusu ve asistan cevabını kaydet (tüm sohbeti değil)
    if len(chatbot_history) < 2:
        return chatbot_history  # Hiçbir şey yapmadan geri dön
    
    last_user_msg = None
    last_assistant_msg = None
    
    for i in range(len(chatbot_history) - 2, -1, -1):
        if chatbot_history[i]["role"] == "user":
            last_user_msg = chatbot_history[i]["content"]
            break

    for i in range(len(chatbot_history) - 1, -1, -1):
        if chatbot_history[i]["role"] == "assistant":
            last_assistant_msg = chatbot_history[i]["content"]
            break

    if last_user_msg and last_assistant_msg:
        qa_pair = {"question": last_user_msg, "answer": last_assistant_msg, "liked": True}
        liked_qa_file = QA_PATH  # Zaten QA_PATH tanımlı (/mnt/data/qa_dataset.jsonl)
        with open(liked_qa_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
    
    return chatbot_history  # Sohbeti aynen geri döndür

    
def respond(msg, chatbot_history):
    # Soruyu chatbot_history'e ekle
    chatbot_history.append({"role": "user", "content": msg})
    
    # Cevabı üret
    answer = generate_answer(msg, chatbot_history)
    
    # Cevabı da chatbot_history'e ekle
    chatbot_history.append({"role": "assistant", "content": answer})
    
    # Soruyu ve cevabı döndür
    return "", chatbot_history  # "Kullanıcı" ve "Montag" yanıtları doğru sırayla dönecek

# --- Gradio arayüzü ---
def create_chat_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""  
        # 📚 Montag Chatbot (Fahrenheit 451)  
        *Ray Bradbury'nin Fahrenheit 451 romanındaki karakter Montag ile sohbet edin*  
        """)

        chatbot = gr.Chatbot(height=500, elem_id="chatbot", type="messages")
        msg = gr.Textbox(label="Montag'a sormak istediğiniz soruyu yazın", placeholder="Kitaplar neden yasaklandı?")
        
        with gr.Row():
            like_btn = gr.Button("👍 Beğendim")
            dislike_btn = gr.Button("👎 Beğenmedim (Alternatif Cevap)")
        clear = gr.Button("🧹 Sohbeti Temizle")

        msg.submit(respond, [msg, chatbot], [msg, chatbot])  # Soruyu ve cevabı ayrı baloncuklarda gönder
        like_btn.click(save_liked, chatbot, chatbot)  # Here, we're saving liked answers
        dislike_btn.click(regenerate_answer, chatbot, [msg, chatbot])
        clear.click(lambda: [], None, chatbot, queue=False)

        # Stil düzenlemesi: kullanıcı soruları sola, botun cevapları sağa yaslanacak
        demo.css = """
            #chatbot .message:nth-child(odd) {
                text-align: left;
                background-color: #f1f1f1;
                border-radius: 15px;
                padding: 10px;
            }
            #chatbot .message:nth-child(even) {
                text-align: right;
                background-color: #f0f0f0;
                border-radius: 15px;
                padding: 10px;
            }
        """

    return demo

# Başlat
if __name__ == "__main__":
    demo = create_chat_interface()
    demo.launch(share=True)
