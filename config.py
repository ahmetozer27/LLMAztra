# === config.py ===

# Bu dosya tüm model ayarlarını içerir.
config = {
    # Tokenizer ayarları
    "vocab_size": 50257,  # Tokenizer'ın kelime dağarcığı büyüklüğü
    "block_size": 512,  # Maksimum bağlam (context) uzunluğu (kaç token'e kadar bakabilir)

    # Model mimarisi
    "n_layer": 12,  # Transformer katmanı sayısı (derinlik)
    "n_head": 12,  # Attention başlığı (paralel attention işlemi)
    "d_model": 768,  # Her token'in vektör boyutu
    "ffn_hidden": 3072,  # Feedforward katmanının genişliği
    "dropout": 0.1,  # Dropout oranı

    # Eğitim parametreleri
    "batch_size": 4,  # Aynı anda işlenen örnek sayısı
    "learning_rate": 5e-4,  # Öğrenme hızı
    "epochs": 100,  # Eğitim süresi
    "weight_decay": 0.01,  # Ağırlıkların düzenlenmesi (overfitting önleme)
    "warmup_steps": 1000,  # Learning rate warmup adımları
    "gradient_clip_val": 1.0,  # Gradient clipping değeri

    # Optimizasyon ve sampling
    "temperature": 0.8,  # Sampling'de rastgelelik (düşükse daha kesin cevap)
    "top_k": 50,  # En iyi 50 token içinden seçim yapılır
    "top_p": 0.95,  # Nükleus sampling: olasılıkla seçilen token oranı
    "repetition_penalty": 1.1,  # Aynı kelime tekrarını cezalandırma

    # Kaydetme/Yükleme
    "save_path": "./checkpoints/model.pt",  # Model ağırlıklarının kaydedileceği yer
    "tokenizer_path": "./tokenizer/",  # Tokenizer dizini
    "dataset_path_txt": "./data/dataset.txt",  # Eğitim verisinin yolu txt
    "dataset_path_jsonl": "./data/dataset.jsonl",  # Eğitim verisinin yolu jsonl

    # Checkpoint ve logging
    "save_every": 1000,  # Kaç adımda bir checkpoint kaydet
    "log_every": 100,  # Kaç adımda bir log yazdır
    "eval_every": 500,  # Kaç adımda bir değerlendirme yap

    # Data preprocessing
    "stride_ratio": 0.5,  # Overlap oranı (0.5 = %50 overlap)
    "max_length": None,  # Maksimum veri uzunluğu (None = sınırsız)
}