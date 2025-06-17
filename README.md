# 🧠 Custom Transformer Language Model

Bu proje, sıfırdan bir dil modeli (LLM) eğitmek için tasarlanmıştır. GPT benzeri transformer yapısı, özel tokenizer ve metin üretimi destekler.

---

## 🚀 Özellikler

- Byte-level BPE Tokenizer
- Transformer tabanlı dil modeli (multi-head attention, position embedding)
- Özelleştirilebilir eğitim ve sampling
- Basit text veri ile sıfırdan model eğitimi
- Metin üretimi (`generate.py`) üzerinden çalıştırılabilir

---

## 📁 Dosya Yapısı

```bash
Turkish_LLM_ModelAztra/
├── data/
│   └── dataset.txt         # Eğitim verisi (plain text, .jsonl veya .csv olabilir)
├── tokenizer/
│   └── train_tokenizer.py  # Tokenizer eğitimi (BPE veya SentencePiece)
├── model/
│   └── model.py            # Transformer mimarisi tanımı
├── trainer/
│   └── train.py            # Eğitim döngüsü ve kayıt
├── inference/
│   └── generate.py         # Eğitilmiş modelle metin üretimi
├── config.py               # Tüm model ve eğitim parametrelerini içerir
├── main.py                 # Ana çalıştırma kısmını içerir 
├── requirements.txt        # Gerekli Python paketleri
└── README.md               # Proje dokümantasyonu     # Eğitim verisi (kendi verinle değiştir)
