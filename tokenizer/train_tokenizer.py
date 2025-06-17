# === train_tokenizer.py ===

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast
import os
import json
import logging
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def jsonl_to_text():
    """JSONL dosyasını metin dosyasına çevir"""
    input_path = config["dataset_path_jsonl"]
    output_path = config["dataset_path_txt"]

    if not os.path.exists(input_path):
        logger.error(f"JSONL dosyası bulunamadı: {input_path}")
        return False

    # Çıktı dizinini oluştur
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_lines = 0
    processed_lines = 0

    with open(input_path, "r", encoding="utf-8") as f_in, \
            open(output_path, "w", encoding="utf-8") as f_out:

        for i, line in enumerate(f_in, start=1):
            total_lines += 1
            line = line.strip()

            if not line:
                continue  # Boş satırı atla

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Hatalı JSONL satırı {i}: {e}")
                continue

            # Farklı JSON formatlarını destekle
            text_parts = []

            # Soru-cevap formatı
            if "question" in data and "answer" in data:
                question = data["question"].strip()
                answer = data["answer"].strip()
                if question and answer:
                    text_parts.append(f"Soru: {question}")
                    text_parts.append(f"Cevap: {answer}")

            # Sadece text alanı
            elif "text" in data:
                text = data["text"].strip()
                if text:
                    text_parts.append(text)

            # Prompt-completion formatı
            elif "prompt" in data and "completion" in data:
                prompt = data["prompt"].strip()
                completion = data["completion"].strip()
                if prompt and completion:
                    text_parts.append(f"{prompt} {completion}")

            # Input-output formatı
            elif "input" in data and "output" in data:
                input_text = data["input"].strip()
                output_text = data["output"].strip()
                if input_text and output_text:
                    text_parts.append(f"{input_text} {output_text}")

            # Diğer formatlar için tüm string alanları birleştir
            else:
                for key, value in data.items():
                    if isinstance(value, str) and value.strip():
                        text_parts.append(value.strip())

            if text_parts:
                # Özel tokenlarla çevir
                full_text = "<bos> " + " ".join(text_parts) + " <eos>"
                f_out.write(full_text + "\n")
                processed_lines += 1

            # İlerleme raporu
            if i % 10000 == 0:
                logger.info(f"İşlenen satır: {i}")

    logger.info(f"✅ Dönüştürme tamamlandı!")
    logger.info(f"   Toplam satır: {total_lines}")
    logger.info(f"   İşlenen satır: {processed_lines}")
    logger.info(f"   Çıktı dosyası: {output_path}")

    return processed_lines > 0


def train_tokenizer():
    """Tokenizer eğitimi"""
    logger.info("🚀 Tokenizer eğitimi başlıyor...")

    # Çıktı dizinini oluştur
    os.makedirs(config["tokenizer_path"], exist_ok=True)

    # Eğitim dosyasını kontrol et
    if not os.path.exists(config["dataset_path_txt"]):
        logger.error(f"Eğitim dosyası bulunamadı: {config['dataset_path_txt']}")
        return False

    # Dosya boyutunu kontrol et
    file_size = os.path.getsize(config["dataset_path_txt"])
    if file_size == 0:
        logger.error("Eğitim dosyası boş!")
        return False

    logger.info(f"Eğitim dosyası boyutu: {file_size / (1024 * 1024):.2f} MB")

    # Tokenizer oluştur
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Pre-tokenizer (Byte-level BPE)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    # Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    # Özel tokenlar
    special_tokens = [
        "<pad>",  # Padding token
        "<bos>",  # Begin of sequence
        "<eos>",  # End of sequence
        "<unk>",  # Unknown token
        "<mask>",  # Mask token (future use)
        "<sep>",  # Separator token
        "<cls>",  # Classification token
    ]

    # Trainer oluştur
    trainer = trainers.BpeTrainer(
        vocab_size=config["vocab_size"],
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    logger.info(f"Tokenizer eğitimi başlıyor...")
    logger.info(f"  Kelime dağarcığı boyutu: {config['vocab_size']}")
    logger.info(f"  Özel tokenlar: {len(special_tokens)}")

    # Eğitim
    try:
        tokenizer.train(files=[config["dataset_path_txt"]], trainer=trainer)
        logger.info("✅ Tokenizer eğitimi tamamlandı!")
    except Exception as e:
        logger.error(f"❌ Tokenizer eğitimi başarısız: {e}")
        return False

    # Kaydet
    tokenizer_path = os.path.join(config["tokenizer_path"], "tokenizer.json")
    tokenizer.save(tokenizer_path)
    logger.info(f"💾 Tokenizer kaydedildi: {tokenizer_path}")

    # HuggingFace tokenizer olarak kaydet
    try:
        hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

        # Özel tokenları ayarla
        hf_tokenizer.bos_token = "<bos>"
        hf_tokenizer.eos_token = "<eos>"
        hf_tokenizer.unk_token = "<unk>"
        hf_tokenizer.pad_token = "<pad>"
        hf_tokenizer.mask_token = "<mask>"

        # Kaydet
        hf_tokenizer.save_pretrained(config["tokenizer_path"])
        logger.info(f"💾 HuggingFace tokenizer kaydedildi: {config['tokenizer_path']}")

    except Exception as e:
        logger.warning(f"⚠️ HuggingFace tokenizer kaydedilemedi: {e}")

    # Test et
    test_tokenizer(tokenizer_path)

    return True


def test_tokenizer(tokenizer_path):
    """Tokenizer test fonksiyonu"""
    logger.info("🧪 Tokenizer test ediliyor...")

    try:
        # Tokenizer yükle
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

        # Test metinleri
        test_texts = [
            "Merhaba, nasılsın?",
            "Bu bir test metnidir.",
            "Türkçe karakterler: ğüşıöç",
            "123 sayılar ve !@# özel karakterler",
            "Uzun bir metin örneği: " + "kelime " * 20
        ]

        for text in test_texts:
            # Encode
            tokens = tokenizer.encode(text)

            # Decode
            decoded = tokenizer.decode(tokens)

            logger.info(f"📝 Orijinal: {text}")
            logger.info(f"🔢 Tokenlar: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            logger.info(f"🔤 Token sayısı: {len(tokens)}")
            logger.info(f"🔄 Decode: {decoded}")
            logger.info("-" * 40)

        # Özel tokenları test et
        special_tests = {
            "<bos>": tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else None,
            "<eos>": tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None,
            "<pad>": tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else None,
            "<unk>": tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') else None,
        }

        logger.info("🎯 Özel tokenlar:")
        for token, token_id in special_tests.items():
            logger.info(f"  {token}: {token_id}")

        logger.info("✅ Tokenizer test başarılı!")

    except Exception as e:
        logger.error(f"❌ Tokenizer test başarısız: {e}")


def main():
    """Ana fonksiyon"""
    logger.info("🔧 Tokenizer eğitim süreci başlıyor...")

    # 1. JSONL'den text'e çevir (eğer JSONL varsa)
    if os.path.exists(config["dataset_path_jsonl"]):
        logger.info("📄 JSONL dosyası bulundu, metne çevriliyor...")
        if not jsonl_to_text():
            logger.error("❌ JSONL dönüştürme başarısız!")
            return

    # 2. Text dosyasını kontrol et
    if not os.path.exists(config["dataset_path_txt"]):
        logger.error(f"❌ Eğitim dosyası bulunamadı: {config['dataset_path_txt']}")
        logger.info("💡 Lütfen veri dosyanızı hazırlayın veya JSONL dosyasını yerleştirin.")
        return

    # 3. Tokenizer eğit
    if train_tokenizer():
        logger.info("🎉 Tokenizer eğitimi başarıyla tamamlandı!")
    else:
        logger.error("❌ Tokenizer eğitimi başarısız!")


