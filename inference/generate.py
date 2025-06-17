# === generate.py ===

import torch
import os
from model.aztra_transformer import TransformerLM
from transformers import PreTrainedTokenizerFast
from config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextGenerator:
    def __init__(self, model_path=None, tokenizer_path=None):
        """
        Text Generator class
        """
        self.model_path = model_path or config['save_path']
        self.tokenizer_path = tokenizer_path or config['tokenizer_path']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load tokenizer
        self.tokenizer = self._load_tokenizer()

        # Load model
        self.model = self._load_model()

        logger.info("TextGenerator başarıyla yüklendi!")

    def _load_tokenizer(self):
        """Tokenizer yükleme"""
        tokenizer_file = os.path.join(self.tokenizer_path, 'tokenizer.json')

        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(f"Tokenizer bulunamadı: {tokenizer_file}")

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)

        # Özel tokenlar ekle
        special_tokens = {
            'bos_token': '<bos>',
            'eos_token': '<eos>',
            'pad_token': '<pad>',
            'unk_token': '<unk>'
        }

        tokenizer.add_special_tokens(special_tokens)
        logger.info("Tokenizer yüklendi!")

        return tokenizer

    def _load_model(self):
        """Model yükleme"""
        if not os.path.exists(self.model_path):
            # Eğer best model varsa onu yükle
            best_model_path = self.model_path.replace('.pt', '_best.pt')
            if os.path.exists(best_model_path):
                self.model_path = best_model_path
                logger.info("En iyi model yükleniyor...")
            else:
                raise FileNotFoundError(f"Model bulunamadı: {self.model_path}")

        # Model oluştur
        model = TransformerLM(config)

        # Checkpoint yükle
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Checkpoint yüklendi: epoch {checkpoint.get('epoch', 'N/A')}")
        else:
            # Eski format
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()

        # Model bilgilerini yazdır
        total_params = model.count_parameters()
        logger.info(f"Model yüklendi! Parametre sayısı: {total_params:,}")

        return model

    def generate(self, prompt="", max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.95,
                 repetition_penalty=1.1, do_sample=True, num_return_sequences=1):
        """
        Metin üretme fonksiyonu

        Args:
            prompt: Başlangıç metni
            max_new_tokens: Maksimum yeni token sayısı
            temperature: Sampling sıcaklığı (yüksek = daha rastgele)
            top_k: Top-k sampling
            top_p: Nucleus sampling
            repetition_penalty: Tekrar cezası
            do_sample: Sampling kullan (False ise greedy)
            num_return_sequences: Kaç tane sonuç döndürülsün
        """

        # Prompt'u tokenize et
        if prompt:
            input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(prompt)
        else:
            input_ids = [self.tokenizer.bos_token_id]

        # Block size sınırı
        if len(input_ids) > config['block_size']:
            input_ids = input_ids[-config['block_size']:]
            logger.warning(f"Prompt çok uzun, son {config['block_size']} token alındı")

        input_tensor = torch.tensor([input_ids] * num_return_sequences, dtype=torch.long).to(self.device)

        generated_sequences = []

        for i in range(num_return_sequences):
            sequence = input_tensor[i:i + 1].clone()

            if do_sample:
                # Sampling ile üretim
                generated = self.model.generate(
                    sequence,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
            else:
                # Greedy decoding
                generated = self._greedy_generate(sequence, max_new_tokens)

            # Tekrar cezası uygula (basit implementasyon)
            if repetition_penalty != 1.0:
                generated = self._apply_repetition_penalty(generated, repetition_penalty)

            generated_sequences.append(generated[0])

        # Decode ve temizle
        results = []
        for seq in generated_sequences:
            text = self.tokenizer.decode(seq.tolist(), skip_special_tokens=True)
            text = self._clean_text(text)
            results.append(text)

        return results if num_return_sequences > 1 else results[0]

    def _greedy_generate(self, input_ids, max_new_tokens):
        """Greedy decoding"""
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Context window sınırı
                if input_ids.size(1) > config['block_size']:
                    input_ids = input_ids[:, -config['block_size']:]

                # Forward pass
                logits = self.model(input_ids)
                next_token_logits = logits[:, -1, :]

                # En yüksek olasılıklı token'ı seç
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # EOS token kontrolü
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Sequence'e ekle
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def _apply_repetition_penalty(self, input_ids, penalty):
        """Basit tekrar cezası (Bu daha gelişmiş olabilir)"""
        # Bu implementasyon çok basit, gerçek uygulamada daha sofistike olmalı
        return input_ids

    def _clean_text(self, text):
        """Metni temizle"""
        # Fazla boşlukları kaldır
        text = ' '.join(text.split())

        # Başlangıç ve bitiş boşluklarını kaldır
        text = text.strip()

        return text

    def chat(self, message, history=None, max_length=200):
        """
        Sohbet modu için özel generate fonksiyonu
        """
        if history is None:
            history = []

        # Sohbet geçmişini oluştur
        conversation = ""
        for h in history[-5:]:  # Son 5 mesajı al
            conversation += f"İnsan: {h['human']}\nAsistan: {h['assistant']}\n"

        conversation += f"İnsan: {message}\nAsistan:"

        response = self.generate(
            prompt=conversation,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

        # Sadece asistan cevabını al
        if "Asistan:" in response:
            response = response.split("Asistan:")[-1].strip()

        # İnsan: ile başlıyorsa kes
        if "İnsan:" in response:
            response = response.split("İnsan:")[0].strip()

        return response


def generate_interactive():
    """
    Interaktif metin üretim fonksiyonu
    """
    try:
        generator = TextGenerator()

        print("🤖 Aztra Text Generator")
        print("Metin üretmek için prompt girin, 'quit' yazarak çıkın.")
        print("-" * 50)

        while True:
            prompt = input("\n📝 Prompt: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Görüşürüz!")
                break

            if not prompt:
                prompt = ""

            print("\n🔄 Üretiliyor...")

            try:
                # Parametreleri kullanıcıdan al (opsiyonel)
                result = generator.generate(
                    prompt=prompt,
                    max_new_tokens=config.get('max_new_tokens', 100),
                    temperature=config.get('temperature', 0.8),
                    top_k=config.get('top_k', 50),
                    top_p=config.get('top_p', 0.95),
                    do_sample=True
                )

                print(f"\n🎯 Sonuç:\n{result}")
                print("-" * 50)

            except Exception as e:
                print(f"❌ Hata: {e}")

    except Exception as e:
        print(f"❌ Generator yüklenemedi: {e}")
        print("🔧 Önce modeli eğitmeyi deneyin!")


def generate_batch(prompts, output_file=None):
    """
    Toplu metin üretim fonksiyonu
    """
    try:
        generator = TextGenerator()
        results = []

        print(f"📦 {len(prompts)} prompt için metin üretiliyor...")

        for i, prompt in enumerate(prompts, 1):
            print(f"⏳ İşleniyor ({i}/{len(prompts)}): {prompt[:50]}...")

            result = generator.generate(
                prompt=prompt,
                max_new_tokens=200,
                temperature=0.8,
                num_return_sequences=1
            )

            results.append({
                'prompt': prompt,
                'generated': result
            })

        # Dosyaya kaydet
        if output_file:
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 Sonuçlar kaydedildi: {output_file}")

        return results

    except Exception as e:
        print(f"❌ Batch generation hatası: {e}")
        return []


# Basit kullanım fonksiyonu
def generate(prompt=None, max_new_tokens=100):
    """
    Basit generate fonksiyonu (backward compatibility için)
    """
    try:
        generator = TextGenerator()
        return generator.generate(
            prompt=prompt or "",
            max_new_tokens=max_new_tokens,
            temperature=config.get('temperature', 0.8),
            top_k=config.get('top_k', 50),
            top_p=config.get('top_p', 0.95)
        )
    except Exception as e:
        print(f"❌ Generation hatası: {e}")
        return ""