## ML Optimizer (EXHAUSTIVE) — Kullanım ve Performans İpuçları

### Amaç
`ml_optimizer.py` geçmiş oran verilerinizden (odds.json) her tahmin türü için en iyi filtre ve tolerans kombinasyonlarını bulur. Çıktı, Türkçe özet JSON formatındadır.

### Hızlı Başlangıç
```bash
pip install -r requirements.txt
python ml_optimizer.py
```

### Çıktı
Script, aşağıdakine benzer bir JSON üretir:
```json
{
  "tahmin_turleri": [
    {
      "tahmin_adi": "2.5+ Gol Tahmini",
      "filtreler": [
        {"filtre_adi": "Double Chance 1-2", "tolerans": 0.05, "f1_score": 0.62, "precision": 0.65, "recall": 0.58, "support": 1240}
      ]
    }
  ],
  "kullanici_notlari": ["..."]
}
```


### Büyük Veri İçin Öneriler
- SSD kullanın; HDD okuma yavaşlatır
- Python 3.10+ ve 64-bit ortam tercih edin
- Maksimum filtre sayısını düşürün: `max_filters=min(5, len(fields))`


### Dosya Beklentileri
- `odds.json`: JSON array; alanlar: `odd_1`, `odd_x`, `odd_2`, `odd_1x`, `odd_12`, `odd_x2`, `bts_yes`, `bts_no`, `o+X.y`, `u+X.y`, `event_ft_result`, `event_halftime_result`, `event_date`.
- Eksik/bozuk kayıtlar otomatik atlanır.

### Notlar
- Her tahmin için minimum 3 filtre kullanılır, üst sınır varsayılan 6.
- F1, precision, recall metriğiyle en iyi kombinasyonlar döndürülür.
- Tüm CPU çekirdekleri kullanılır; Windows’ta antivirüs/Defender eşzamanlı tarama yavaşlatabilir.


