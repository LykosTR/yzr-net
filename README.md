<div align="center">

<img src="YZR.webp" alt="YZR Net" width="140"/>

# YZR Net

[yzrnet.blogspot.com](https://yzrnet.blogspot.com/)

[![Lisans: MIT](https://img.shields.io/badge/lisans-MIT-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Blogger-orange)
![Sürüm](https://img.shields.io/badge/s%C3%BCr%C3%BCm-v2.1.3-2E7D32)

</div>

---

## ✨ Hakkında

YZR Net, farklı alanlarda ihtiyaç duyulabilecek araçlar, uygulamalar ve dijital deneyimler geliştirmeyi amaçlayan bağımsız bir projedir. Kullanışlı, erişilebilir ve mümkün olduğunca sade çözümler sunmayı; bunu yaparken hesap oluşturma, ek yazılım kurma veya kişisel verileri bir sunucuya gönderme zorunluluğunu ortadan kaldırmayı hedefler. Araçların büyük bölümü doğrudan kullanıcının tarayıcısında çalışır.

## 🧭 Açık Veri Altyapısı

Bu depo, YZR Net'in çeşitli özelliklerinde yararlandığı açık veri akışlarını sağlar. Farklı kamuya açık kaynaklardan elde edilen veriler, web ortamında kullanılmaya uygun biçimde hazırlanarak yayımlanır. Rüzgâr, deprem ve yangın gibi farklı alanlara ait veriler, bu altyapının sunduğu içerikler arasındadır.

Depo, YZR Net'in web arayüzünden ziyade onu destekleyen veri altyapısına odaklanır. Kullanılan veriler, kendi kaynaklarının lisans ve atıf koşullarına tabidir.

## 📡 Açık veri akışları

GitHub Actions, herkese açık veri setlerini indirir, tarayıcı dostu küçük dosyalara çevirir ve **`data`** dalına yayımlar. Site bu dosyaları doğrudan `raw.githubusercontent.com` üzerinden okur (CORS `*`). API anahtarı ya da gizli bilgi kullanılmaz.

| Akış | Kaynak | Güncelleme |
|---|---|---|
| Rüzgâr 10 m, 1° küresel | NOAA/NCEP GFS | 04:30, 10:30, 16:30, 22:30 UTC |
| Depremler ve yangınlar (24 s) | USGS, NASA VIIRS (Esri Living Atlas) | 30 dakikada bir |
| Kutup ışığı tahmini | NOAA SWPC OVATION | 10 dakikada bir |
| Okyanus akıntıları | Copernicus Marine Service | saatte bir |
| Gök cisimleri: gezegenler, cüce gezegenler, doğal uydular, asteroitler (yörünge elemanları) | NASA/JPL Horizons, NASA/JPL SBDB | saatte bir |

## 📡 Veri kaynakları

Veri önce `data` dalından okunur; dosya yoksa ya da eskiyse ilgili servise doğrudan düşülür.

| Veri | Kaynak |
|---|---|
| Uydu görüntüsü | Esri World Imagery, Clarity |
| 3D harita ve yükseklik | ArcGIS Maps SDK for JavaScript |
| Depremler | USGS |
| Yangınlar | NASA FIRMS / VIIRS |
| Rüzgâr | NOAA GFS |
| Okyanus akıntıları | Copernicus Marine Service |
| Gök cisimleri | NASA/JPL Horizons, SBDB |
| Hava durumu | Open-Meteo |

## 📜 Veri hakları

Üçüncü taraf veriler kendi sağlayıcılarının lisans ve kullanım koşullarına tabidir. MIT lisansı yalnızca bu deponun kodunu kapsar.

## ⚖️ Lisans

[MIT](LICENSE) © 2026 YZR Net
