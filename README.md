<div align="center">

<img src="YZR.webp" alt="YZR Net" width="140"/>

# YZR Net

[yzrnet.blogspot.com](https://yzrnet.blogspot.com/)

[![Kod: MIT](https://img.shields.io/badge/kod-MIT-blue.svg)](LICENSE)
[![Veri: CC BY 4.0](https://img.shields.io/badge/veri-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA.md)
![Platform](https://img.shields.io/badge/platform-Blogger-orange)
![Sürüm](https://img.shields.io/badge/s%C3%BCr%C3%BCm-v2.4.6-2E7D32)

</div>

---

## ✨ Hakkında

YZR Net, farklı alanlarda ihtiyaç duyulabilecek araçlar, uygulamalar ve dijital deneyimler geliştirmeyi amaçlayan bağımsız bir projedir. Kullanışlı, erişilebilir ve mümkün olduğunca sade çözümler sunmayı; bunu yaparken hesap oluşturma, ek yazılım kurma veya kişisel verileri bir sunucuya gönderme zorunluluğunu ortadan kaldırmayı hedefler. Araçların büyük bölümü doğrudan kullanıcının tarayıcısında çalışır.

## 🧭 Açık Veri Altyapısı

Bu depo, YZR Net'in çeşitli özelliklerinde yararlandığı açık veri akışlarını sağlar. Farklı kamuya açık kaynaklardan elde edilen veriler, web ortamında kullanılmaya uygun biçimde hazırlanarak yayımlanır. Rüzgâr, deprem ve yangın gibi farklı alanlara ait veriler, bu altyapının sunduğu içerikler arasındadır.

Depo, YZR Net'in web arayüzünden ziyade onu destekleyen veri altyapısına odaklanır. Kullanılan veriler, kendi kaynaklarının lisans ve atıf koşullarına tabidir.

## 📡 Açık veri akışları

GitHub Actions açık veri setlerini indirir, tarayıcı için küçük dosyalara dönüştürür ve **`data`** dalına (büyük kataloglar **`catalog`** dalına) yayımlar. Site bu dosyaları `raw.githubusercontent.com` üzerinden okur; dosya yoksa ilgili servise düşer. API anahtarı veya gizli bilgi kullanılmaz.

| Akış | Ne | Kaynak | Güncelleme |
|---|---|---|---|
| Rüzgâr | 10 m, 1° küresel rüzgâr | NOAA GFS | 6 saatte bir |
| Depremler ve yangınlar | Son 24 saat | USGS, NASA VIIRS | 30 dakikada bir |
| Kutup ışığı | Tahmin haritası | NOAA SWPC | 10 dakikada bir |
| Okyanus akıntıları | Yüzey akıntıları | Copernicus Marine | saatte bir |
| Gök cisimleri | Gezegenler, uydular, asteroit yörüngeleri, Ay konumu | NASA/JPL Horizons, SBDB | saatte bir |
| Kuyruklu yıldızlar | Tüm kuyruklu yıldızlar (~4100) | NASA/JPL SBDB | 6 saatte bir |
| Asteroit kataloğu | Numaralı tüm asteroitler | NASA/JPL SBDB | günde bir |
| Yıldız kataloğu | En parlak 1 milyon yıldız | ESA Gaia DR3, HYG v4.1 (CC BY-SA 4.0) | haftada bir |
| Galaksi kataloğu | ~50 bin galaksi | HyperLEDA | ayda bir |
| Samanyolu panoraması | 360° galaktik görüntü | ESO / S. Brunier (CC BY 4.0) | ayda bir |
| Yüzey haritaları | Ay, Merkür, Venüs, Mars, Ceres, Pluto ve büyük uyduların renk ve kabartma haritaları | USGS Astrogeology, NASA (LRO, MESSENGER, Magellan, Viking, MOLA, Dawn, New Horizons, Galileo, Voyager, Cassini), DLR | — |
| Diğer cisim dokuları | Dev gezegenler, asteroit kayası; Uranüs uyduları, Haumea, Gonggong ve Sedna (temsilî) | NASA/JPL ve görev ekipleri; [Astra-Planetshine](https://www.deviantart.com/astra-planetshine) (sanatçının izniyle) | — |
| Dünya atmosferi | Saçılma tabloları (gökyüzü, alacakaranlık, pus) | Hillaire (2020), Bruneton (2017) modelleri | — |
| Bulut katmanı | Dünya bulut dokuları | [SatVisor](https://github.com/satvisorcom/satvisor), AGPL-3.0 | — |
| Dünya haritası ve hava | Uydu görüntüsü, 3B harita, hava durumu (depoda tutulmaz) | Esri, ArcGIS, Open-Meteo (doğrudan servisten) | canlı |

## ⚖️ Lisans ve atıf

- **Kod** (betikler, iş akışları): [MIT](LICENSE). Kopyalarda telif satırı korunmalıdır.
- **Bu deponun ürettiği veri dosyaları:** [CC BY 4.0](LICENSE-DATA.md). Kullanan herkes **“YZR Net”** adını ve [yzrnet.blogspot.com](https://yzrnet.blogspot.com/) ya da [github.com/LykosTR/yzr-net](https://github.com/LykosTR/yzr-net) bağlantısını belirtmelidir.
- Kaynağının lisansı daha sıkı olan veriler o lisansla dağıtılır (yıldız kataloğu CC BY-SA 4.0, bulut katmanı AGPL-3.0). Üçüncü taraf kaynakların koşulları ayrıca geçerlidir.

© 2026 YZR Net
