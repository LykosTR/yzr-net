<div align="center">

<img src="YZR.webp" alt="YZR Net" width="140"/>

# YZR Net

[yzrnet.blogspot.com](https://yzrnet.blogspot.com/)

[![Lisans: MIT](https://img.shields.io/badge/lisans-MIT-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Blogger-orange)
![Sürüm](https://img.shields.io/badge/s%C3%BCr%C3%BCm-v2.4.5-2E7D32)

</div>

---

## ✨ Hakkında

YZR Net, farklı alanlarda ihtiyaç duyulabilecek araçlar, uygulamalar ve dijital deneyimler geliştirmeyi amaçlayan bağımsız bir projedir. Kullanışlı, erişilebilir ve mümkün olduğunca sade çözümler sunmayı; bunu yaparken hesap oluşturma, ek yazılım kurma veya kişisel verileri bir sunucuya gönderme zorunluluğunu ortadan kaldırmayı hedefler. Araçların büyük bölümü doğrudan kullanıcının tarayıcısında çalışır.

## 🧭 Açık Veri Altyapısı

Bu depo, YZR Net'in çeşitli özelliklerinde yararlandığı açık veri akışlarını sağlar. Farklı kamuya açık kaynaklardan elde edilen veriler, web ortamında kullanılmaya uygun biçimde hazırlanarak yayımlanır. Rüzgâr, deprem ve yangın gibi farklı alanlara ait veriler, bu altyapının sunduğu içerikler arasındadır.

Depo, YZR Net'in web arayüzünden ziyade onu destekleyen veri altyapısına odaklanır. Kullanılan veriler, kendi kaynaklarının lisans ve atıf koşullarına tabidir.

## 📡 Açık veri akışları

GitHub Actions, herkese açık veri setlerini indirir, tarayıcı dostu küçük dosyalara çevirir ve **`data`** dalına yayımlar (büyük kataloglar **`catalog`** dalında). Site bu dosyaları doğrudan `raw.githubusercontent.com` üzerinden okur (CORS `*`); dosya yoksa ya da eskiyse ilgili servise düşer. API anahtarı ya da gizli bilgi kullanılmaz.

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
| Yüzey haritaları | Ay, Merkür, Venüs, Mars, Ceres, Pluto ve büyük uyduların renk ve kabartma haritaları | USGS Astrogeology, NASA (LRO, MESSENGER, Magellan, Viking, MOLA, Dawn, New Horizons, Galileo, Voyager, Cassini), DLR | elle |
| Diğer cisim dokuları | Dev gezegenler, cüce gezegenler, asteroit kayası | NASA/JPL ve görev ekipleri | elle |
| Dünya atmosferi | Saçılma tabloları (gökyüzü, alacakaranlık, pus) | Hillaire (2020), Bruneton (2017) modelleri | elle |
| Bulut katmanı | Dünya bulut dokuları | [SatVisor](https://github.com/satvisorcom/satvisor), AGPL-3.0 | değişmez |

Uranüs uyduları ile Gonggong ve Sedna haritaları, [Astra-Planetshine](https://www.deviantart.com/astra-planetshine/art/Uranus-Project-Missing-Data-Titania-646920848) çalışmasına dayanır (sanatçının izniyle, atıf ile). Gonggong ve Sedna temsilî küreledir. Uydu görüntüsü (Esri), 3B harita (ArcGIS) ve hava durumu (Open-Meteo) doğrudan ilgili servislerden okunur.

## 📜 Veri hakları

Üçüncü taraf veriler kendi sağlayıcılarının lisans ve kullanım koşullarına tabidir. MIT lisansı yalnızca bu deponun kodunu kapsar.

## ⚖️ Lisans

[MIT](LICENSE) © 2026 YZR Net
