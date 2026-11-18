<div align="center">

<img src="YZR.webp" alt="YZR Net" width="140"/>

# YZR Net

[yzrnet.blogspot.com](https://yzrnet.blogspot.com/)

[![Lisans: MIT](https://img.shields.io/badge/lisans-MIT-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Blogger-orange)
![Sürüm](https://img.shields.io/badge/s%C3%BCr%C3%BCm-v2.3.3-2E7D32)

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
| Gök cisimleri: gezegenler, cüce gezegenler, doğal uydular, asteroitler (yörünge elemanları), Ay (saatlik konum vektörleri) | NASA/JPL Horizons, NASA/JPL SBDB | saatte bir |
| Kuyruklu yıldızlar: NASA kataloğundaki tüm kuyruklu yıldızlar (~4100; yörünge elemanları, çekirdek çapı, albedo, dönme süresi, parlaklık katsayıları) | NASA/JPL SBDB | 6 saatte bir |
| Asteroit kataloğu: numaralı asteroitlerin tümü (ad dizini + yörünge elemanları, **`catalog`** dalı) | NASA/JPL SBDB | günde bir (03:20 UTC) |
| Yıldız kataloğu: Güneş'e ~3 kpc içindeki en parlak 1 milyon yıldız (konum, paralaks, parlaklık, renk, sıcaklık, yarıçap, ışıma gücü, adlar ve katalog numaraları; **`catalog`** dalı, `stars/`) | ESA Gaia DR3, HYG v4.1 | haftada bir (Pazar 03:40 UTC) |
| Galaksi kataloğu: HyperLEDA'daki B parlaklığı < 15 olan ~50 bin galaksi (3B konum, çap, eksen oranı, açı, morfoloji, bar, B-V, PGC; Messier ve ortak adlar; **`catalog`** dalı, `galaxies/`) | HyperLEDA (Lyon Gözlemevi) | ayda bir (ayın 1'i 04:10 UTC) |
| Samanyolu panoraması: yıldızları ayıklanmış, arka plan ışığı çıkarılmış 360° galaktik koordinat görüntüsü (4096×2048 ve 2048×1024 WebP; **`catalog`** dalı, `milkyway/`) | ESO / S. Brunier (eso0932a), CC BY 4.0 | ayda bir (ayın 1'i 04:10 UTC) |
| Ay yüzey haritaları: renk (LROC WAC) ve gerçek ölçekli kabartmadan (LOLA 64 piksel/derece) hesaplanmış normal haritası (8K/4K/2K WebP, `moon/`) | NASA SVS (CGI Moon Kit), LRO LROC, LRO LOLA | elle (`moon` iş akışı) |
| Merkür yüzey haritaları: renk (MDIS MD3 mozaiği, doğal gri tona çekilmiş) ve gerçek ölçekli kabartmadan (MESSENGER 665 m DEM) hesaplanmış normal haritası (8K/4K/2K WebP, `mercury/`) | USGS Astrogeology, NASA/JHUAPL/CIW (MESSENGER) | elle (`mercury` iş akışı) |
| Venüs yüzey haritaları: sitenin önceki Magellan tabanlı renk haritası (kuzey yukarı çevrilmiş) ve gerçek ölçekli Magellan topografyasından normal haritası (8K/4K/2K WebP, `venus/`) | USGS Astrogeology, NASA/JPL (Magellan) | elle (`venus` iş akışı) |
| Mars yüzey haritaları: Viking renkli mozaiği ve gerçek ölçekli MOLA kabartmasından normal haritası (8K/4K/2K WebP, `mars/`) | USGS Astrogeology, NASA/JPL (Viking), NASA/GSFC (MGS MOLA) | elle (`mars` iş akışı) |
| Dünya bulut katmanı dokuları (`clouds/`, sabit dosya) | [SatVisor](https://github.com/satvisorcom/satvisor), AGPL-3.0 | değişmez |

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
| Gök cisimleri ve kuyruklu yıldızlar | NASA/JPL Horizons, SBDB |
| Yıldızlar | ESA Gaia DR3 (AIP Gaia aynası), HYG v4.1 (CC BY-SA 4.0: adlar, Hipparcos parlak yıldızları) |
| Galaksiler | HyperLEDA (Lyon Gözlemevi) |
| Samanyolu panoraması | ESO/S. Brunier (CC BY 4.0) |
| Ay yüzeyi | NASA's Scientific Visualization Studio, LRO LROC (NASA/GSFC/ASU), LRO LOLA (NASA/GSFC) |
| Merkür yüzeyi | NASA/JHUAPL/Carnegie Institution of Washington (MESSENGER MDIS), USGS Astrogeology |
| Venüs yüzeyi | NASA/JPL (Magellan), USGS Astrogeology |
| Mars yüzeyi | NASA/JPL (Viking), NASA/GSFC MOLA Science Team, USGS Astrogeology |
| Hava durumu | Open-Meteo |

## 📜 Veri hakları

Üçüncü taraf veriler kendi sağlayıcılarının lisans ve kullanım koşullarına tabidir. MIT lisansı yalnızca bu deponun kodunu kapsar.

## ⚖️ Lisans

[MIT](LICENSE) © 2026 YZR Net
