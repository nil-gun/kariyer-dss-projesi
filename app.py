import warnings
warnings.filterwarnings("ignore") 

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- YENİ KÜTÜPHANE İÇE AKTARIMI ---
from google import genai
from google.genai import types
import datetime 

# --- 1. SAYFA, API VE YAPAY ZEKA AYARLARI ---
st.set_page_config(page_title="KAYKARDES | Akıllı KDS", layout="wide", page_icon="🎯")

# Yeni istemci (client) yapısı
API_ANAHTARI = "AIzaSyC9rGq7Js-OQGErLBhHilxdF24c1sOjdgg" 
client = genai.Client(api_key=API_ANAHTARI)

su_an = datetime.datetime.now().strftime("%d %B %Y, Saat: %H:%M")
sistem_talimati = f"Sen çok tecrübeli bir Endüstri Mühendisliği Kariyer Danışmanısın. Bugünün güncel tarihi ve saati şudur: {su_an}. Kullanıcı sana yıl, ay, gün veya saat sorarsa her zaman bu güncel zaman bilgisini kullanarak cevap ver."

# Yeni konfigürasyon yapısı
config = types.GenerateContentConfig(
    system_instruction=sistem_talimati,
)

# --- HAFIZA (SESSION STATE) YAPILANDIRMASI ---
if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = []
    
if "chat_oturumu" not in st.session_state:
    st.session_state.chat_oturumu = client.chats.create(
        model='gemini-2.5-flash',
        config=config
    )

# --- 2. MODELLERİ YÜKLEME ---
@st.cache_resource
def load_models():
    try:
        s_calisan = joblib.load("models/scaler_calisan.joblib")
        m_calisan = joblib.load("models/model_calisan.joblib")
        s_ogrenci = joblib.load("models/scaler_ogrenci.joblib")
        m_ogrenci = joblib.load("models/model_ogrenci.joblib")
        return s_calisan, m_calisan, s_ogrenci, m_ogrenci
    except Exception as e:
        st.error(f"Modeller yüklenirken hata oluştu: {e}")
        return None, None, None, None

s_calisan, m_calisan, s_ogrenci, m_ogrenci = load_models()

# --- 3. YARDIMCI SÖZLÜKLER ---
sektor_haritasi = {0: "Akademi", 1: "Hizmet", 2: "Üretim", 3: "Diğer"}
ders_notlari = {"Dersi almadım": 0, "Orta (CC-DC)": 1, "İyi (BB-CB)": 2, "Çok İyi (AA-BA)": 3}

# --- 4. ARAYÜZ TASARIMI ---
st.title("🎯 KAYKARDES: Yapay Zeka Destekli Karar Destek Sistemi")
st.markdown("Endüstri Mühendisliği yetkinliklerinize göre size en uygun kariyer sektörünü tahmin eder ve kişiselleştirilmiş bir yol haritası sunar.")

tab1, tab2 = st.tabs(["👔 Çalışanlar İçin KDS", "🎓 Öğrenciler İçin KDS"])

# ==========================================
# SEKME 1: ÇALIŞANLAR İÇİN KDS
# ==========================================
with tab1:
    st.header("Çalışan Profil Analizi")
    with st.form("calisan_formu_tam"):
        
        st.subheader("📌 Kişisel ve Akademik Bilgiler")
        c1, c2 = st.columns(2) 
        with c1:
            ort = st.number_input("Genel Not Ortalaması (GNO)", 0.0, 4.0, 3.00, step=0.01)
            ing_secim = st.selectbox("İngilizce Seviyesi", ["Kötü", "Orta", "İyi"], index=1)
        with c2:
            cinsiyet = st.selectbox("Cinsiyet", ["Kadın", "Erkek"])
            kulup = st.selectbox("Kulüp Üyeliği", ["Evet", "Hayır"])

        st.markdown("---")
        
        st.subheader("💼 Tecrübe ve Sektör Bilgisi")
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            endustri_staj = st.selectbox("Endüstri Stajı", ["Yok", "Üretim", "Hizmet"])
        with t2:
            gonullu_staj = st.selectbox("Gönüllü Staj", ["Yapmadım", "Üretim","Hizmet"])
        with t3:
            eski_sektor = st.selectbox("Eski Sektör", ["Hayır,çalışmadım", "Hizmet","Üretim","Yazılım"])
        with t4:
            memnuniyet = st.slider("Mevcut Durum Memnuniyeti", 0, 5, 3)

        st.markdown("---")
        
        st.subheader("📚 Lisans Ders Notları")
        d1, d2, d3, d4 = st.columns(4)
        not_secenekleri = list(ders_notlari.keys())
        
        with d1:
            bg = st.selectbox("Bilgisayar Programlama", not_secenekleri)
            hs = st.selectbox("Hizmet Sistemleri", not_secenekleri)
            ybs = st.selectbox("Yönetim Bilişim Sistemleri", not_secenekleri)
            ie = st.selectbox("İleri Excel", not_secenekleri, key="ie_dersi") 
        with d2:
            tt = st.selectbox("Tahmin Teknikleri", not_secenekleri)
            upk1 = st.selectbox("Üretim Planlama-1", not_secenekleri)
            upk2 = st.selectbox("Üretim Planlama-2", not_secenekleri)
        with d3:
            ya1 = st.selectbox("Yöneylem Araştırması-1", not_secenekleri)
            ya2 = st.selectbox("Yöneylem Araştırması-2", not_secenekleri)
            bz = st.selectbox("Benzetim", not_secenekleri)
        with d4:
            ttp = st.selectbox("Tesis Tasarımı ve Planlama", not_secenekleri)
            ietd = st.selectbox("İş Etüdü", not_secenekleri, key="ietd_dersi") 
            erg = st.selectbox("Ergonomi", not_secenekleri)

        submit_calisan = st.form_submit_button("Analiz Et ve Raporla 🚀")

    if submit_calisan:
        # Metin verilerini sayısallaştırma
        c_val = 0 if cinsiyet == "Kadın" else 1
        k_val = 0 if kulup == "Yok" else 1
        e_staj = 0 if endustri_staj == "Yok" else 1
        g_staj = 0 if gonullu_staj == "Yapmadım" else 1
        eskisek_val = ["Hayır,çalışmadım", "Hizmet","Üretim","Yazılım"].index(eski_sektor)

        ing_harita = {"Kötü": 1, "Orta": 2, "İyi": 3}
        ing_seviye = ing_harita[ing_secim]

        d_vals = [
            ders_notlari[bg], ders_notlari[hs], ders_notlari[ybs],
            ders_notlari[ie], ders_notlari[tt], ders_notlari[upk1],
            ders_notlari[upk2], ders_notlari[ya1], ders_notlari[ya2],
            ders_notlari[bz], ders_notlari[ttp], ders_notlari[ietd],
            ders_notlari[erg]
        ]

        model_girdisi = d_vals + [ing_seviye, c_val, k_val, ort, memnuniyet, eskisek_val, e_staj, g_staj]
        X_calisan = np.array([model_girdisi])
        
        try:
            X_calisan_scaled = s_calisan.transform(X_calisan)
            tahmin = m_calisan.predict(X_calisan_scaled)[0]
            sektor = sektor_haritasi.get(tahmin, "Bilinmeyen")
            
            # Yeni analiz yapıldığında eski sohbeti sıfırlıyoruz
            st.session_state.mesajlar = []
            st.session_state.chat_oturumu = client.chats.create(model='gemini-2.5-flash', config=config)
            
            st.success(f"🎯 KDS Tahmini: Makine Öğrenmesi modelimiz en uygun kariyer alanını **{sektor}** olarak belirledi.")
            
            with st.spinner("🤖 Kariyer gelişim raporunuz yapay zeka tarafından hazırlanıyor..."):
                ilk_prompt = f"""Karar Destek Sistemimiz (KDS), profesyonel bir Endüstri Mühendisi için '{sektor}' sektörünü en uygun alan olarak öngördü.
                Kişinin Özellikleri: Not Ortalaması: {ort}, İngilizce Seviyesi: {ing_secim}.
                
                GÖREVİN: Bu kişiye '{sektor}' sektöründe nasıl hızla yükselebileceğini anlatan kurumsal ve vizyoner bir 'Kariyer Gelişim Raporu' oluştur.
                Raporun sonunda öğrenciye sormak istediği bir şey olup olmadığını sor."""
                
                try:
                    cevap = st.session_state.chat_oturumu.send_message(ilk_prompt)
                    st.session_state.mesajlar.append({"role": "assistant", "content": cevap.text})
                except Exception as e:
                    st.error(f"🚨 Yapay zeka sistemi bir hata verdi! Detay: {e}")
        except Exception as e:
            st.error(f"🚨 Sistem çalışırken genel bir hata oluştu! Detay: {e}")

# ==========================================
# SEKME 2: ÖĞRENCİLER İÇİN KDS
# ==========================================
with tab2:
    st.header("Öğrenci Kariyer Yol Haritası")
    
    # Mavi kutu (st.info) kaldırıldı, doğrudan forma geçiyoruz
    with st.form("ogrenci_formu_tam"):
        
        # --- BÖLÜM 1: Genel ve Akademik Bilgiler ---
        st.subheader("📌 Akademik ve Kişisel Bilgiler")
        c1, c2, c3 = st.columns(3)
        with c1:
            ort_ogr = st.number_input("Genel Not Ortalaması (GNO) ", 0.0, 4.0, 2.80, step=0.01)
        
            ing_ogr_secim = st.selectbox("İngilizce Seviyesi ", ["Kötü", "Orta", "İyi"], index=1, key="ing_ogr_key")
       
            # Modelde stajlar sayısal (0: Yok, 1: Üretim, 2: Hizmet vb. senin kodundaki encoder'a göre)
            staj_end_secim = st.selectbox("Endüstri Stajı Durumu", ["Yok", "Üretim", "Hizmet"], key="staj_end_key")
            staj_gon_secim = st.selectbox("Gönüllü Staj Durumu", ["Yok", "Üretim", "Hizmet"], key="staj_gon_key")

        st.markdown("---")
        
        # --- BÖLÜM 2: Ders Başarıları (13 Dersin Tamamı) ---
        st.subheader("📚 Lisans Ders Notları")
        d1, d2, d3, d4 = st.columns(4)
        not_secenekleri_ogr = list(ders_notlari.keys())
        
        with d1:
            bg_o = st.selectbox("Bilgisayar Programlama ", not_secenekleri_ogr, key="bg_o")
            hs_o = st.selectbox("Hizmet Sistemleri ", not_secenekleri_ogr, key="hs_o")
            ybs_o = st.selectbox("Yönetim Bilişim Sist. ", not_secenekleri_ogr, key="ybs_o")
            ie_o = st.selectbox("İleri Excel ", not_secenekleri_ogr, key="ie_o")
        with d2:
            tt_o = st.selectbox("Tahmin Teknikleri ", not_secenekleri_ogr, key="tt_o")
            upk1_o = st.selectbox("Üretim Planlama-1 ", not_secenekleri_ogr, key="upk1_o")
            upk2_o = st.selectbox("Üretim Planlama-2 ", not_secenekleri_ogr, key="upk2_o")
        with d3:
            ya1_o = st.selectbox("Yöneylem Araştırması-1 ", not_secenekleri_ogr, key="ya1_o")
            ya2_o = st.selectbox("Yöneylem Araştırması-2 ", not_secenekleri_ogr, key="ya2_o")
            bz_o = st.selectbox("Benzetim ", not_secenekleri_ogr, key="bz_o")
        with d4:
            ttp_o = st.selectbox("Tesis Tasarımı ve Plan. ", not_secenekleri_ogr, key="ttp_o")
            ietd_o = st.selectbox("İş Etüdü", not_secenekleri_ogr, key="ietd_o")
            erg_o = st.selectbox("Ergonomi ", not_secenekleri_ogr, key="erg_o")

        submit_ogrenci = st.form_submit_button("Analiz Et ve Raporla 🚀")

    if submit_ogrenci:
        # Sayısallaştırma İşlemleri
        ing_harita = {"Kötü": 1, "Orta": 2, "İyi": 3}
        ing_val_ogr = ing_harita[ing_ogr_secim]
        
        staj_harita = {"Yok": 0, "Üretim": 1, "Hizmet": 2}
        s_end_val = staj_harita[staj_end_secim]
        s_gon_val = staj_harita[staj_gon_secim]
        
        # 13 Dersin listesi
        d_vals_ogr = [
            ders_notlari[bg_o], ders_notlari[hs_o], ders_notlari[ybs_o],
            ders_notlari[ie_o], ders_notlari[tt_o], ders_notlari[upk1_o],
            ders_notlari[upk2_o], ders_notlari[ya1_o], ders_notlari[ya2_o],
            ders_notlari[bz_o], ders_notlari[ttp_o], ders_notlari[ietd_o],
            ders_notlari[erg_o]
        ]
        
        # Model 2 Girdi Sırası (Tam 17 Özellik):
        # [bg, hs, ybs, ie, tt, upk1, upk2, ya1, ya2, bz, ttp, ietd, erg, ort, ing, endustristaj, gonullustaj]
        model_girdisi_ogr = d_vals_ogr + [ort_ogr, ing_val_ogr, s_end_val, s_gon_val]
        X_ogr = np.array([model_girdisi_ogr])
        
        try:
            X_ogr_scaled = s_ogrenci.transform(X_ogr)
            tahmin_ogr = m_ogrenci.predict(X_ogr_scaled)[0]
            sektor_ogr = sektor_haritasi.get(tahmin_ogr, "Bilinmeyen")
            
            # Sohbet hafızasını temizle
            st.session_state.mesajlar = []
            # Yoğunluk hatasını engellemek için 2.0-flash modelini kullanıyoruz
            st.session_state.chat_oturumu = client.chats.create(model='gemini-2.0-flash', config=config)
            
            st.success(f"🎓 KDS Tahmini: Kişisel ve akademik yetkinliklerinize göre en güçlü olduğunuz alan **{sektor_ogr}** sektörü.")
            
            with st.spinner("🤖 Yol haritanız yapay zeka tarafından çıkarılıyor..."):
                ilk_prompt = f"""Bir Endüstri Mühendisliği öğrencisi için Karar Destek Sistemimiz '{sektor_ogr}' sektörüne yatkınlık tespit etti.
                Öğrenci Verileri: GNO: {ort_ogr}, İngilizce: {ing_ogr_secim}, Staj Durumu: Endüstri({staj_end_secim}), Gönüllü({staj_gon_secim}).
                GÖREVİN: Bu öğrenciye '{sektor_ogr}' alanında uzmanlaşması için alması gereken seçmeli dersler, öğrenmesi gereken yazılımlar ve mezuniyet sonrası ilk 1 yılı için stratejik bir 'Kariyer Yol Haritası' oluştur.
                Raporun sonunda öğrenciye bu plan hakkında sormak istediği bir şey olup olmadığını sor."""
                
                try:
                    cevap = st.session_state.chat_oturumu.send_message(ilk_prompt)
                    st.session_state.mesajlar.append({"role": "assistant", "content": cevap.text})
                except Exception as e:
                    st.error(f"🚨 Yapay zeka sistemi yanıt veremedi. Lütfen 1 dakika sonra tekrar deneyin. (Hata: {e})")
        except Exception as e:
            st.error(f"🚨 Model tahmini sırasında bir hata oluştu! Detay: {e}")

# ==========================================
# İNTERAKTİF SOHBET (CHATBOT) ARAYÜZÜ
# (Sekmelerin Dışında - Sayfanın En Altında Görünür)
# ==========================================
if st.session_state.mesajlar:
    st.markdown("---")
    st.subheader("💬 Kariyer Danışmanınızla Sohbet Edin")
    
    for mesaj in st.session_state.mesajlar:
        with st.chat_message(mesaj["role"]):
            st.markdown(mesaj["content"])
            
    if kullanici_sorusu := st.chat_input("Danışmanınıza sorun (Örn: Bu sektörde mülakatlar nasıl geçer?)..."):
        st.session_state.mesajlar.append({"role": "user", "content": kullanici_sorusu})
        with st.chat_message("user"):
            st.markdown(kullanici_sorusu)
            
        with st.chat_message("assistant"):
            with st.spinner("Danışmanınız yanıtlıyor..."):
                try:
                    yeni_cevap = st.session_state.chat_oturumu.send_message(kullanici_sorusu)
                    st.markdown(yeni_cevap.text)
                    st.session_state.mesajlar.append({"role": "assistant", "content": yeni_cevap.text})
                except Exception as e:
                    st.error(f"🚨 Mesaj gönderilemedi! Hata Detayı:\n\n{e}")
                    st.session_state.mesajlar.pop()