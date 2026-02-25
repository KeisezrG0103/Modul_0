import streamlit as st
import spacy
import pandas as pd
from spacy import displacy
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Modul 0 – Pengolahan Bahasa Alami",
    page_icon="🇮🇩",
    layout="wide",
)

# ── Load model & stemmer (cached) ────────────────────────────────────────────

@st.cache_resource
def load_nlp():
    return spacy.load("id_nusantara")


@st.cache_resource
def load_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()


nlp = load_nlp()
stemmer = load_stemmer()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Navigasi")
menu = st.sidebar.radio(
    "Pilih Topik:",
    [
        "✂️ Tokenisasi",
        "📝 Lematisasi",
        "🏷️ POS Tagging",
        "🔗 Dependency Parsing",
        "🔡 Case Identification",
        "🔬 Morfologi",
        "🌿 Stemming (Sastrawi)",
    ],
)

# ── Default teks ──────────────────────────────────────────────────────────────
DEFAULT_TEXT = (
    "Presiden Joko Widodo tersenyum gembira ketika Kereta Cepat "
    "Jakarta-Bandung atau KCJB bisa meraih kecepatan 350 kilometer per jam "
    "sehingga jarak 142,3 km jalur kereta cepat tersebut bisa ditempuh hanya "
    "dalam waktu 30 menit. Penunjuk kecepatan kereta dengan angka 350 km/h muncul "
    "di layar yang terpajang di bagian atas pintu penghubung antargerbong. "
    "Presiden menegaskan kereta tersebut tetap nyaman digunakan, bahkan saat "
    "mencapai kecepatan maksimal yang diperbolehkan, yaitu 350 km/jam."
)

# ── Input teks (global) ───────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Teks Input")
user_text = st.sidebar.text_area(
    "Masukkan teks Bahasa Indonesia:",
    value=DEFAULT_TEXT,
    height=200,
)

# proses NLP sekali, simpan di session_state agar efisien
if "last_text" not in st.session_state or st.session_state.last_text != user_text:
    st.session_state.doc = nlp(user_text)
    st.session_state.sentences = list(st.session_state.doc.sents)
    st.session_state.last_text = user_text

doc = st.session_state.doc
sentences = st.session_state.sentences
# ═════════════════════════════════════════════════════════════════════════════
# HALAMAN: TOKENISASI
# ═════════════════════════════════════════════════════════════════════════════
if menu == "✂️ Tokenisasi":
    st.title("✂️ Tokenisasi")

    st.subheader("Tokenisasi Kalimat")
    st.markdown(
        "Memecah teks menjadi kalimat-kalimat menggunakan sentence segmenter spaCy.")

    for i, sent in enumerate(sentences):
        st.markdown(f"**Kalimat {i}:** {sent.text}")

    st.divider()

    st.subheader("Tokenisasi Kata")
    sent_idx = st.selectbox(
        "Pilih kalimat:", range(len(sentences)),
        format_func=lambda i: f"Kalimat {i}: {sentences[i].text[:60]}…",
    )
    tokens = [t.text for t in sentences[sent_idx]]
    st.write(pd.DataFrame({"No": range(len(tokens)), "Token": tokens}))

# ═════════════════════════════════════════════════════════════════════════════
# HALAMAN: LEMATISASI
# ═════════════════════════════════════════════════════════════════════════════
elif menu == "📝 Lematisasi":
    st.title("📝 Lematisasi")
    st.markdown(
        "Lematisasi mengubah kata ke bentuk dasarnya (lemma) dan bentuk normalnya (norm)." 
    )

    sent_idx = st.selectbox(
        "Pilih kalimat:", range(len(sentences)),
        format_func=lambda i: f"Kalimat {i}: {sentences[i].text[:60]}…",
    )
    sent = sentences[sent_idx]

    rows = [
        {
            "Token": t.text,
            "Lemma": t.lemma_,
            "Norm": t.norm_,
            "Perbedaan": f"{t.lemma_} → {t.norm_}" if t.lemma_ != t.norm_ else "",
        }
        for t in sent
    ]
    df = pd.DataFrame(rows)

    def highlight_changed(row):
        changed = (row["Token"] != row["Lemma"]) or (
            row["Token"] != row["Norm"]) or (row["Perbedaan"] != "")
        color = "background-color: #d4f4dd" if changed else ""
        return [color] * len(row)

    st.dataframe(df.style.apply(highlight_changed, axis=1),
                 use_container_width=True)
    st.caption(
        "🟢 Baris berwarna hijau = kata berubah oleh proses lemmatisasis.")

# ═════════════════════════════════════════════════════════════════════════════
# HALAMAN: POS TAGGING
# ═════════════════════════════════════════════════════════════════════════════
elif menu == "🏷️ POS Tagging":
    st.title("🏷️ POS Tagging")

    tab1, tab2 = st.tabs(
        ["Coarse-grained (Universal POS)", "Fine-grained (Tag)"])

    sent_idx = st.selectbox(
        "Pilih kalimat:", range(len(sentences)),
        format_func=lambda i: f"Kalimat {i}: {sentences[i].text[:60]}…",
    )
    sent = sentences[sent_idx]

    with tab1:
        st.markdown(
            "**Universal POS** (`token.pos_`) – kategori sintaksis umum.")
        rows = [{"Token": t.text, "POS": t.pos_,
                 "Penjelasan": spacy.explain(t.pos_) or ""} for t in sent]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab2:
        st.markdown("**Fine-grained tag** (`token.tag_`) – tag lebih spesifik.")
        rows = [{"Token": t.text, "Tag": t.tag_,
                 "Penjelasan": spacy.explain(t.tag_) or ""} for t in sent]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# HALAMAN: DEPENDENCY PARSING
# ═════════════════════════════════════════════════════════════════════════════
elif menu == "🔗 Dependency Parsing":
    st.title("🔗 Dependency Parsing")
    st.markdown("Identifikasi relasi ketergantungan sintaksis antar kata.")

    sent_idx = st.selectbox(
        "Pilih kalimat:", range(len(sentences)),
        format_func=lambda i: f"Kalimat {i}: {sentences[i].text[:60]}…",
    )
    sent = sentences[sent_idx]

    # Tabel dependensi
    rows = [
        {
            "Token": t.text,
            "Dep": t.dep_,
            "Penjelasan Dep": spacy.explain(t.dep_) or "",
            "Head": t.head.text,
        }
        for t in sent
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Visualisasi displaCy
    st.subheader("Visualisasi Dependency Tree")
    svg = displacy.render(sent, style="dep", options={"compact": True})
    st.components.v1.html(
        f"<div style='overflow-x:auto'>{svg}</div>",
        height=350,
        scrolling=True,
    )

# ═════════════════════════════════════════════════════════════════════════════
# HALAMAN: CASE IDENTIFICATION
# ═════════════════════════════════════════════════════════════════════════════
elif menu == "🔡 Case Identification":
    st.title("🔡 Case Identification")
    st.markdown(
        "Menampilkan bentuk **lowercase** (`token.lower_`) dari setiap token.")

    sent_idx = st.selectbox(
        "Pilih kalimat:", range(len(sentences)),
        format_func=lambda i: f"Kalimat {i}: {sentences[i].text[:60]}…",
    )
    sent = sentences[sent_idx]
    rows = [{"Token (Asli)": t.text, "Lowercase": t.lower_} for t in sent]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# HALAMAN: MORFOLOGI
# ═════════════════════════════════════════════════════════════════════════════
elif menu == "🔬 Morfologi":
    st.title("🔬 Morfologi")
    st.markdown(
        "Analisis fitur morfologis setiap token menggunakan `token.morph`.")

    sent_idx = st.selectbox(
        "Pilih kalimat:", range(len(sentences)),
        format_func=lambda i: f"Kalimat {i}: {sentences[i].text[:60]}…",
    )
    sent = sentences[sent_idx]
    rows = [{"Token": t.text, "Morph": str(t.morph)} for t in sent]
    df = pd.DataFrame(rows)

    # highlight jika morph tidak kosong
    def highlight_changed_morph(row):
        color = "background-color: #d4f4dd" if row["Morph"] not in (
            "", "[]", "None") else ""
        return [color] * len(row)

    st.dataframe(df.style.apply(highlight_changed_morph, axis=1),
                 use_container_width=True)


    # Detail per token
    st.subheader("Detail Fitur Morfologi per Token")
    token_list = [t.text for t in sent]
    selected_token = st.selectbox("Pilih token:", token_list)
    token_obj = [t for t in sent if t.text == selected_token][0]
    morph_dict = token_obj.morph.to_dict()
    if morph_dict:
        st.json(morph_dict)
    else:
        st.info("Tidak ada fitur morfologi untuk token ini.")

# ═════════════════════════════════════════════════════════════════════════════
# HALAMAN: STEMMING (SASTRAWI)
# ═════════════════════════════════════════════════════════════════════════════
elif menu == "🌿 Stemming (Sastrawi)":
    st.title("🌿 Stemming dengan Sastrawi")
    sent_idx = st.selectbox(
        "Pilih kalimat:", range(len(sentences)),
        format_func=lambda i: f"Kalimat {i}: {sentences[i].text[:60]}…",
    )
    sent = sentences[sent_idx]

    # Kalimat penuh
    st.subheader("Stemming Kalimat Penuh")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Sebelum Stemming:**")
        st.write(sent.text)
    with col2:
        st.markdown("**Sesudah Stemming:**")
        st.write(stemmer.stem(sent.text))

    st.divider()

    # Per token
    st.subheader("Stemming Per Token")
    rows = [
        {
            "Token": t.text,
            "Lowercase": t.text.lower(),
            "Stem": stemmer.stem(t.text.lower()),
        }
        for t in sent
    ]
    df = pd.DataFrame(rows)
    # highlight jika stem berbeda

    def highlight_changed(row):
        color = "background-color: #d4f4dd" if row["Lowercase"] != row["Stem"] else ""
        return [color] * len(row)

    st.dataframe(df.style.apply(highlight_changed, axis=1),
                 use_container_width=True)
    st.caption(
        "🟢 Baris berwarna hijau = kata berhasil di-stem (berubah dari bentuk aslinya).")

    # Perbandingan Lemma vs Stem
    st.divider()
    st.subheader("Perbandingan: Lemma (spaCy) vs Stem (Sastrawi)")
    rows_cmp = [
        {
            "Token": t.text,
            "Lemma (spaCy)": t.lemma_,
            "Stem (Sastrawi)": stemmer.stem(t.text.lower()),
        }
        for t in sent
    ]
    df_cmp = pd.DataFrame(rows_cmp)

    # highlight jika lemma berbeda dari stem
    def highlight_changed_cmp(row):
        lemma = row["Lemma (spaCy)"]
        stem = row["Stem (Sastrawi)"]
        color = "background-color: #d4f4dd" if lemma != stem else ""
        return [color] * len(row)

    st.dataframe(df_cmp.style.apply(highlight_changed_cmp, axis=1),
                 use_container_width=True)
    st.caption(
        "🟢 Baris berwarna hijau = Lemma (spaCy) berbeda dari Stem (Sastrawi).")
