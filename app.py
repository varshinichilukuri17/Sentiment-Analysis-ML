import streamlit as st
import joblib
import random

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="CinePulse AI",
    page_icon="🎬",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

.main-title {
    text-align: center;
    padding-top: 20px;
    padding-bottom: 10px;
}

.main-title h1 {
    color: #a855f7;
    font-size: 3rem;
}

.main-title p {
    color: #cbd5e1;
    font-size: 1.1rem;
}

.assistant-box {
    padding: 15px;
    border-radius: 15px;
    background-color: rgba(255,255,255,0.05);
    margin-bottom: 20px;
}

.quote-box {
    text-align: center;
    padding: 15px;
    border-radius: 15px;
    background-color: rgba(255,255,255,0.04);
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown("""
<div class="main-title">
<h1>🎬 CinePulse AI</h1>
<p>Analyze audience reactions and movie reviews using AI</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------- ASSISTANT ---------------- #

st.markdown("""
<div class="assistant-box">

### 🤖 CinePulse Assistant

Drop movie review below and I'll analyze:

✅ Audience Mood

✅ Review Strength

✅ Review Interpretation

✅ Confidence Score

</div>
""", unsafe_allow_html=True)

# ---------------- SAMPLE BUTTONS ---------------- #

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🎬 Action Movie"):
        st.session_state.review_text = (
            "The action sequences were thrilling and kept me engaged throughout the movie."
        )

with col2:
    if st.button("❤️ Romantic Movie"):
        st.session_state.review_text = (
            "The chemistry between the lead actors was amazing and the story was heartwarming."
        )

with col3:
    if st.button("😂 Comedy Movie"):
        st.session_state.review_text = (
            "The movie was hilarious and had me laughing from start to finish."
        )

with col4:
    if st.button("😡 Bad Movie"):
        st.session_state.review_text = (
            "This was one of the worst movies I have ever watched. The story was boring."
        )

# ---------------- INPUT ---------------- #

review = st.text_area(
    "🎭 Enter Movie Review",
    value=st.session_state.review_text,
    height=220,
    placeholder="Paste a movie review here..."
)

# ---------------- ANALYZE BUTTON ---------------- #

if st.button("🚀 Analyze Movie Review", use_container_width=True):

    if review.strip() == "":
        st.warning("Please enter a movie review.")
    else:

        review_vector = vectorizer.transform([review])

        prediction = model.predict(review_vector)

        probability = model.predict_proba(review_vector)

        confidence = round(max(probability[0]) * 100, 2)

        st.divider()

        if prediction[0] == "positive":

            mood = "😊 Positive Audience Reaction"
            strength = "🔥 Strong Positive Review"

            insight = """
The reviewer expresses strong appreciation,
enjoyment, and satisfaction with the movie.
"""

            st.success(mood)

        else:

            mood = "😞 Negative Audience Reaction"
            strength = "⚠️ Strong Negative Review"

            insight = """
The reviewer expresses dissatisfaction,
criticism, or disappointment with the movie.
"""

            st.error(mood)

        # ---------------- REPORT ---------------- #

        st.markdown("## 🎭 Movie Review Analysis")

        st.write(f"### {mood}")

        st.write(f"**Review Strength:** {strength}")

        st.write(f"**Confidence Score:** {confidence}%")

        st.progress(confidence / 100)

        st.write("### 💡 Review Interpretation")

        st.info(insight)

# ---------------- QUOTES ---------------- #

quotes = [
    "🎬 Every movie review tells a story.",
    "🍿 Great films create lasting impressions.",
    "🎭 Audience opinions shape cinema.",
    "⭐ Reviews help viewers discover great movies.",
    "🎥 Every film evokes an emotion."
]

st.divider()

st.markdown(
    f"""
    <div class="quote-box">
    <h4>{random.choice(quotes)}</h4>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- FOOTER ---------------- #

st.divider()

st.caption(
    "🎬 CinePulse AI | IMDb 50K Reviews | TF-IDF + Logistic Regression"
)
