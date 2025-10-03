import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
try:
    # لمنع عرض RerunException كخطأ عند استخدام st.rerun()
    from streamlit.runtime.scriptrunner.script_runner import RerunException as _RerunException
except Exception:
    _RerunException = None

# إعداد الصفحة (يوضع مبكراً قبل أي أوامر Streamlit أخرى)
st.set_page_config(
    page_title="💓 نظام تنبؤ أمراض القلب",
    page_icon="❤️",
    layout="wide"
)

# دالة مساعدة لقراءة CSV مع اكتشاف الفاصل تلقائياً (يدعم "," و ";" وغيرهما)
def read_csv_auto(file_like):
    try:
        # محاولة اكتشاف الفاصل تلقائياً
        return pd.read_csv(file_like, sep=None, engine="python")
    except Exception:
        try:
            # محاولة ثانية مع الفاصلة المنقوطة
            return pd.read_csv(file_like, sep=';')
        except Exception:
            # أخيراً محاولـة القراءة الافتراضية
            return pd.read_csv(file_like)

# تحميل النموذج مع التخزين المؤقت واكتشاف المسار تلقائياً
@st.cache_resource(show_spinner=False)
def load_model():
    # نحاول التحميل من نفس مجلد الملف الحالي ثم المسار الحالي كاحتياط
    possible_paths = [
        Path(__file__).parent / "heart_model.pkl",
        Path.cwd() / "heart_model.pkl",
    ]
    for p in possible_paths:
        if p.exists():
            obj = joblib.load(str(p))
            # دعم شكلين: نموذج خام، أو قاموس يحوي الميتاداتا
            if isinstance(obj, dict) and 'estimator' in obj:
                est = obj['estimator']
                feats = obj.get('features')
                # إن لم تتوفر الميزات في الحمولة، حاول استخراجها من الـ Pipeline
                if not feats:
                    try:
                        if hasattr(est, 'named_steps') and 'pre' in est.named_steps:
                            pre = est.named_steps['pre']
                            if hasattr(pre, 'feature_names_in_'):
                                feats = list(pre.feature_names_in_)
                    except Exception:
                        pass
                st.session_state.model_features = feats
                # حمل المقاييس والخرائط إن وجدت
                st.session_state.model_metrics = obj.get('metrics')
                st.session_state.target_mapping = obj.get('target_mapping')
                return est
            else:
                st.session_state.model_features = None
                st.session_state.model_metrics = None
                st.session_state.target_mapping = None
                return obj
    raise FileNotFoundError("لم يتم العثور على ملف النموذج heart_model.pkl. ضع الملف في مجلد التطبيق.")

# تحميل كسول: لا نحمل النموذج حتى نحتاجه بالفعل (بعد تسجيل الدخول)
def get_model():
    return load_model()


# CSS لتوضيح الخطوط والألوان وتطبيق RTL
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;700;800&family=Tajawal:wght@400;700&display=swap');
    /* خلفية التطبيق واتجاه RTL */
    .stApp { background: linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%); color: var(--text); direction: rtl; text-align: right; }

    /* طباعة عربية أنيقة وكبيرة لسهولة القراءة */
    /* لوحة ألوان (3 ألوان أساسية ومتناغمة) */
    :root { --text:#1e252f; --muted:#6b7280; --primary:#2a9d8f; --primary-600:#21867a; --accent:#e9c46a; --neutral:#264653; --bg1:#ffffff; --bg2:#f9fbff; --card:#ffffff; --shadow: 0 10px 28px rgba(0,0,0,0.08); --base-font:20px; --space:14px; }
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown p, .stTextInput label, .stSelectbox label, .stRadio label { font-family: 'Cairo', 'Tajawal', 'Segoe UI', Tahoma, sans-serif; color: var(--text); text-align: right; line-height: 1.7; }
    h1 { font-size: 52px; font-weight: 800; letter-spacing: -0.5px; line-height: 1.2; }
    h2 { font-size: 36px; font-weight: 800; line-height: 1.3; }
    h3 { font-size: 26px; font-weight: 800; color: var(--text); }
    p, label, li, .stRadio, .stSelectbox, .stNumberInput, .stMetric { font-size: var(--base-font); font-weight: 600; }

    /* عناصر إدخال RTL */
    input, textarea, select { direction: rtl !important; text-align: right !important; }

    /* الحاوية العامة */
    .block-container { padding-top: 2.2rem; max-width: 1100px; }

    /* الشريط الجانبي */
    section[data-testid="stSidebar"] { background: #fbfbfd; border-inline-start: 1px solid #eef2f5; }

    /* تحسين مظهر زر الـ Popover (تفاصيل) ليكون صغيراً وبسيطاً وبدون حدود */
    div[data-testid="stPopover"] > button { font-size: 12px; padding: 2px 8px; border-radius: 10px; font-weight: 600; background: transparent; color:#6b7280; border:none; box-shadow:none; }
    div[data-testid="stPopover"] > button:hover { transform: translateY(-1px); background:#f5f6f8; color:#374151; }
    /* جسم الـ Popover لقراءة مريحة وتباين جيد */
    div[role="dialog"] { border-radius: 12px !important; border:1px solid #e5e7eb; box-shadow: var(--shadow) !important; }

    /* تصغير عناوين البطاقات لراحة العين */
    .card h4 { font-size: 18px !important; margin: 0 0 6px !important; font-weight: 800 !important; }

    /* ضبط تباعد الأعمدة والصفوف داخل النماذج */
    div[data-testid="column"] > div { margin-bottom: var(--space); }
    .card + .card { margin-top: calc(var(--space) * 1.2); }
    /* .section-title تعريف موحّد لاحقاً */

    /* جعل عناصر الإدخال ممتدة ومتناسقة العرض */
    .stTextInput, .stNumberInput, .stSelectbox, .stRadio { width: 100%; }
    /* حدود وخلفيات لطيفة لتمييز المدخلات */
    .stNumberInput input, .stTextInput input { border-radius: 10px !important; border: 1px solid #e5e7eb !important; background:#f9fafb !important; padding: 10px !important; }
    .stNumberInput input:focus, .stTextInput input:focus { outline: none !important; border-color: var(--primary) !important; box-shadow: 0 0 0 3px rgba(42,157,143,0.15) !important; background:#ffffff !important; }
    /* Selectbox container */
    .stSelectbox > div { border-radius: 10px !important; border: 1px solid #e5e7eb !important; background:#f9fafb !important; }
    .stSelectbox > div:focus-within { border-color: var(--primary) !important; box-shadow: 0 0 0 3px rgba(42,157,143,0.15) !important; background:#ffffff !important; }
    /* Radio group */
    .stRadio div[role="radiogroup"] { background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:8px 10px; }

    /* الأزرار */
    .stButton > button { border-radius: 12px; font-weight: 800; padding: 0.7rem 1.1rem; background: var(--primary); color: #fff !important; border: none; box-shadow: var(--shadow); transition: transform .12s ease, background .2s ease; }
    .stButton > button * { color: #fff !important; }
    .stButton > button:hover { background: var(--primary-600); transform: translateY(-1px); color: #fff !important; }
    .stButton > button:active { transform: translateY(0); color: #fff !important; }

    /* البطاقات العامة */
    .card { background: var(--card); border-radius: 16px; padding: 18px 20px; box-shadow: var(--shadow); border: 1px solid #eef2f5; margin-bottom: var(--space); }
    .card .stNumberInput, .card .stSelectbox, .card .stRadio, .card .stTextInput { margin-bottom: 8px; }

    /* تحسين التباين لعناصر المساعدة والنص الخافت (تعريف موحّد لاحقاً) */

    /* مقاييس/جداول */
    .stMetric { background: var(--card); border-radius: 14px; padding: 14px; box-shadow: var(--shadow); }
    .stDataFrame { background: var(--card); border-radius: 12px; box-shadow: var(--shadow); }
    /* فرض RTL على الجداول */
    div[data-testid="stDataFrame"] { direction: rtl; }
    div[data-testid="stDataFrame"] table { direction: rtl; }
    div[data-testid="stDataFrame"] th, div[data-testid="stDataFrame"] td { text-align: right !important; }

    /* لوحات النتائج */
    .result-panel { border-radius: 14px; padding: 16px 18px; border: 1px solid #eef2f5; box-shadow: var(--shadow); }
    .risk-low { background:rgba(46, 204, 113, .12); }
    .risk-mid { background:rgba(233, 196, 106, .18); }
    .risk-high { background:rgba(231, 76, 60, .14); }

    /* إشعار متوسط الشاشة مع حركة اختفاء تلقائية بعد ~5 ثوان */
    #center-toast { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 99999; background: rgba(17,24,39,0.92); color:#fff; padding: 12px 18px; border-radius: 12px; box-shadow: var(--shadow); font-weight: 800; letter-spacing: .2px; opacity: 1; pointer-events: none; animation: centerToast 5.4s ease forwards; }
    @keyframes centerToast { 0%, 85% { opacity: 1; } 100% { opacity: 0; } }

    /* تحسين الحقول */
    .stTextInput > div, .stNumberInput > div, .stSelectbox > div, .stRadio > div, .stSlider > div { max-width: 520px; }
    .stNumberInput input, .stTextInput input, .stSelectbox, .stRadio { font-size: 20px !important; }
    .stSlider { font-size: 20px !important; }

    /* شريط عناوين فرعي */
    .section-title { font-weight: 800; font-size: 22px; margin: 8px 0 4px; }
    .muted { color: var(--muted); }

    /* بطاقات إحصاءات في الصفحة الرئيسية */
    .stat { background: var(--card); border:1px solid #eef2f5; box-shadow: var(--shadow); padding:16px; border-radius:14px; text-align:center; }
    .stat .label { color: var(--muted); font-size:16px; }
    .stat .value { font-size:32px; font-weight:800; }

    /* حركات خفيفة */
    @keyframes fadeIn { from {opacity:0; transform: translateY(6px)} to {opacity:1; transform: translateY(0)} }
    .fade-in { animation: fadeIn .35s ease both; }
    </style>
    """,
    unsafe_allow_html=True
)

# حالة الجلسة الافتراضية
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = "ضيف"
if "nav" not in st.session_state:
    st.session_state.nav = "الصفحة الرئيسية"
if "page" not in st.session_state:
    st.session_state.page = "form"

# إشعار وسطي مخصص: دالة عرض ودالة Queue للرسالة
def render_center_toast():
    msg = st.session_state.pop("center_toast", None)
    if msg:
        st.markdown(
            f"""
            <div id="center-toast">{msg}</div>
            <script>
              const t = document.getElementById('center-toast');
              if (t) {{
                // إبقاء الإشعار مرئياً 5 ثوانٍ ثم تلاشيه وإزالته
                setTimeout(()=>{{ t.style.opacity='0'; }}, 5000);
                setTimeout(()=>{{ if (t && t.parentNode) t.parentNode.removeChild(t); }}, 5400);
              }}
            </script>
            """,
            unsafe_allow_html=True,
        )

def queue_center_toast(message: str):
    st.session_state.center_toast = message

# الشريط الجانبي
with st.sidebar:
    st.title("❤️ Heart AI")
    if st.session_state.authenticated:
        st.caption(f"مرحباً، {st.session_state.username}")
        options = ["الصفحة الرئيسية", "نموذج التنبؤ", "رفع ملف CSV", "تدريب النموذج", "حول"]
        icon_map = {
            "الصفحة الرئيسية": "🏠 الصفحة الرئيسية",
            "نموذج التنبؤ": "🩺 نموذج التنبؤ",
            "رفع ملف CSV": "📂 رفع ملف CSV",
            "تدريب النموذج": "🧠 تدريب النموذج",
            "حول": "ℹ️ حول",
        }
        nav = st.radio(
            "التنقّل",
            options,
            index=options.index(st.session_state.nav) if st.session_state.nav in options else 0,
            format_func=lambda x: icon_map.get(x, x)
        )
        st.session_state.nav = nav
        st.divider()
        # زر تسجيل الخروج أسفل القائمة وبشكل واضح
        if st.button("🚪 تسجيل الخروج", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.username = "ضيف"
            st.session_state.page = "form"
            st.rerun()
    else:
        st.subheader("🔐 تسجيل الدخول")
        username = st.text_input("اسم المستخدم", key="login_user")
        password = st.text_input("كلمة المرور", type="password", key="login_pass")
        if st.button("دخول"):
            if password == "1234":
                st.session_state.authenticated = True
                st.session_state.username = username or "مستخدم"
                st.session_state.nav = "الصفحة الرئيسية"
                queue_center_toast(f"✅ مرحباً {st.session_state.username}! تم تسجيل الدخول بنجاح 👋")
                st.rerun()
            else:
                st.error("كلمة المرور غير صحيحة.")

if not st.session_state.authenticated:
    st.stop()

# عرض أي إشعار وسطي مخصص (إن وُجد) بعد تهيئة الشريط الجانبي
render_center_toast()

def predict_single(arr: np.ndarray):
    model = get_model()
    # ميزات مطلوبة افتراضياً للاستخدام كأسماء أعمدة عند غياب model_features
    REQUIRED_FEATURES = [
        "age","sex","cp","trtbps","chol","fbs","restecg","thalachh","exng","oldpeak","slp","caa","thall"
    ]
    feats = st.session_state.get("model_features")
    cols = None
    if isinstance(feats, (list, tuple)) and len(feats) == arr.shape[1]:
        cols = list(feats)
    elif len(REQUIRED_FEATURES) == arr.shape[1]:
        cols = REQUIRED_FEATURES

    # دوماً حاول تمرير DataFrame بأسماء أعمدة ليستطيع ColumnTransformer الفهرسة بالأسماء
    if cols is not None:
        # أنشئ DataFrame بأسماء الأعمدة المطلوبة فقط، ولا تسقط إلى مصفوفة بلا أسماء
        X = pd.DataFrame(arr, columns=cols)
        # تحويل إجباري للأرقام
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce')
        if X.isna().any().any():
            raise ValueError("المدخلات تحتوي قيماً غير رقمية/ناقصة بعد التحويل. رجاءً صحح القيم.")
    else:
        raise ValueError("تعذر مطابقة أسماء الميزات مع مدخلات النموذج. تحقّق من توافق النموذج مع الخصائص الأساسية أو أعد التدريب.")
    prediction = model.predict(X)
    # محاولة الحصول على احتمال حقيقي، ثم قرار محوّل بالسيجمويد، وإلا لا يوجد احتمال
    proba = None
    try:
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0][1])
    except Exception:
        proba = None
    if proba is None:
        try:
            if hasattr(model, "decision_function"):
                val = float(model.decision_function(X)[0])
                proba = 1.0 / (1.0 + np.exp(-val))
        except Exception:
            proba = None
    return prediction[0], proba


# الصفحة الرئيسية
if st.session_state.nav == "الصفحة الرئيسية":
    # Hero
    st.markdown("<div class='card fade-in'>" 
                "<h2 style='margin:0'>💓 نظام تنبؤ أمراض القلب</h2>"
                "<p class='muted' style='margin:.25rem 0 0'>تحليل سريع وبسيط لتقدير نسبة الخطر بناءً على 13 سمة إكلينيكية</p>"
                "</div>", unsafe_allow_html=True)

    # إحصاءات وبطاقات معلومات لملء المساحة
    cA, cB, cC = st.columns(3)
    with cA:
        last_acc = None
        mm = st.session_state.get("model_metrics")
        # محاولة تحميل المقاييس من ملف النموذج إذا لم تكن محملة بعد
        if not isinstance(mm, dict):
            try:
                possible_paths = [
                    Path(__file__).parent / "heart_model.pkl",
                    Path.cwd() / "heart_model.pkl",
                ]
                for p in possible_paths:
                    if p.exists():
                        obj = joblib.load(str(p))
                        if isinstance(obj, dict) and 'metrics' in obj:
                            st.session_state.model_metrics = obj.get('metrics')
                            mm = st.session_state.model_metrics
                        break
            except Exception:
                pass
        if isinstance(mm, dict):
            last_acc = mm.get("accuracy")
        acc_txt = f"{last_acc*100:.1f}%" if isinstance(last_acc, (int, float)) else "—"
        st.markdown(f"<div class='stat fade-in'><div class='label'>دقة النموذج</div><div class='value'>{acc_txt}</div></div>", unsafe_allow_html=True)
    with cB:
        st.markdown("<div class='stat fade-in'><div class='label'>عدد السمات</div><div class='value'>13</div></div>", unsafe_allow_html=True)
    with cC:
        st.markdown("<div class='stat fade-in'><div class='label'>طرق الإدخال</div><div class='value'>فردي / CSV</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>دليل سريع</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='card fade-in'><h4>🩺 تنبؤ فوري</h4><p class='muted'>أدخل بياناتك الأساسية واحصل على نسبة الخطر مباشرةً.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card fade-in'><h4>📂 رفع CSV</h4><p class='muted'>حمّل ملفاً يحتوي على أعمدة معروفة للحصول على نتائج دفعة واحدة.</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card fade-in'><h4>🧠 تدريب نموذج</h4><p class='muted'>درّب نموذجاً جديداً من بياناتك الخاصة وفعّله فوراً.</p></div>", unsafe_allow_html=True)

    # أزرار إجراءات سريعة تغير التنقل
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("ابدأ التنبؤ الآن 🩺"):
            st.session_state.nav = "نموذج التنبؤ"
            st.rerun()
    with c2:
        if st.button("رفع ملف CSV 📂"):
            st.session_state.nav = "رفع ملف CSV"
            st.rerun()
    with c3:
        if st.button("تدريب نموذج 🧠"):
            st.session_state.nav = "تدريب النموذج"
            st.rerun()

# نموذج التنبؤ (فردي)
elif st.session_state.nav == "نموذج التنبؤ":
    st.header(f"🩺 أهلاً {st.session_state.username} — أدخل بياناتك الطبية")

    # تحذير مكونات النموذج بالنسبة لواجهة الإدخال

    # تحذير في حال عدم تطابق الميزات المتوقعة مع واجهة الإدخال الثابتة
    required_features = [
        "age","sex","cp","trtbps","chol","fbs","restecg","thalachh","exng","oldpeak","slp","caa","thall"
    ]
    expected = st.session_state.get("model_features")
    if expected and list(expected) != required_features:
        st.warning(
            "⚠️ النموذج المحمّل يتوقّع حقولاً مختلفة (أسماء/ترتيب) عن واجهة الإدخال الحالية.\n"
            "أعد تدريب النموذج أو استخدم نموذجاً مطابقاً لهذه الميزات: " + ", ".join(required_features)
        )

    # مفاتيح للقيم لإتاحة زر إعادة الضبط
    keys = {
        "age": "inp_age", "sex": "inp_sex", "cp": "inp_cp", "trtbps": "inp_trtbps",
        "chol": "inp_chol", "fbs": "inp_fbs", "restecg": "inp_restecg", "thalachh": "inp_thalachh",
        "exng": "inp_exng", "oldpeak": "inp_oldpeak", "slp": "inp_slp", "caa": "inp_caa", "thall": "inp_thall"
    }

    def help_popover(title: str, body: str):
        with st.popover("ℹ️", use_container_width=True):
            st.markdown(f"<h5 style='margin:0 0 6px'>{title}</h5>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(body, unsafe_allow_html=True)

    st.markdown("<div class='section-title'>المدخلات</div>", unsafe_allow_html=True)
    with st.expander("📘 توثيق الترميزات المتوقعة"):
        st.markdown(
            """
            - sex: 0=أنثى، 1=ذكر
            - cp: 0=ألم نموذجي، 1=ألم غير نموذجي، 2=ألم غير قلبي، 3=لا أعراض
            - restecg: 0=طبيعي، 1=شذوذ ST-T، 2=تضخّم بطين أيسر محتمل
            - fbs: 0=لا، 1=نعم (سكر صائم > 120 mg/dl)
            - exng: 0=لا، 1=نعم (ألم مع المجهود)
            - slp: 0=هابط، 1=مستوٍ، 2=صاعد
            - caa: 0..4 عدد الأوعية الرئيسية
            - thall: 0=طبيعي، 1=عيب ثابت، 2=عيب قابل للعكس، 3=غير محدد/حسب المصدر
            """
        )
    # تخطيط احترافي: مجموعات ضمن بطاقات، وكل بطاقة بعمودين لسهولة الإدخال
    # المجموعة 1: البيانات الأساسية
    st.markdown("""
        <div class='card fade-in'>
            <h4 style='margin-top:0'>👤 البيانات الأساسية</h4>
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        c, h = st.columns([4, 1])
        with c:
            age = st.number_input("العمر (سنة)", min_value=18, max_value=120, value=30, key=keys["age"]) 
        with h:
            help_popover("العمر", """
            <p>العمر بالسنوات.<br/>قيم أعلى قد ترتبط بخطر أعلى حسب الحالة السريرية.</p>
            """)
    with col2:
        c, h = st.columns([4, 1])
        with c:
            sex = st.radio("الجنس", ["ذكر", "أنثى"], key=keys["sex"]) 
        with h:
            help_popover("الجنس (sex)", """
            <ul>
                <li>0 = أنثى</li>
                <li>1 = ذكر</li>
            </ul>
            <p>يُستخدم ترميز رقمي لهذه القيم داخل النموذج.</p>
            """)

    # المجموعة 2: أعراض وفحوصات أساسية
    st.markdown("""
        <div class='card fade-in'>
            <h4 style='margin-top:0'>🩺 الأعراض والفحوصات الأساسية</h4>
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        c, h = st.columns([4, 1])
        with c:
            cp = st.selectbox("نوع ألم الصدر (cp)", [0, 1, 2, 3], key=keys["cp"]) 
        with h:
            help_popover("cp (ألم الصدر)", """
            <ul>
                <li>0 = ألم نموذجي</li>
                <li>1 = ألم غير نموذجي</li>
                <li>2 = ألم غير قلبي</li>
                <li>3 = لا أعراض</li>
            </ul>
            """)
    with col2:
        c, h = st.columns([4, 1])
        with c:
            restecg = st.selectbox("نتائج تخطيط القلب (restecg)", [0, 1, 2], key=keys["restecg"]) 
        with h:
            help_popover("restecg (تخطيط القلب)", """
            <ul>
                <li>0 = طبيعي</li>
                <li>1 = شذوذ ST-T</li>
                <li>2 = تضخّم بطين أيسر محتمل</li>
            </ul>
            """)

    # المجموعة 3: قياسات حيوية
    st.markdown("""
        <div class='card fade-in'>
            <h4 style='margin-top:0'>📊 القياسات الحيوية</h4>
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        c, h = st.columns([4, 1])
        with c:
            trtbps = st.number_input("ضغط انقباضي (mm Hg)", min_value=80, max_value=250, value=120, key=keys["trtbps"]) 
        with h:
            help_popover("trtbps (الضغط الانقباضي)", """
            <p>ضغط الدم الانقباضي أثناء الراحة (mm Hg).<br/>تقريباً: 90–120 ضمن الطبيعي.</p>
            """)
    with col2:
        c, h = st.columns([4, 1])
        with c:
            chol = st.number_input("الكوليسترول (mg/dl)", min_value=80, max_value=700, value=200, key=keys["chol"]) 
        with h:
            help_popover("chol (الكوليسترول)", """
            <ul>
                <li>&lt; 200 مرغوب</li>
                <li>200–239 حدّي</li>
                <li>≥ 240 مرتفع</li>
            </ul>
            """)
    col1, col2 = st.columns(2)
    with col1:
        c, h = st.columns([4, 1])
        with c:
            fbs = st.radio("سكر صائم > 120 mg/dl", [0, 1], key=keys["fbs"]) 
        with h:
            help_popover("fbs (سكر صائم &gt; 120)", """
            <ul>
                <li>0 = لا</li>
                <li>1 = نعم</li>
            </ul>
            """)
    with col2:
        c, h = st.columns([4, 1])
        with c:
            thalachh = st.number_input("أقصى معدل ضربات القلب", min_value=60, max_value=220, value=150, key=keys["thalachh"]) 
        with h:
            help_popover("thalachh (أقصى نبض)", """
            <p>أعلى معدل ضربات قلب تم الوصول إليه أثناء الاختبار.</p>
            """)

    # المجموعة 4: المجهود و ST
    st.markdown("""
        <div class='card fade-in'>
            <h4 style='margin-top:0'>🏃‍♂️ المجهود وقياس ST</h4>
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        c, h = st.columns([4, 1])
        with c:
            exng = st.radio("ألم عند المجهود (exng)", [0, 1], key=keys["exng"]) 
        with h:
            help_popover("exng (ألم مجهود)", """
            <ul>
                <li>0 = لا</li>
                <li>1 = نعم</li>
            </ul>
            """)
    with col2:
        c, h = st.columns([4, 1])
        with c:
            oldpeak = st.number_input("انخفاض ST (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key=keys["oldpeak"]) 
        with h:
            help_popover("oldpeak (انخفاض ST)", """
            <p>انخفاض مقطع ST مقارنةً بالراحة.<br/>قيم أعلى قد تترافق مع خطر أعلى.</p>
            """)
    col1, col2 = st.columns(2)
    with col1:
        c, h = st.columns([4, 1])
        with c:
            slp = st.selectbox("ميل مقطع ST (slp)", [0, 1, 2], key=keys["slp"]) 
        with h:
            help_popover("slp (ميل ST)", """
            <ul>
                <li>0 = هابط</li>
                <li>1 = مستوٍ</li>
                <li>2 = صاعد</li>
            </ul>
            """)

    # المجموعة 5: الأوعية والثلاسيميا
    st.markdown("""
        <div class='card fade-in'>
            <h4 style='margin-top:0'>🧬 الأوعية والثلاسيميا</h4>
        </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        c, h = st.columns([4, 1])
        with c:
            caa = st.selectbox("عدد الأوعية الرئيسية (caa)", [0, 1, 2, 3, 4], key=keys["caa"]) 
        with h:
            help_popover("caa (الأوعية الرئيسية)", """
            <ul>
                <li>0 = لا يوجد</li>
                <li>1 = وعاء واحد</li>
                <li>2 = وعاءان</li>
                <li>3 = ثلاثة</li>
                <li>4 = أربعة</li>
            </ul>
            """)
    with col2:
        c, h = st.columns([4, 1])
        with c:
            thall = st.selectbox("نوع الثلاسيميا (thall)", [0, 1, 2, 3], key=keys["thall"]) 
        with h:
            help_popover("thall (الثلاسيميا)", """
            <ul>
                <li>0 = طبيعي</li>
                <li>1 = عيب ثابت</li>
                <li>2 = عيب قابل للعكس</li>
                <li>3 = غير محدد/حسب المصدر</li>
            </ul>
            """)

    # إعداد القيم لاستخدامها في التنبؤ

    sex_val = 1 if st.session_state.get(keys["sex"], "ذكر") == "ذكر" else 0
    age = st.session_state.get(keys["age"], 30)
    cp = st.session_state.get(keys["cp"], 0)
    trtbps = st.session_state.get(keys["trtbps"], 120)
    chol = st.session_state.get(keys["chol"], 200)
    fbs = st.session_state.get(keys["fbs"], 0)
    restecg = st.session_state.get(keys["restecg"], 0)
    thalachh = st.session_state.get(keys["thalachh"], 150)
    exng = st.session_state.get(keys["exng"], 0)
    oldpeak = st.session_state.get(keys["oldpeak"], 1.0)
    slp = st.session_state.get(keys["slp"], 0)
    caa = st.session_state.get(keys["caa"], 0)
    thall = st.session_state.get(keys["thall"], 0)

    features = np.array([[age, sex_val, cp, trtbps, chol, fbs,
                          restecg, thalachh, exng, oldpeak, slp,
                          caa, thall]])

    # تحقق إدخال إضافي
    issues = []
    if chol > 600:
        issues.append("قيمة الكوليسترول تبدو عالية للغاية (>600 mg/dl)")
    if trtbps < 70 or trtbps > 250:
        issues.append("ضغط انقباضي خارج النطاق المنطقي (70–250)")
    if thalachh < 50 or thalachh > 230:
        issues.append("أقصى معدل ضربات القلب خارج النطاق المعتاد (50–230)")
    if oldpeak < 0 or oldpeak > 10:
        issues.append("قيمة oldpeak خارج النطاق (0–10)")
    if issues:
        st.warning("\n".join(["⚠️ تحقّق من المدخلات:"] + [f"- {m}" for m in issues]))

    if st.button("🔍 تنبؤ"):
        try:
            with st.spinner("جاري تحليل البيانات..."):
                pred, proba = predict_single(features)
                risk_value = float(proba) * 100 if proba is not None else None
                st.session_state.prediction = int(pred)
                st.session_state.risk_value = risk_value
                st.session_state.page = "result"
                # إشعار وسطي مختصر
                if risk_value is not None:
                    if st.session_state.prediction == 1:
                        queue_center_toast(f"⚠️ خطر محتمل ({risk_value:.1f}%)")
                    else:
                        queue_center_toast(f"✅ خطر منخفض ({risk_value:.1f}%)")
                else:
                    queue_center_toast("ℹ️ تم عرض التصنيف فقط لعدم توفر احتمال دقيق")
                # إعادة تشغيل فورية لإظهار الإشعار والانتقال للنتيجة
                st.rerun()
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            # لا تعرض RerunException كمخالفة؛ أعد رميها لتقوم ستريملت بإعادة التشغيل
            if _RerunException is not None and isinstance(e, _RerunException):
                raise e
            st.exception(e)

    # عرض النتيجة ضمن صفحة نموذج التنبؤ
    if st.session_state.page == "result":
        st.subheader(f"📊 نتيجة التنبؤ للمستخدم {st.session_state.username}")
        risk_value = st.session_state.get("risk_value", 0.0)
        pred_label = st.session_state.get("prediction", 0)

        # لوحة ملونة حسب مستوى الخطر
        if risk_value is None:
            level_class = "risk-mid"
        else:
            level_class = "risk-high" if risk_value >= 70 else ("risk-mid" if risk_value >= 30 else "risk-low")
        if pred_label == 1:
            msg = f"⚠️ هناك احتمال لإصابتك بمرض القلب." + (f" نسبة الخطر: {risk_value:.1f}%" if risk_value is not None else "")
            advice = """
            - 🥗 تناول غذاء صحي قليل الدهون والملح
            - 🏃‍♀️ مارس الرياضة بانتظام
            - 🚭 توقف عن التدخين
            - 🩺 راقب ضغط الدم والسكر
            - 😌 تجنب التوتر
            - 🩹 راجع الطبيب دوريًا
            """
            color = "red"
        else:
            msg = f"✅ لا توجد مؤشرات قوية على مرض القلب." + (f" نسبة الخطر: {risk_value:.1f}%" if risk_value is not None else "")
            advice = """
            - 🥗 استمر في نمط حياة صحي
            - 🚶 مارس المشي أو الرياضة الخفيفة
            - ⚖️ حافظ على وزن مثالي
            - 🛌 نم جيدًا
            - 🩺 تابع فحوصاتك الدورية
            """
            color = "green"

        st.markdown(f"<div class='result-panel {level_class}'>{msg}</div>", unsafe_allow_html=True)

        # بطاقات مقاييس سريعة
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("نسبة الخطر %", f"{risk_value:.1f}%" if risk_value is not None else "غير متاح")
        with m2:
            st.metric("التصنيف", "خطر" if pred_label == 1 else "غير خطر")
        with m3:
            # لا نعرض أي عتبة هنا بناءً على طلبك
            st.write("")

        # رسم نسبة الخطر باستخدام Plotly Gauge
        if risk_value is not None:
            fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_value,
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "#c6f5d9"},
                    {'range': [30, 70], 'color': "#fff2cc"},
                    {'range': [70, 100], 'color': "#f4c7c3"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_value
                }
            }
            ))
            st.plotly_chart(fig, use_container_width=True)

        # مخطط دونات لنسبة الخطر مقابل الأمان
        if risk_value is not None:
            safe = max(0.0, 100.0 - risk_value)
            pie = go.Figure(data=[go.Pie(labels=["خطر", "آمن"], values=[risk_value, safe], hole=.6, marker_colors=["#e76f51", "#2a9d8f"])])
            pie.update_layout(showlegend=True)
            st.plotly_chart(pie, use_container_width=True)
        else:
            st.info("تم عرض التصنيف فقط لعدم توفر احتمالات دقيقة من النموذج الحالي.")

        # جدول مقارنات بالقيم المرجعية
        ref = pd.DataFrame({
            'الميزة': ['العمر', 'ضغط انقباضي', 'الكوليسترول', 'أقصى نبض'],
            'قيمة المستخدم': [age, trtbps, chol, thalachh],
            'المدى المرجعي التقريبي': ['—', '90-120', '<200 مرغوب', '120-170']
        })
        st.markdown("### مقارنة بالقيم المرجعية")
        st.dataframe(ref, use_container_width=True)

        with st.expander("🩺 نصائح"):
            st.markdown(advice)

        # عرض المدخلات
        st.markdown("### المدخلات المستخدمة")
        df_in = pd.DataFrame(features, columns=[
            "age","sex","cp","trtbps","chol","fbs","restecg","thalachh","exng","oldpeak","slp","caa","thall"
        ])
        st.dataframe(df_in, use_container_width=True)

        # تنزيل تقرير بسيط
        report = f"prediction,{pred_label}\n" + (f"risk_percent,{risk_value:.2f}\n" if risk_value is not None else "risk_percent,NA\n")
        st.download_button("⬇️ تنزيل النتيجة (CSV)", data=report, file_name="heart_result.csv", mime="text/csv")

        if st.button("🔄 إعادة التنبؤ"):
            st.session_state.page = "form"
            st.rerun()

# صفحة تدريب النموذج
elif st.session_state.nav == "تدريب النموذج":
    st.header("🧠 تدريب نموذج جديد من CSV")
    st.caption("يمكنك رفع ملف أو عدة ملفات CSV وسيتم دمجها. اختر العمود الهدف ثم الإعدادات.")

    uploads = st.file_uploader("اختر ملف/ملفات CSV", type=["csv"], accept_multiple_files=True)
    if uploads:
        try:
            dfs = []
            required_features = [
                "age","sex","cp","trtbps","chol","fbs","restecg","thalachh","exng","oldpeak","slp","caa","thall"
            ]
            # تحقق لكل ملف على حدة قبل الدمج
            for f in uploads:
                df_i = read_csv_auto(f)
                miss_i = [c for c in required_features if c not in df_i.columns]
                if miss_i:
                    st.error(f"لا يمكن قبول الملف '{getattr(f, 'name', 'CSV')}' لغياب الخصائص: " + ", ".join(miss_i))
                    st.stop()
                dfs.append(df_i)
            df_all = pd.concat(dfs, axis=0, ignore_index=True)
            st.success(f"تم تحميل {len(uploads)} ملف/ملفات. الإجمالي: {df_all.shape[0]} سجل، {df_all.shape[1]} عمود")
            st.dataframe(df_all.head(20), use_container_width=True)

            # فرض توافق الخصائص الأساسية مع واجهة الإدخال
            required_features = [
                "age","sex","cp","trtbps","chol","fbs","restecg","thalachh","exng","oldpeak","slp","caa","thall"
            ]
            missing_train = [c for c in required_features if c not in df_all.columns]
            if missing_train:
                st.error("لا يمكن التدريب: الملف لا يحتوي على جميع الخصائص الأساسية المطلوبة: " + ", ".join(missing_train))
                st.stop()

            # اختيار العمود الهدف
            # حصر التدريب على نفس الخصائص فقط لضمان تطابق النموذج مع الواجهة
            target_col = st.selectbox("اختر العمود الهدف (التصنيف)", options=[c for c in df_all.columns if c not in required_features])

            # إعدادات التدريب
            with st.expander("⚙️ إعدادات التدريب"):
                algo = st.selectbox("الخوارزمية", ["LogisticRegression", "RandomForest"])
                test_size = st.slider("نسبة الاختبار (test_size)", 0.1, 0.4, 0.2, step=0.05)
                random_state = st.number_input("random_state", min_value=0, value=42, step=1)
                class_weight_balanced = st.checkbox("استخدام class_weight='balanced'", value=True)
                scale_features = st.checkbox("تقييس السمات (StandardScaler) - مفيد مع LogisticRegression", value=True)

            # تثبيت أعمدة السمات على الخصائص الأساسية حصراً
            feat_cols = required_features
            st.info("سيتم تدريب النموذج باستخدام الخصائص الأساسية فقط لضمان التوافق مع واجهة التنبؤ.")

            st.divider()
            if st.button("🚀 بدء التدريب"):
                try:
                    if target_col not in df_all.columns:
                        st.error("العمود الهدف غير موجود.")
                        st.stop()

                    # إزالة السجلات ذات القيم المفقودة في الأعمدة المستخدمة
                    work = df_all[feat_cols + [target_col]].dropna()

                    # تحويل الهدف إلى int إن أمكن مع توثيق الخريطة
                    y = work[target_col]
                    y_mapping = None
                    if y.dtype != int and y.dtype != np.int64:
                        try:
                            y = y.astype(int)
                        except Exception:
                            # محاولة تحويل الفئات النصية إلى أرقام
                            y_cat = y.astype("category")
                            y_mapping = dict(enumerate(y_cat.cat.categories))
                            y = y_cat.cat.codes

                    X = work[feat_cols]

                    # معالجة الأنواع غير الرقمية تلقائياً
                    num_cols = [c for c in feat_cols if np.issubdtype(X[c].dtype, np.number)]
                    cat_cols = [c for c in feat_cols if c not in num_cols]

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) > 1 else None
                    )

                    # إنشاء بايبلاين ما قبل المعالجة
                    transformers = []
                    if num_cols:
                        transformers.append(("num", StandardScaler() if (algo == "LogisticRegression" and scale_features) else "passthrough", num_cols))
                    if cat_cols:
                        transformers.append(("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols))
                    pre = ColumnTransformer(transformers=transformers, remainder='drop')

                    # إنشاء النموذج
                    if algo == "LogisticRegression":
                        base = LogisticRegression(max_iter=200, class_weight='balanced' if class_weight_balanced else None)
                        clf = Pipeline([
                            ("pre", pre),
                            ("clf", base)
                        ])
                    else:
                        base = RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight='balanced' if class_weight_balanced else None)
                        clf = Pipeline([
                            ("pre", pre),
                            ("clf", base)
                        ])

                    with st.spinner("جاري تدريب النموذج..."):
                        clf.fit(X_train, y_train)

                    # تقييم
                    y_pred = clf.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"✅ الدقة (Accuracy): {acc:.3f}")

                    try:
                        if hasattr(clf, "predict_proba"):
                            y_proba = clf.predict_proba(X_test)[:, 1]
                            try:
                                auc = roc_auc_score(y_test, y_proba)
                                st.info(f"ROC-AUC: {auc:.3f}")
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # تقرير تصنيفي
                    st.markdown("### تقرير تصنيفي")
                    rep = classification_report(y_test, y_pred, output_dict=False, digits=3)
                    st.code(rep)

                    # مصفوفة الالتباس
                    st.markdown("### مصفوفة الالتباس")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = go.Figure(data=go.Heatmap(z=cm, colorscale='Blues'))
                    fig_cm.update_layout(xaxis_title='Predicted', yaxis_title='Actual')
                    st.plotly_chart(fig_cm, use_container_width=True)

                    # أهمية السمات أو المعاملات
                    st.divider()
                    st.markdown("### أهمية السمات")
                    try:
                        final_est = clf.named_steps.get("clf", clf)
                        if hasattr(final_est, "feature_importances_") and len(cat_cols) == 0:
                            imp = pd.Series(final_est.feature_importances_, index=feat_cols).sort_values(ascending=False)
                            st.bar_chart(imp)
                        elif hasattr(final_est, "coef_") and len(cat_cols) == 0:
                            coef = pd.Series(final_est.coef_[0], index=feat_cols).sort_values(ascending=False)
                            st.bar_chart(coef)
                        else:
                            st.caption("تعذر إظهار أهمية السمات عند استخدام ترميز/تحويلات تجعل أسماء السمات مختلفة عن الأصل.")
                    except Exception:
                        st.caption("لا تتوفر أهمية السمات لهذا النموذج.")

                    # عرض خريطة ترميز الهدف إن وُجدت
                    if y_mapping is not None:
                        st.markdown("### خريطة ترميز الهدف")
                        map_df = pd.DataFrame(list(y_mapping.items()), columns=["label_id", "label_name"]).sort_values("label_id")
                        st.dataframe(map_df, use_container_width=True)

                    # حفظ النموذج
                    model_path = Path(__file__).parent / "heart_model.pkl"
                    try:
                        # حفظ مع ميتاداتا: الأعمدة، المقاييس، وخريطة الهدف إن وجدت
                        metrics = {"accuracy": float(acc)}
                        payload = {"estimator": clf, "features": feat_cols, "metrics": metrics}
                        if y_mapping is not None:
                            payload["target_mapping"] = y_mapping
                        joblib.dump(payload, model_path)
                        st.success(f"💾 تم حفظ النموذج في: {model_path}")
                        # تعيين المقاييس فوراً في الجلسة لعرض الدقة على الصفحة الرئيسية
                        st.session_state.model_metrics = metrics

                        # إتاحة التنزيل
                        with open(model_path, "rb") as mf:
                            st.download_button("⬇️ تنزيل النموذج المدرب", data=mf.read(), file_name="heart_model.pkl")

                        # زر لتفعيل النموذج فوراً في التطبيق
                        st.divider()
                        if st.button("✅ استخدام النموذج المدرب الآن"):
                            load_model.clear()
                            _ = load_model()
                            queue_center_toast("✅ تم تفعيل النموذج الجديد")
                    except Exception as e:
                        st.error(f"تعذر حفظ النموذج: {e}")

                except Exception as e:
                    st.exception(e)

        except Exception as e:
            st.exception(e)


# التنبؤ الدفعي عبر CSV
elif st.session_state.nav == "رفع ملف CSV":
    st.header("📂 رفع ملف CSV للتنبؤ الدفعي")
    st.caption("ينبغي أن يحتوي الملف على الأعمدة التالية مرتبة أو بأسماء مطابقة: age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall")

    uploaded = st.file_uploader("اختر ملف CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = read_csv_auto(uploaded)
            required_cols = ["age","sex","cp","trtbps","chol","fbs","restecg","thalachh","exng","oldpeak","slp","caa","thall"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                st.error("الأعمدة الناقصة: " + ", ".join(missing))
            else:
                # استخدم DataFrame بأسماء الأعمدة ليتوافق مع ColumnTransformer
                X = df[required_cols]
                model = get_model()
                # تنبؤ متجه بالكامل
                y_pred = model.predict(X)
                # احتمالات: predict_proba ثم decision_function->sigmoid، وإلا نستخدم التصنيف مع تحذير
                y_proba = None
                try:
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(X)[:, 1] * 100.0
                except Exception:
                    y_proba = None
                if y_proba is None:
                    try:
                        if hasattr(model, "decision_function"):
                            vals = model.decision_function(X)
                            y_proba = (1.0 / (1.0 + np.exp(-vals))) * 100.0
                    except Exception:
                        y_proba = None
                if y_proba is None:
                    st.info("ℹ️ لا يوفر النموذج احتمالات دقيقة؛ تم استخدام التصنيف الثنائي كمرجع تقريبي.")
                    y_proba = y_pred.astype(float) * 100.0
                out = df.copy()
                out["prediction"] = y_pred.astype(int)
                out["risk_percent"] = np.round(y_proba, 2)
                st.success("تم الحساب بنجاح")

                # ملخصات تفاعلية
                cA, cB, cC, cD = st.columns(4)
                with cA:
                    st.metric("عدد السجلات", len(out))
                with cB:
                    st.metric("متوسط نسبة الخطر %", f"{out['risk_percent'].mean():.2f}")
                with cC:
                    st.metric("أعلى نسبة خطر %", f"{out['risk_percent'].max():.2f}")
                with cD:
                    st.metric("أقل نسبة خطر %", f"{out['risk_percent'].min():.2f}")

                # عتبة الخطر وعرض الأعلى خطراً
                st.markdown("### عتبة تحديد الحالات عالية الخطر")
                thr = st.slider("اختر العتبة (%)", 0.0, 100.0, 50.0, step=1.0)
                out["high_risk"] = out["risk_percent"] >= thr
                st.write(f"عدد الحالات عالية الخطر (≥ {thr:.0f}%): {int(out['high_risk'].sum())}")

                # مخطط هيستوجرام لنسبة الخطر
                hist_fig = go.Figure(data=[go.Histogram(x=out["risk_percent"], nbinsx=20, marker_color="#c0392b")])
                hist_fig.update_layout(title="توزيع نسبة الخطر (%)", xaxis_title="نسبة الخطر", yaxis_title="عدد السجلات")
                st.plotly_chart(hist_fig, use_container_width=True)

                # عدادات للفئات المتنبأ بها
                counts = out["prediction"].value_counts().sort_index()
                bar_fig = go.Figure(data=[go.Bar(x=["غير خطر (0)", "خطر (1)"], y=[counts.get(0,0), counts.get(1,0)], marker_color=["#2ecc71", "#e74c3c"])])
                bar_fig.update_layout(title="عدد السجلات حسب الفئة المتنبأ بها", xaxis_title="الفئة", yaxis_title="العدد")
                st.plotly_chart(bar_fig, use_container_width=True)

                # جدول أعلى الحالات خطراً مع إمكانية تصفية حسب العتبة
                st.markdown("### أعلى الحالات خطراً")
                top_n = st.number_input("أعرض أعلى N", min_value=5, max_value=100, value=10, step=1)
                top_df = out[out["high_risk"]].sort_values("risk_percent", ascending=False).head(int(top_n))
                st.dataframe(top_df, use_container_width=True)

                buf = StringIO()
                out.to_csv(buf, index=False)
                st.download_button("⬇️ تنزيل النتائج (CSV)", data=buf.getvalue(), file_name="heart_batch_results.csv", mime="text/csv")
        except Exception as e:
            st.exception(e)

# صفحة حول
elif st.session_state.nav == "حول":
    st.header("ℹ️ حول التطبيق")
    # اجلب الدقة ديناميكياً كما في الصفحة الرئيسية
    last_acc = None
    mm = st.session_state.get("model_metrics")
    if not isinstance(mm, dict):
        try:
            possible_paths = [
                Path(__file__).parent / "heart_model.pkl",
                Path.cwd() / "heart_model.pkl",
            ]
            for p in possible_paths:
                if p.exists():
                    obj = joblib.load(str(p))
                    if isinstance(obj, dict) and 'metrics' in obj:
                        st.session_state.model_metrics = obj.get('metrics')
                        mm = st.session_state.model_metrics
                    break
        except Exception:
            pass
    if isinstance(mm, dict):
        last_acc = mm.get("accuracy")
    acc_txt = f"{last_acc*100:.1f}%" if isinstance(last_acc, (int, float)) else "—"

    st.markdown(
        f"""
        هذا تطبيق لتقدير خطر أمراض القلب اعتماداً على خصائص سريرية عددها 13، وتم تدريب النموذج واختباره بدقة تقارب {acc_txt} على مجموعة الاختبار وفقاً لتجارب التدريب المرفقة.
        - **التقنيات**: Streamlit، scikit-learn، Plotly، NumPy، Pandas.
        """
    )

    st.markdown("### 👥 الإشراف والفريق")
    c1, c2 = st.columns(2)
    with c2:
        st.markdown(
            """
            <div class='card fade-in' style='text-align:center'>
                <h4>👨‍🏫 المشرف</h4>
                <p><b>د. صفوان الشيباني</b></p>
            </div>
            """, unsafe_allow_html=True)
    with c1:
        st.markdown(
            """
            <div class='card fade-in' style='text-align:center'>
                <h4>👩‍💻 أعضاء الفريق</h4>
                <ul style="list-style:none; padding:0; margin:0">
                    <p><b>آية النويرة</b></p>
                    <p><b>رحاب بنقة</b></p>
                    <p><b>غيداء مقبولي</b></p>
                </ul>
            </div>
            """, unsafe_allow_html=True)
