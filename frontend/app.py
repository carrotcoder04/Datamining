import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import os

API_URL = "http://localhost:8000"
CLUSTER_NAMES = {0: "🔴 Cụm 0 — Khách hàng Tiềm năng", 1: "🟢 Cụm 1 — Khách hàng Trung thành", 2: "🔵 Cụm 2 — Khách hàng Ít hoạt động"}
CLUSTER_COLORS = {"0": "#ef4444", "1": "#22c55e", "2": "#3b82f6"}

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide", page_icon="🎯")

# ── CSS tùy chỉnh ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155; border-radius: 12px;
    padding: 20px 24px; text-align: center;
}
.metric-card .label { color: #94a3b8; font-size: 13px; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; }
.metric-card .value { color: #f1f5f9; font-size: 32px; font-weight: 700; margin-top: 6px; }
.metric-card .delta { font-size: 12px; margin-top: 4px; }
.cluster-badge {
    display:inline-block; padding: 4px 14px; border-radius: 999px;
    font-size: 13px; font-weight: 600; margin: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for key, default in [("prediction_result", None), ("feedback_submitted", False)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 Customer Segmentation")
    st.markdown("Hệ thống Phân đoạn & Gợi ý Khách hàng")
    st.divider()

    st.markdown("### 📁 Nạp dữ liệu mới")
    uploaded_file = st.file_uploader("Upload CSV/Excel giao dịch", type=["csv", "xlsx"])
    if st.button("⬆️ Nạp lên hệ thống", use_container_width=True):
        if uploaded_file:
            files = {"file": (uploaded_file.name, uploaded_file, "text/csv")}
            res = requests.post(f"{API_URL}/upload", files=files)
            st.success("Upload thành công!") if res.status_code == 200 else st.error("Upload thất bại!")
        else:
            st.warning("Chọn file trước!")

    st.divider()
    st.caption("MLOps Retrain: **02:00 AM** định kỳ")
    st.caption("Auto-retrain khi óð **feedback sai cụm** (ngưỡng: 10)")
    st.caption("Model: **K-Means (k=3)** + Cosine CF")

# ── Helper: load data ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_cluster_data():
    path = "rfm_data_with_clusters.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["Cluster_Label"] = df["Cluster"].astype(str)
    df["Cluster_Name"] = df["Cluster"].map({0: "Tiềm năng", 1: "Trung thành", 2: "Ít hoạt động"})
    return df

# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📊 Dashboard Tổng Quan", "🔮 Dự đoán & Phản hồi", "🛍️ Gợi ý Sản phẩm"])

# ══ TAB 1: DASHBOARD ══════════════════════════════════════════════════════════
with tab1:
    df = load_cluster_data()

    if df is None:
        st.info("⚙️ Chưa có dữ liệu phân cụm. Vui lòng chạy `python core/preprocessing.py` rồi `python core/clustering.py`.")
        st.stop()

    # Metrics row
    total_customers = len(df)
    cluster_counts = df["Cluster"].value_counts()
    avg_monetary = df["Monetary"].mean()
    avg_recency = df["Recency"].mean()

    c0, c1, c2, c3 = st.columns(4)
    for col, label, value, color in [
        (c0, "Tổng Khách hàng", f"{total_customers:,}", "#6366f1"),
        (c1, "🔴 Tiềm năng", f"{cluster_counts.get(0, 0):,}", "#ef4444"),
        (c2, "🟢 Trung thành", f"{cluster_counts.get(1, 0):,}", "#22c55e"),
        (c3, "🔵 Ít hoạt động", f"{cluster_counts.get(2, 0):,}", "#3b82f6"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value" style="color:{color};">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Pie + 3D
    col_pie, col_3d = st.columns([1, 1.6])
    with col_pie:
        st.markdown("#### Phân bổ Phân khúc")
        pie_data = df["Cluster_Name"].value_counts().reset_index()
        pie_data.columns = ["Cluster", "Count"]
        fig_pie = px.pie(pie_data, values="Count", names="Cluster", hole=0.45,
                         color="Cluster",
                         color_discrete_map={"Tiềm năng": "#ef4444", "Trung thành": "#22c55e", "Ít hoạt động": "#3b82f6"})
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_3d:
        st.markdown("#### Không gian 3D RFM")
        sample = df.sample(min(2000, len(df))) if len(df) > 2000 else df
        fig_3d = px.scatter_3d(sample, x="Recency", y="Frequency", z="Monetary",
                                color="Cluster_Name", opacity=0.75,
                                color_discrete_map={"Tiềm năng": "#ef4444", "Trung thành": "#22c55e", "Ít hoạt động": "#3b82f6"},
                                labels={"Cluster_Name": "Phân khúc"})
        fig_3d.update_traces(marker=dict(size=3))
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30),
                              paper_bgcolor="rgba(0,0,0,0)", legend_title="Phân khúc")
        st.plotly_chart(fig_3d, use_container_width=True)

    # Row 3: Bar chart RFM trung bình theo cụm + Box plots
    st.markdown("#### Đặc điểm RFM trung bình theo Phân khúc")
    summary = df.groupby("Cluster_Name")[["Recency", "Frequency", "Monetary"]].mean().reset_index()

    col_r, col_f, col_m = st.columns(3)
    for col, metric, color, title in [
        (col_r, "Recency", "#ef4444", "📅 Recency (ngày — càng thấp càng tốt)"),
        (col_f, "Frequency", "#22c55e", "🔁 Frequency (lần mua)"),
        (col_m, "Monetary", "#3b82f6", "💰 Monetary ($ tổng chi tiêu)"),
    ]:
        fig = px.bar(summary, x="Cluster_Name", y=metric, color="Cluster_Name",
                     color_discrete_map={"Tiềm năng": "#ef4444", "Trung thành": "#22c55e", "Ít hoạt động": "#3b82f6"},
                     text_auto=".0f")
        fig.update_layout(title=title, showlegend=False, margin=dict(l=0, r=0, t=40, b=0),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis_title="", yaxis_title=metric)
        col.plotly_chart(fig, use_container_width=True)

# ══ TAB 2: PREDICT & FEEDBACK ════════════════════════════════════════════════
with tab2:
    st.markdown("### 🔮 Dự đoán Phân khúc Khách hàng")
    st.caption("Nhập chỉ số RFM của một khách hàng để hệ thống dự đoán họ thuộc nhóm nào.")

    with st.form("predict_form", clear_on_submit=False):
        col_id, col_r, col_f, col_m = st.columns([1.2, 1, 1, 1])
        c_id = col_id.text_input("Mã KH (Customer ID)", value="NEW_001")
        r_val = col_r.number_input("Recency (Ngày kể từ mua cuối)", min_value=0, value=30, help="Số ngày kể từ lần mua hàng gần nhất")
        f_val = col_f.number_input("Frequency (Số lần mua)", min_value=1, value=5, help="Tổng số lần mua hàng")
        m_val = col_m.number_input("Monetary (Tổng chi tiêu $)", min_value=0.0, value=500.0, help="Tổng giá trị đơn hàng")
        submitted = st.form_submit_button("🚀 Dự đoán ngay", use_container_width=True, type="primary")

    if submitted:
        try:
            res = requests.post(f"{API_URL}/predict", json={"Recency": r_val, "Frequency": f_val, "Monetary": m_val})
            if res.status_code == 200:
                data = res.json()
                if "error" in data:
                    st.error(data["error"])
                else:
                    cluster = data["Predicted_Cluster"]
                    st.session_state.prediction_result = {"customer_id": c_id, "recency": r_val, "frequency": f_val, "monetary": m_val, "predicted_cluster": cluster}
                    st.session_state.feedback_submitted = False
        except Exception as e:
            st.error(f"Lỗi kết nối API: {e}")

    if st.session_state.prediction_result and not st.session_state.feedback_submitted:
        pred = st.session_state.prediction_result
        c = pred["predicted_cluster"]
        color_map = {0: "#ef4444", 1: "#22c55e", 2: "#3b82f6"}
        desc_map = {0: "Khách hàng có tiềm năng cao — Cần chiến lược chăm sóc để tăng tần suất mua.", 
                    1: "Khách hàng trung thành, mua thường xuyên — Duy trì ưu đãi giữ chân.", 
                    2: "Khách hàng ít hoạt động — Cần chiến dịch tái kích hoạt."}
        
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{color_map[c]}22,{color_map[c]}11);
             border:1.5px solid {color_map[c]}; border-radius:12px; padding:20px; margin:16px 0;">
            <div style="font-size:22px;font-weight:700;color:{color_map[c]};">
                {CLUSTER_NAMES[c]}
            </div>
            <div style="color:#cbd5e1;margin-top:8px;">{desc_map[c]}</div>
            <div style="margin-top:12px;display:flex;gap:24px;">
                <span style="color:#94a3b8;">Recency: <b style="color:#f1f5f9;">{pred['recency']} ngày</b></span>
                <span style="color:#94a3b8;">Frequency: <b style="color:#f1f5f9;">{pred['frequency']} lần</b></span>
                <span style="color:#94a3b8;">Monetary: <b style="color:#f1f5f9;">${pred['monetary']:,.0f}</b></span>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### 🧑‍⚖️ Kết quả có chính xác không? (Human-in-the-Loop)")
        col_like, col_dislike = st.columns(2)
        with col_like:
            if st.button("👍 Chính xác", use_container_width=True, type="primary"):
                requests.post(f"{API_URL}/feedback", json={"customer_id": pred["customer_id"], "recency": pred["recency"], "frequency": pred["frequency"], "monetary": pred["monetary"], "predicted_cluster": c, "correct": True})
                st.session_state.feedback_submitted = True
                st.rerun()
        with col_dislike:
            corr = st.selectbox("Cụm đúng:", [0, 1, 2], format_func=lambda x: CLUSTER_NAMES[x])
            if st.button("👎 Báo sai cụm", use_container_width=True):
                requests.post(f"{API_URL}/feedback", json={"customer_id": pred["customer_id"], "recency": pred["recency"], "frequency": pred["frequency"], "monetary": pred["monetary"], "predicted_cluster": c, "correct": False, "user_suggested_cluster": corr})
                st.session_state.feedback_submitted = True
                st.rerun()

    elif st.session_state.feedback_submitted:
        st.success("✅ Phản hồi đã được ghi nhận. Cảm ơn bạn! Hệ thống sẽ học từ dữ liệu này.")

# ══ TAB 3: RECOMMENDATION ════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🛍️ Hệ thống Gợi ý Sản phẩm")

    sub1, sub2 = st.tabs(["👥 Collaborative Filtering", "🔗 Association Rules (FP-Growth)"])

    with sub1:
        st.markdown("""
        > **Cơ chế**: Tìm 20 khách hàng có hành vi mua hàng tương đồng nhất (Cosine Similarity trên ma trận mua hàng),
        > sau đó gợi ý sản phẩm mà họ đã mua nhưng khách hàng hiện tại **chưa từng mua**.
        """)
        rec_cid = st.text_input("Nhập CustomerID", value="17850", key="rec_cid",
                                 help="Ví dụ: 17850, 12348, 12349...")
        if st.button("🔍 Tìm gợi ý cá nhân hóa", use_container_width=True, key="btn_rec"):
            with st.spinner("Đang phân tích hành vi mua hàng..."):
                try:
                    res = requests.get(f"{API_URL}/recommend/{rec_cid}")
                    data = res.json()
                    if "error" in data:
                        st.error(f"❌ {data['error']}")
                    else:
                        c = data["cluster"]
                        color = ["#ef4444", "#22c55e", "#3b82f6"][c]
                        st.markdown(f"""
                        <div style="background:{color}18;border:1px solid {color};border-radius:10px;padding:14px;margin-bottom:12px;">
                            <b style="color:{color};">{CLUSTER_NAMES[c]}</b><br>
                            <span style="color:#94a3b8;font-size:13px;">
                                Phân tích <b>{data.get('similar_customers_used','?')}</b> KH tương đồng 
                                trong nhóm <b>{data['cluster_size']}</b> KH
                                {"&nbsp;⚠️ <i>Outlier — Cụm dự đoán bằng model</i>" if data.get('is_outlier_customer') else ""}
                            </span>
                        </div>""", unsafe_allow_html=True)

                        recs = data.get("recommendations", [])
                        if recs:
                            for item in recs:
                                score = item.get("similarity_score", 0)
                                bar = int(score / max(r.get("similarity_score",1) for r in recs) * 100) if recs else 0
                                st.markdown(f"""
                                <div style="background:#1e293b;border-radius:8px;padding:12px 16px;margin-bottom:8px;display:flex;align-items:center;gap:16px;">
                                    <span style="color:{color};font-weight:700;font-size:18px;min-width:28px;">#{item['rank']}</span>
                                    <div style="flex:1;">
                                        <div style="color:#f1f5f9;font-weight:600;">{item['product']}</div>
                                        <div style="background:#334155;border-radius:4px;height:6px;margin-top:6px;">
                                            <div style="background:{color};width:{bar}%;height:6px;border-radius:4px;"></div>
                                        </div>
                                    </div>
                                    <span style="color:#64748b;font-size:13px;">{score:.3f}</span>
                                </div>""", unsafe_allow_html=True)
                        else:
                            st.info("Không tìm được gợi ý. KH này có thể đã mua hầu hết sản phẩm trong cụm.")
                except Exception as e:
                    st.error(str(e))

    with sub2:
        st.markdown("""
        > **Cơ chế**: Khai thác 12,201 luật kết hợp dạng **{A, B, C} → {D, E}** từ lịch sử giao dịch (FP-Growth).
        > Quản lý chọn **một hoặc nhiều sản phẩm** ỗ giỏ hàng, hệ thống tỏ ng hợp các luật có liên quan
        > và gợi ý sản phẩm phù hợp nhất (xếp hạng theo **Lift Score**).
        """)

        # ── Tải danh sách sản phẩm (cache 10 phút) ──────────────────────────
        @st.cache_data(ttl=600)
        def load_product_list():
            if os.path.exists("clean_data.csv"):
                df_prod = pd.read_csv("clean_data.csv", usecols=["Description"])
                return sorted(df_prod["Description"].dropna().unique().tolist())
            return []

        all_products = load_product_list()

        col_select, col_btn = st.columns([4, 1])
        with col_select:
            if all_products:
                selected_products = st.multiselect(
                    label="🛒 Chọn sản phẩm trong giỏ hàng (có thể chọn nhiều):",
                    options=all_products,
                    placeholder="Bắt đầu gõ tên sản phẩm...",
                    key="basket_products"
                )
            else:
                # Fallback: nhập thủ công nếu chưa có clean_data.csv
                raw_input = st.text_area(
                    "Nhập tên sản phẩm (mỗi dòng 1 sản phẩm, viết HOA):",
                    value="WHITE HANGING HEART T-LIGHT HOLDER\nREGENCY CAKESTAND 3 TIER",
                    key="basket_products_raw"
                )
                selected_products = [p.strip() for p in raw_input.splitlines() if p.strip()]
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            top_n = st.number_input("Độ dài kết quả", min_value=1, max_value=20, value=5, key="basket_top_n")

        search_btn = st.button("🔍 Tìm sản phẩm mua kèm", use_container_width=True, key="btn_basket", type="primary")

        if search_btn:
            if not selected_products:
                st.warning("⚠️ Vui lòng chọn ít nhất 1 sản phẩm.")
            else:
                with st.spinner(f"Trị ng lợc {len(selected_products)} sản phẩm qua 12,201 luật kết hợp..."):
                    try:
                        res = requests.post(
                            f"{API_URL}/recommend_apriori_basket",
                            json={"products": selected_products, "top_n": int(top_n)}
                        )
                        data = res.json()

                        if "error" in data:
                            st.error(f"❌ {data['error']}")
                        else:
                            recs = data.get("recommendations", [])
                            total_matched = data.get("total_rules_matched", 0)

                            # Hiển thị giỏ hàng đầu vào
                            st.markdown("**🛒 Giỏ hàng đầu vào:**")
                            basket_html = " ".join(
                                f'<span class="cluster-badge" style="background:#1e293b;color:#6366f1;border:1px solid #6366f1;">{p}</span>'
                                for p in selected_products
                            )
                            st.markdown(basket_html, unsafe_allow_html=True)

                            st.markdown(f"""
                            <div style="background:#1e293b18;border:1px solid #334155;border-radius:10px;
                                        padding:12px 16px;margin:12px 0;color:#94a3b8;font-size:13px;">
                                📊 Tổng số luật kết hợp khớp: <b style="color:#f1f5f9;">{total_matched}</b>
                                &nbsp;&nbsp;│&nbsp;&nbsp;
                                Ố giỏ hàng <b style="color:#f1f5f9;">{len(selected_products)}</b> sản phẩm
                            </div>""", unsafe_allow_html=True)

                            if not recs:
                                st.warning("⚠️ Không có gợi ý nào phù hợp với giỏ hàng này.")
                            else:
                                st.markdown("**📦 Sản phẩm gợi ý mua kèm (sắp xếp theo Lift Score tổng hợp):**")
                                max_score = recs[0]["lift_score"] if recs else 1
                                for item in recs:
                                    bar_pct = int(item["lift_score"] / max_score * 100)
                                    st.markdown(f"""
                                    <div style="background:#1e293b;border-radius:8px;padding:12px 16px;margin-bottom:8px;">
                                        <div style="display:flex;align-items:center;gap:16px;">
                                            <span style="color:#6366f1;font-weight:700;font-size:18px;min-width:28px;">
                                                #{item['rank']}
                                            </span>
                                            <div style="flex:1;">
                                                <div style="color:#f1f5f9;font-weight:600;margin-bottom:6px;">
                                                    {item['product']}
                                                </div>
                                                <div style="background:#334155;border-radius:4px;height:6px;">
                                                    <div style="background:#6366f1;width:{bar_pct}%;height:6px;border-radius:4px;"></div>
                                                </div>
                                            </div>
                                            <span style="color:#64748b;font-size:12px;white-space:nowrap;">
                                                Lift: {item['lift_score']:.2f}
                                            </span>
                                        </div>
                                    </div>""", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Lỗi kết nối API: {e}")

