"""Growth Model Analyzer — Stage 6 with AIC: reproducible research application."""
import numpy as np
from io import BytesIO
import hashlib
import pandas as pd
import streamlit as st

from core.data_loader import DataValidationError, normalize, read_excel, read_pasted
from core.fitting import FitError, fit_model
from core.models import MODEL_NAMES
from core.reporting import comparison_table, prediction_table, diagnostic_table
from core.charts import growth_figure, residual_figure, residual_scatter_figure, criteria_figure, publication_png, residual_png
from core.verification import VerificationError, verify_proposition_1
from core.quality import quality_report
from core.export import report_json

st.set_page_config(page_title="Growth Model Analyzer", page_icon="📈", layout="wide")
st.title("Growth Model Analyzer")
# st.caption("Mean-Substituted Nonlinear Least Squares Regression | Versi 0.7 — Validasi, pelaporan, dan AIC")
st.caption("Mean-Substituted Nonlinear Least Squares Regression | Versi 0.7")
st.info("Unggah Excel atau paste data, jalankan estimasi, lalu periksa kurva, residual, dan hasil perbandingan model.")

with st.sidebar:
    st.header("Pengaturan input")
    fmt = st.radio("Format data", ["A — Individu (wide)", "B — Rata-rata"], index=0)
    source = st.radio("Sumber data", ["Paste dari Excel", "Upload Excel (.xlsx)"])

format_key = "A" if fmt.startswith("A") else "B"
with st.expander("Contoh format yang diterima", expanded=True):
    if format_key == "A":
        st.code("time\tIndividu_01\tIndividu_02\tIndividu_03\n1\t35\t37\t34\n2\t42\t44\t40\n3\t51\t54\t49", language="text")
        st.caption("Baris = waktu; kolom = individu. Sel individu boleh kosong.")
    else:
        st.code("time\tmean_weight\tn\n1\t35.33\t3\n2\t42.00\t3\n3\t51.33\t3", language="text")
        st.caption("n adalah jumlah individu yang membentuk rata-rata pada setiap waktu.")

frame = None
current_context = None
try:
    if source == "Paste dari Excel":
        default = ("time\tIndividu_01\tIndividu_02\tIndividu_03\n1\t35\t37\t34\n2\t42\t44\t40\n3\t51\t54\t49") if format_key == "A" else ("time\tmean_weight\tn\n1\t35.33\t3\n2\t42.00\t3\n3\t51.33\t3")
        text = st.text_area("Paste tabel berikut header-nya", value=default, height=190, key=f"paste_{format_key}")
        delimiter = st.selectbox("Pemisah kolom", ["Otomatis", "Tab", "Titik koma", "Koma"])
        delimiter_key = {"Otomatis": "auto", "Tab": "tab", "Titik koma": "semicolon", "Koma": "comma"}[delimiter]
        current_context = (format_key, source, delimiter_key, hashlib.sha256(text.encode("utf-8")).hexdigest())
        if st.button("Parse dan validasi", type="primary"):
            frame = read_pasted(text, delimiter_key)
    else:
        upload = st.file_uploader("Pilih file Excel .xlsx", type=["xlsx"])
        if upload is not None:
            try:
                book = pd.ExcelFile(BytesIO(upload.getvalue()), engine="calamine")
                sheet = st.selectbox("Pilih worksheet", book.sheet_names)
            except Exception as exc:
                raise DataValidationError(f"Tidak dapat membaca daftar worksheet: {exc}") from exc
            current_context = (format_key, source, sheet, hashlib.sha256(upload.getvalue()).hexdigest())
            if st.button("Baca dan validasi", type="primary"):
                frame = read_excel(BytesIO(upload.getvalue()), sheet_name=sheet)

    if frame is not None:
        result = normalize(frame, format_key)
        # Store only the validated result. A changed format/source clears the preview below.
        st.session_state["parsed_result"] = result
        st.session_state["parsed_context"] = current_context
        st.success("Data berhasil dibaca dan divalidasi.")
except DataValidationError as exc:
    st.session_state.pop("parsed_result", None)
    st.error(str(exc))

result = st.session_state.get("parsed_result")
if result is not None and st.session_state.get("parsed_context") == current_context:
    st.subheader("Pratinjau data")
    a, b, c = st.columns(3)
    a.metric("Titik waktu", len(result.mean))
    b.metric("Observasi individu", int(result.mean["n"].sum()))
    c.metric("Replikasi seimbang", "Ya" if result.is_balanced else "Tidak")
    for notice in result.warnings:
        st.warning(notice)
    if result.format_name == "A":
        st.markdown("**Data individu (wide)**")
        st.dataframe(result.raw, width='stretch', hide_index=True)
    st.markdown("**Data rata-rata untuk analisis**")
    st.dataframe(result.mean, width='stretch', hide_index=True)
    st.download_button(
        "Unduh hasil rata-rata (CSV)",
        data=result.mean.to_csv(index=False).encode("utf-8-sig"),
        file_name="mean_substituted.csv", mime="text/csv",
    )
    with st.expander("Pemeriksaan kualitas data", expanded=False):
        quality = quality_report(result)
        st.dataframe(quality.summary, hide_index=True, width='stretch')
        for notice in quality.warnings:
            st.warning(notice)
        st.dataframe(quality.by_time, hide_index=True, width='stretch')
        st.download_button(
            "Unduh laporan kualitas data (CSV)",
            data=quality.by_time.to_csv(index=False).encode("utf-8-sig"),
            file_name="kualitas_data.csv", mime="text/csv",
        )
    st.divider()
    st.subheader("Estimasi model pertumbuhan")
    model_choice = st.selectbox("Model", ["Semua model", *MODEL_NAMES])
    available = ["Rata-rata (tanpa bobot)", "Rata-rata berbobot Nᵢ"]
    if result.long is not None:
        available.append("Data individu lengkap")
    basis_choice = st.selectbox("Dasar estimasi", available,
                                index=0 if result.is_balanced else 1)
    basis = {"Rata-rata (tanpa bobot)": "mean",
             "Rata-rata berbobot Nᵢ": "weighted_mean",
             "Data individu lengkap": "raw"}[basis_choice]
    if not result.is_balanced and basis == "mean":
        st.warning("Replikasi tidak seimbang: mean tanpa bobot tidak dijamin ekuivalen dengan data individu. Gunakan bobot Nᵢ untuk mempertahankan objektif OLS.")
    st.caption("x₀ = rata-rata bobot pada waktu pertama; waktu digeser t' = t − t₀. "
               "Waktu pertama tidak dimasukkan dalam SSE karena merupakan kondisi awal tetap.")

    models = list(MODEL_NAMES) if model_choice == "Semua model" else [model_choice]
    if st.button("Jalankan estimasi", type="primary", key="fit_run"):
        results = []
        errors = {}
        with st.spinner("Mengestimasi parameter nonlinear…"):
            for name in models:
                try:
                    results.append(fit_model(result, name, basis))
                except FitError as exc:
                    errors[name] = str(exc)
        st.session_state["fit_results"] = results
        st.session_state["fit_errors"] = errors
        st.session_state["fit_context"] = (current_context, model_choice, basis)

    if st.session_state.get("fit_context") == (current_context, model_choice, basis):
        results = st.session_state.get("fit_results", [])
        errors = st.session_state.get("fit_errors", {})
        for name, msg in errors.items():
            st.error(f"{name}: {msg}")
        if results:
            st.success(f"{len(results)} model berhasil diestimasi.")
            st.subheader("Ringkasan hasil estimasi")
            summary = comparison_table(results)
            st.dataframe(summary, width='stretch', hide_index=True,
                         column_config={"K": st.column_config.NumberColumn(format="%.5f"),
                                        "a": st.column_config.NumberColumn(format="%.6f"),
                                        "r": st.column_config.NumberColumn(format="%.6f"),
                                        "RMSE mean": st.column_config.NumberColumn(format="%.4f"),
                                        "MAE mean": st.column_config.NumberColumn(format="%.4f"),
                                        "R² mean": st.column_config.NumberColumn(format="%.6f"),
                                        "SSE mean": st.column_config.NumberColumn(format="%.3f"),
                                        "AIC mean": st.column_config.NumberColumn(format="%.3f")})
            if len(results) > 1:
                best = min(results, key=lambda f: f.rmse)
                st.info(f"RMSE terkecil pada data rata-rata: {best.model}. "
                        "Perbandingan ini dilakukan pada data yang digunakan dalam analisis, "
                        "bukan jaminan akurasi prediksi di luar sampel. "
                        "Perhatikan pula pola residual dan plausibilitas parameter.")

            if len(results) > 1 and all(f.basis == "mean" for f in results):
                finite_aic = [f for f in results if np.isfinite(f.aic_mean)]
                if finite_aic:
                    best_aic = min(finite_aic, key=lambda f: f.aic_mean)
                    st.info(f"AIC terkecil (evaluasi data rata-rata): {best_aic.model} "
                            f"({best_aic.aic_mean:.3f}). AIC lebih kecil lebih disukai "
                            "di bawah asumsi Gaussian dengan varians konstan.")
            st.caption("Metrik pada tabel dihitung dari rata-rata per waktu setelah titik awal, "
                       "terlepas dari basis yang digunakan saat fitting. "
                       "Hal ini menjaga pembandingan antarmodel pada pengamatan yang sama.")
            show_raw = False
            if result.long is not None:
                show_raw = st.checkbox("Tampilkan titik individu pada grafik", value=True)
            tab_growth, tab_resid, tab_compare, tab_exports = st.tabs([
                "📈 Kurva pertumbuhan", "🔎 Diagnostik residual",
                "📊 Perbandingan model", "📥 Ekspor",
            ])
            with tab_growth:
                st.plotly_chart(growth_figure(result, results, show_raw), width='stretch')
                st.caption("Titik merah = mean setiap waktu; garis = hasil estimasi model. "
                           "Pilih/nonaktifkan kurva melalui legenda grafik.")
            with tab_resid:
                st.plotly_chart(residual_figure(results), width='stretch')
                st.plotly_chart(residual_scatter_figure(results), width='stretch')
                st.caption("Residual didefinisikan sebagai mean observasi dikurangi mean prediksi. "
                           "Pola melengkung, tren terhadap waktu, atau perubahan sebaran bisa "
                           "menandakan ketidakcocokan model atau heteroskedastisitas. "
                           "Grafik ini bersifat diagnostik, bukan uji hipotesis formal.")
                st.markdown("**Rincian residual setiap model**")
                model_resid = st.selectbox("Model untuk tabel residual", [f.model for f in results])
                chosen = next(f for f in results if f.model == model_resid)
                st.dataframe(diagnostic_table(chosen).table, width='stretch',
                             hide_index=True)
            with tab_compare:
                metric = st.selectbox("Metrik pembanding", ["RMSE", "MAE", "AIC"])
                st.plotly_chart(criteria_figure(results, metric), width='stretch')
                st.caption("RMSE, MAE dan AIC dihitung dari rata-rata setiap waktu setelah titik awal. "
                           "AIC = m·ln(SSE/m) + 2p (konstanta model-independen diabaikan), "
                           "dengan p=2 untuk Logistic/Gompertz dan p=3 untuk Richards. "
                           "Interpretasi AIC sebagai kriteria pemilihan model berlaku untuk "
                           "fitting rata-rata tanpa bobot dengan galat Gaussian ber-varians sama. "
                           "Untuk fitting berbobot/data individu, nilai ini hanyalah skor evaluasi "
                           "pada rata-rata, bukan AIC likelihood objektif fitting tersebut. "
                           "Hasil dalam sampel bukan ukuran kinerja di luar sampel.")
            with tab_exports:
                st.markdown("**Laporan reproduksibilitas**")
                st.download_button(
                    "Unduh laporan analisis (JSON)",
                    data=report_json(result, results),
                    file_name="laporan_analisis.json",
                    mime="application/json",
                )
                st.markdown("**Tabel hasil**")
                st.download_button("Unduh ringkasan parameter & evaluasi (CSV)",
                    data=summary.to_csv(index=False).encode("utf-8-sig"),
                    file_name="ringkasan_model.csv", mime="text/csv")
                predictions = prediction_table(results)
                st.download_button("Unduh observasi dan prediksi (CSV)",
                    data=predictions.to_csv(index=False).encode("utf-8-sig"),
                    file_name="prediksi_model.csv", mime="text/csv")
                all_residuals = pd.concat([diagnostic_table(f).table.assign(model=f.model)
                                           for f in results], ignore_index=True)
                st.download_button("Unduh tabel seluruh residual (CSV)",
                    data=all_residuals.to_csv(index=False).encode("utf-8-sig"),
                    file_name="residual_model.csv", mime="text/csv")
                st.divider()
                st.markdown("**Grafik**")
                st.download_button("Unduh grafik pertumbuhan interaktif (HTML)",
                    data=growth_figure(result, results, show_raw).to_html(
                        full_html=True, include_plotlyjs=True).encode("utf-8"),
                    file_name="kurva_pertumbuhan.html", mime="text/html")
                st.download_button("Unduh grafik pertumbuhan (PNG, 300 DPI)",
                    data=publication_png(result, results, show_individuals=show_raw),
                    file_name="kurva_pertumbuhan.png", mime="image/png")
                st.download_button("Unduh grafik residual (PNG, 300 DPI)",
                    data=residual_png(results), file_name="residual_pertumbuhan.png",
                    mime="image/png")


    # Tahap 5 is independent of the basis chosen for the ordinary dashboard fit.
    st.divider()
    st.subheader("Verifikasi")
    st.markdown("Menguji identitas **d₁(θ) = dᵥ(θ) + C** dan, bila replikasi seimbang, "
                "**d₁(θ) = N d₂(θ) + C** pada Logistic, Gompertz, atau Richards. "
                "Semua SSE dihitung setelah waktu awal (yang dipakai untuk menetapkan x₀).")
    if result.long is None:
        st.info("Input Format B hanya memuat rata-rata. Untuk menghitung SSE data lengkap, "
                "konstanta C, dan membandingkan estimasi individu, unggah Format A.")
    else:
        counts_fitted = result.mean["n"].to_numpy()[1:]
        balanced_fitted = len(set(counts_fitted.tolist())) == 1
        if balanced_fitted:
            st.success(f"Replikasi seimbang pada titik waktu yang di-fit: N = {counts_fitted[0]}. "
                       "Proposition 1 menjamin peminimum OLS data lengkap dan rata-rata sama.")
        else:
            st.warning("Replikasi tidak seimbang pada titik waktu yang di-fit. "
                       "OLS data lengkap ekuivalen dengan fitting rata-rata berbobot nᵢ, "
                       "bukan selalu dengan rata-rata tanpa bobot.")
        verify_models = st.multiselect("Model untuk diverifikasi", list(MODEL_NAMES), default=list(MODEL_NAMES))
        verification_context = (current_context, tuple(verify_models))
        if st.button("Jalankan verifikasi", type="primary", key="run_verification"):
            outputs = []
            problems = {}
            with st.spinner("Menghitung hasil dari data individu dan dua jenis rata-rata…"):
                for name in verify_models:
                    try:
                        outputs.append(verify_proposition_1(result, name))
                    except (VerificationError, FitError) as exc:
                        problems[name] = str(exc)
            st.session_state["verifications"] = outputs
            st.session_state["verification_errors"] = problems
            st.session_state["verification_context"] = verification_context
        if st.session_state.get("verification_context") == verification_context:
            for name, msg in st.session_state.get("verification_errors", {}).items():
                st.error(f"{name}: {msg}")
            verifications = st.session_state.get("verifications", [])
            if verifications:
                st.caption("Identitas SSE diperiksa secara numerik pada tiga vektor parameter "
                           "(hasil fitting raw, mean, dan weighted). Selisih parameter hasil "
                           "optimasi dapat tidak tepat nol akibat toleransi numerik atau minimum lokal.")
                for verification in verifications:
                    with st.expander(f"{verification.model} — hasil verifikasi", expanded=True):
                        st.markdown("**Perbandingan estimasi parameter**")
                        st.dataframe(verification.parameter_comparison, hide_index=True,
                                     width='stretch')
                        st.markdown("**Nilai fungsi objektif pada setiap hasil estimasi**")
                        st.dataframe(verification.summary, hide_index=True,
                                     width='stretch')
                        st.markdown("**Pemeriksaan identitas SSE**")
                        st.dataframe(verification.identity_checks, hide_index=True,
                                     width='stretch')
                        if not verification.is_balanced:
                            st.caption("Kolom d₁ − N d₂ − C dikosongkan karena nᵢ "
                                       "berbeda antarwaktu. Identitas berbobot tetap berlaku.")
                        else:
                            st.caption("Kolom d₁ − N d₂ − C harus mendekati nol "
                                       "(perbedaan floating-point diperbolehkan).")
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        pred = verification.prediction_comparison
                        fig.add_scatter(x=pred["time"], y=pred["mean_weight"], mode="markers",
                                        name="Data rata-rata", marker_color="#C83C44")
                        for label, key, color, dash in [
                            ("Data individu", "pred_raw", "#181818", "solid"),
                            ("Rata-rata", "pred_mean", "#2781C3", "dash"),
                            ("Rata-rata berbobot", "pred_weighted", "#CB9116", "dot")]:
                            fig.add_scatter(x=pred["time"], y=pred[key], mode="lines",
                                            name=label, line={"color": color, "dash": dash})
                        fig.update_layout(title=f"Kurva estimasi — {verification.model}",
                                          xaxis_title="Waktu", yaxis_title="Bobot",
                                          hovermode="x unified", legend_title="Dasar estimasi")
                        st.plotly_chart(fig, width='stretch')
                        st.download_button(
                            f"Unduh hasil verifikasi {verification.model} (CSV)",
                            data=verification.prediction_comparison.to_csv(index=False).encode("utf-8-sig"),
                            file_name=f"verifikasi_{verification.model.lower()}.csv",
                            mime="text/csv", key=f"export_verify_{verification.model}")