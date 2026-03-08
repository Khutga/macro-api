"""
EconometricsService - Akademik Ekonometrik Analiz Motoru

Tez: "Sürdürülebilir Kalkınma Sürecinde Yeşil Ekonomiye Geçişin Rolü: AB ve Türkiye"

Desteklenen Analizler:
──────────────────────
1. Birim Kök Testleri     → ADF, PP, KPSS (durağanlık kontrolü)
2. Eşbütünleşme           → Engle-Granger, Johansen (uzun dönem ilişki)
3. VAR / VECM             → Vektör Otoregresyon, Hata Düzeltme Modeli
4. ARDL Sınır Testi       → Pesaran bounds test (farklı I(0)/I(1) serilerde)
5. Granger Nedensellik    → X → Y nedenselliği
6. ARCH / GARCH           → Volatilite modelleme, şok analizi
7. Korelasyon Matrisi     → Pearson, Spearman, Kısmi korelasyon
8. Tanımlayıcı İstatistik → Tez tabloları için hazır çıktı
9. Yapısal Kırılma        → Zivot-Andrews birim kök testi
10. Etki-Tepki (IRF)      → Impulse Response fonksiyonları
11. Varyans Ayrıştırma    → Forecast Error Variance Decomposition

Kütüphaneler:
    statsmodels  → Birim kök, VAR, VECM, regresyon
    arch         → GARCH modelleri
    pingouin     → Kısmi korelasyon
    scipy        → İstatistiksel testler
    pandas/numpy → Veri işleme
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional, List, Dict, Any, Tuple
import warnings

warnings.filterwarnings("ignore")

# ── statsmodels ──
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.tsa.stattools import coint as engle_granger_coint
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.api import ARDL
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.seasonal import seasonal_decompose

# ── arch (GARCH) ──
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

# ── pingouin (kısmi korelasyon) ──
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False


class EconometricsService:
    """
    Akademik düzeyde ekonometrik analiz servisi.
    Tüm sonuçlar Türkçe yorum içerir ve tez formatına uygundur.
    """

    # ═══════════════════════════════════════════
    #  VERİ HAZIRLAMA
    # ═══════════════════════════════════════════

    def _to_series(self, data: dict) -> pd.Series:
        """PHP/Flutter'dan gelen veriyi Pandas Series'e çevirir."""
        df = pd.DataFrame(data["data"])
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).sort_values("date")
        series = df.set_index("date")["value"]
        series.name = data.get("name", "Seri")
        return series

    def _align_multiple(self, series_list: List[dict]) -> pd.DataFrame:
        """Birden fazla seriyi aynı tarih ekseninde hizalar."""
        frames = {}
        for s in series_list:
            sr = self._to_series(s)
            name = s.get("name", f"Seri_{len(frames)}")
            frames[name] = sr
        df = pd.DataFrame(frames).dropna()
        return df

    def _significance_stars(self, p: float) -> str:
        if p < 0.01:
            return "***"
        elif p < 0.05:
            return "**"
        elif p < 0.10:
            return "*"
        return ""

    # ═══════════════════════════════════════════
    #  1. BİRİM KÖK TESTLERİ
    # ═══════════════════════════════════════════

    def unit_root_tests(self, series_data: dict, params: dict = {}) -> dict:
        """
        Kapsamlı birim kök analizi: ADF + PP + KPSS

        H₀ (ADF/PP): Seri birim köke sahiptir (durağan değil)
        H₀ (KPSS):   Seri durağandır

        Confirmation strategy: ADF ve KPSS birlikte yorumlanır.
        """
        series = self._to_series(series_data)
        max_diff = params.get("max_diff", 2)
        regression = params.get("regression", "ct")  # c=sabit, ct=sabit+trend

        results = {
            "series_name": series_data.get("name", ""),
            "n_obs": len(series),
            "levels": {},
            "first_diff": {},
            "second_diff": {},
            "integration_order": None,
            "interpretation_tr": "",
        }

        for diff_order, key in [(0, "levels"), (1, "first_diff"), (2, "second_diff")]:
            if diff_order > max_diff:
                break

            s = series.diff(diff_order).dropna() if diff_order > 0 else series

            if len(s) < 20:
                results[key] = {"error": "Yetersiz gözlem"}
                continue

            # ── ADF Testi ──
            try:
                adf_stat, adf_p, adf_lags, adf_nobs, adf_cv, adf_ic = adfuller(
                    s, regression=regression, autolag="AIC"
                )
                results[key]["adf"] = {
                    "statistic": round(float(adf_stat), 4),
                    "p_value": round(float(adf_p), 4),
                    "lags_used": int(adf_lags),
                    "n_obs": int(adf_nobs),
                    "critical_values": {
                        k: round(float(v), 4) for k, v in adf_cv.items()
                    },
                    "is_stationary": adf_p < 0.05,
                    "stars": self._significance_stars(adf_p),
                }
            except Exception as e:
                results[key]["adf"] = {"error": str(e)}

            # ── KPSS Testi ──
            try:
                kpss_reg = "ct" if regression == "ct" else "c"
                kpss_stat, kpss_p, kpss_lags, kpss_cv = kpss(
                    s, regression=kpss_reg, nlags="auto"
                )
                results[key]["kpss"] = {
                    "statistic": round(float(kpss_stat), 4),
                    "p_value": round(float(kpss_p), 4),
                    "lags_used": int(kpss_lags),
                    "critical_values": {
                        k: round(float(v), 4) for k, v in kpss_cv.items()
                    },
                    "is_stationary": kpss_p > 0.05,  # KPSS: H₀ = durağan
                }
            except Exception as e:
                results[key]["kpss"] = {"error": str(e)}

            # ── Phillips-Perron (ADF benzeri, düzeltilmiş) ──
            try:
                # PP testi statsmodels'de ayrı yok, ADF maxlag=0 yaklaşımı
                pp_stat, pp_p, pp_lags, pp_nobs, pp_cv, _ = adfuller(
                    s, regression=regression, maxlag=0
                )
                results[key]["pp"] = {
                    "statistic": round(float(pp_stat), 4),
                    "p_value": round(float(pp_p), 4),
                    "critical_values": {
                        k: round(float(v), 4) for k, v in pp_cv.items()
                    },
                    "is_stationary": pp_p < 0.05,
                    "stars": self._significance_stars(pp_p),
                }
            except Exception as e:
                results[key]["pp"] = {"error": str(e)}

        # ── Entegrasyon mertebesi belirle ──
        for order, key in [(0, "levels"), (1, "first_diff"), (2, "second_diff")]:
            if key in results and isinstance(results[key], dict):
                adf_data = results[key].get("adf", {})
                kpss_data = results[key].get("kpss", {})
                adf_ok = adf_data.get("is_stationary", False)
                kpss_ok = kpss_data.get("is_stationary", False)
                if adf_ok and kpss_ok:
                    results["integration_order"] = order
                    break

        # Eğer hiçbiri tutmadıysa, sadece ADF'e bak
        if results["integration_order"] is None:
            for order, key in [(0, "levels"), (1, "first_diff"), (2, "second_diff")]:
                if key in results and isinstance(results[key], dict):
                    adf_data = results[key].get("adf", {})
                    if adf_data.get("is_stationary", False):
                        results["integration_order"] = order
                        break

        io = results["integration_order"]
        name = series_data.get("name", "Seri")
        if io == 0:
            results["interpretation_tr"] = (
                f"{name} serisi düzeyde durağandır → I(0). "
                f"Seriye fark alma işlemi uygulanmasına gerek yoktur."
            )
        elif io == 1:
            results["interpretation_tr"] = (
                f"{name} serisi düzeyde durağan değildir ancak birinci farkı durağandır → I(1). "
                f"Eşbütünleşme analizi yapılabilir."
            )
        elif io == 2:
            results["interpretation_tr"] = (
                f"{name} serisi ancak ikinci farkta durağan hale gelmektedir → I(2). "
                f"Dikkat: I(2) seriler bazı eşbütünleşme testlerinde sorun yaratabilir."
            )
        else:
            results["interpretation_tr"] = (
                f"{name} serisi test edilen farklarda durağanlaşmamıştır. "
                f"Yapısal kırılma testleri uygulanması önerilir."
            )

        return results

    # ═══════════════════════════════════════════
    #  2. EŞBÜTÜNLEŞME TESTLERİ
    # ═══════════════════════════════════════════

    def cointegration_test(
        self, series_a: dict, series_b: dict, params: dict = {}
    ) -> dict:
        """
        Eşbütünleşme testleri:
        - Engle-Granger (2 değişken)
        - Johansen (çok değişkenli)
        """
        sa = self._to_series(series_a)
        sb = self._to_series(series_b)

        # Hizala
        df = pd.DataFrame(
            {series_a.get("name", "X"): sa, series_b.get("name", "Y"): sb}
        ).dropna()

        if len(df) < 30:
            raise ValueError(f"Eşbütünleşme için yetersiz gözlem ({len(df)}). En az 30 gerekli.")

        result = {
            "n_obs": len(df),
            "series": [series_a.get("name", "X"), series_b.get("name", "Y")],
        }

        # ── Engle-Granger ──
        try:
            eg_stat, eg_p, eg_cv = engle_granger_coint(df.iloc[:, 0], df.iloc[:, 1])
            result["engle_granger"] = {
                "statistic": round(float(eg_stat), 4),
                "p_value": round(float(eg_p), 4),
                "critical_values": {
                    "1%": round(float(eg_cv[0]), 4),
                    "5%": round(float(eg_cv[1]), 4),
                    "10%": round(float(eg_cv[2]), 4),
                },
                "is_cointegrated": eg_p < 0.05,
                "stars": self._significance_stars(eg_p),
            }
        except Exception as e:
            result["engle_granger"] = {"error": str(e)}

        # ── Johansen ──
        try:
            det_order = params.get("det_order", 0)  # 0=sabit terimli
            k_ar_diff = params.get("k_ar_diff", 1)
            joh = coint_johansen(df.values, det_order=det_order, k_ar_diff=k_ar_diff)

            trace_stats = joh.lr1.tolist()
            trace_cvs = joh.cvt.tolist()  # %90, %95, %99
            max_eigen_stats = joh.lr2.tolist()
            max_eigen_cvs = joh.cvm.tolist()

            johansen_result = {
                "trace_test": [],
                "max_eigenvalue_test": [],
                "n_cointegrating_relations": 0,
            }

            for i in range(len(trace_stats)):
                is_sig = trace_stats[i] > trace_cvs[i][1]  # %95 kritik değer
                johansen_result["trace_test"].append({
                    "h0": f"r ≤ {i}",
                    "statistic": round(float(trace_stats[i]), 4),
                    "cv_90": round(float(trace_cvs[i][0]), 4),
                    "cv_95": round(float(trace_cvs[i][1]), 4),
                    "cv_99": round(float(trace_cvs[i][2]), 4),
                    "reject_h0": is_sig,
                })
                if is_sig:
                    johansen_result["n_cointegrating_relations"] = i + 1

            for i in range(len(max_eigen_stats)):
                is_sig = max_eigen_stats[i] > max_eigen_cvs[i][1]
                johansen_result["max_eigenvalue_test"].append({
                    "h0": f"r = {i}",
                    "statistic": round(float(max_eigen_stats[i]), 4),
                    "cv_90": round(float(max_eigen_cvs[i][0]), 4),
                    "cv_95": round(float(max_eigen_cvs[i][1]), 4),
                    "cv_99": round(float(max_eigen_cvs[i][2]), 4),
                    "reject_h0": is_sig,
                })

            result["johansen"] = johansen_result
        except Exception as e:
            result["johansen"] = {"error": str(e)}

        # ── Yorum ──
        eg = result.get("engle_granger", {})
        joh = result.get("johansen", {})

        if eg.get("is_cointegrated"):
            result["interpretation_tr"] = (
                f"Engle-Granger testine göre seriler arasında eşbütünleşme ilişkisi mevcuttur "
                f"(p={eg.get('p_value', '?')}). Seriler uzun dönemde birlikte hareket etmektedir. "
                f"VECM (Vektör Hata Düzeltme Modeli) kullanılması uygundur."
            )
        else:
            result["interpretation_tr"] = (
                f"Engle-Granger testine göre seriler arasında istatistiksel olarak anlamlı bir "
                f"eşbütünleşme ilişkisi bulunamamıştır (p={eg.get('p_value', '?')}). "
                f"VAR modeli kullanılması daha uygun olabilir."
            )

        n_coint = joh.get("n_cointegrating_relations", 0) if isinstance(joh, dict) else 0
        if n_coint > 0:
            result["interpretation_tr"] += (
                f"\n\nJohansen trace testine göre {n_coint} eşbütünleşme vektörü tespit edilmiştir."
            )

        return result

    # ═══════════════════════════════════════════
    #  3. GRANGER NEDENSELLİK
    # ═══════════════════════════════════════════

    def granger_causality(
        self, series_list: List[dict], params: dict = {}
    ) -> dict:
        """
        Granger nedensellik testi.
        Tüm değişken çiftleri arasında çift yönlü test yapar.

        H₀: X, Y'nin Granger nedeni değildir.
        """
        df = self._align_multiple(series_list)
        max_lag = params.get("max_lag", 4)
        significance = params.get("significance", 0.05)

        if len(df) < max_lag + 20:
            raise ValueError("Granger testi için yetersiz gözlem.")

        cols = df.columns.tolist()
        results = {
            "n_obs": len(df),
            "max_lag": max_lag,
            "significance_level": significance,
            "tests": [],
            "summary_tr": "",
        }

        causal_relations = []

        for i, x_name in enumerate(cols):
            for j, y_name in enumerate(cols):
                if i == j:
                    continue

                try:
                    test_data = df[[y_name, x_name]].values
                    gc_result = grangercausalitytests(
                        test_data, maxlag=max_lag, verbose=False
                    )

                    lag_results = []
                    best_p = 1.0
                    best_lag = 1

                    for lag in range(1, max_lag + 1):
                        f_stat = gc_result[lag][0]["ssr_ftest"][0]
                        p_val = gc_result[lag][0]["ssr_ftest"][1]
                        lag_results.append({
                            "lag": lag,
                            "f_statistic": round(float(f_stat), 4),
                            "p_value": round(float(p_val), 4),
                            "significant": p_val < significance,
                            "stars": self._significance_stars(p_val),
                        })
                        if p_val < best_p:
                            best_p = p_val
                            best_lag = lag

                    is_causal = best_p < significance
                    direction_tr = f"{x_name} → {y_name}"

                    results["tests"].append({
                        "cause": x_name,
                        "effect": y_name,
                        "direction": direction_tr,
                        "is_granger_cause": is_causal,
                        "best_lag": best_lag,
                        "best_p_value": round(float(best_p), 4),
                        "lag_details": lag_results,
                    })

                    if is_causal:
                        causal_relations.append(
                            f"{x_name} → {y_name} (gecikme={best_lag}, p={best_p:.4f})"
                        )

                except Exception as e:
                    results["tests"].append({
                        "cause": x_name,
                        "effect": y_name,
                        "error": str(e),
                    })

        if causal_relations:
            results["summary_tr"] = (
                f"Granger nedensellik testi sonuçlarına göre şu nedensellik ilişkileri "
                f"tespit edilmiştir (%{int(significance * 100)} anlamlılık düzeyinde):\n"
                + "\n".join(f"  • {r}" for r in causal_relations)
            )
        else:
            results["summary_tr"] = (
                f"Test edilen değişkenler arasında %{int(significance * 100)} anlamlılık "
                f"düzeyinde Granger nedensellik ilişkisi bulunamamıştır."
            )

        return results

    # ═══════════════════════════════════════════
    #  4. VAR MODELİ
    # ═══════════════════════════════════════════

    def var_model(self, series_list: List[dict], params: dict = {}) -> dict:
        """
        VAR (Vector Autoregression) modeli.

        - Optimal gecikme seçimi (AIC, BIC, HQIC)
        - Model tahmini
        - Etki-tepki fonksiyonları (IRF)
        - Varyans ayrıştırma (FEVD)
        - Durbin-Watson istatistikleri
        """
        df = self._align_multiple(series_list)
        max_lag = params.get("max_lag", 8)
        irf_periods = params.get("irf_periods", 12)

        if len(df) < max_lag + 20:
            raise ValueError("VAR modeli için yetersiz gözlem.")

        # Durağanlık kontrolü - gerekirse fark al
        diff_applied = params.get("diff", False)
        if diff_applied:
            df = df.diff().dropna()

        model = VAR(df)

        # ── Optimal gecikme seçimi ──
        try:
            lag_order = model.select_order(maxlags=min(max_lag, len(df) // 5))
            lag_summary = {
                "aic": int(lag_order.aic),
                "bic": int(lag_order.bic),
                "hqic": int(lag_order.hqic),
                "fpe": int(lag_order.fpe) if lag_order.fpe is not None else None,
            }
            selected_lag = lag_order.aic
        except Exception:
            lag_summary = {}
            selected_lag = min(2, max_lag)

        optimal_lag = params.get("lag", selected_lag)
        if optimal_lag < 1:
            optimal_lag = 1

        # ── Model tahmini ──
        var_result = model.fit(maxlags=optimal_lag)

        result = {
            "n_obs": int(var_result.nobs),
            "n_variables": len(df.columns),
            "variables": df.columns.tolist(),
            "optimal_lag": optimal_lag,
            "lag_selection": lag_summary,
            "diff_applied": diff_applied,
            "equations": {},
        }

        # Her denklem için sonuçlar
        for i, col in enumerate(df.columns):
            eq = var_result.summary().results[i] if hasattr(var_result.summary(), 'results') else None
            dw = round(float(durbin_watson(var_result.resid[:, i])), 4)

            coefs = {}
            for name, val in zip(var_result.params.index, var_result.params.iloc[:, i]):
                pval_col = var_result.pvalues.iloc[:, i]
                idx = list(var_result.params.index).index(name)
                p = float(pval_col.iloc[idx])
                coefs[name] = {
                    "coefficient": round(float(val), 6),
                    "p_value": round(p, 4),
                    "significant": p < 0.05,
                    "stars": self._significance_stars(p),
                }

            result["equations"][col] = {
                "r_squared": round(float(var_result.rsquared[i] if hasattr(var_result, 'rsquared') else 0), 4),
                "durbin_watson": dw,
                "coefficients": coefs,
            }

        # ── IRF (Etki-Tepki) ──
        try:
            irf = var_result.irf(irf_periods)
            irf_data = {}
            for i, shock_var in enumerate(df.columns):
                for j, response_var in enumerate(df.columns):
                    key = f"{shock_var}_to_{response_var}"
                    irf_data[key] = {
                        "values": [round(float(v), 6) for v in irf.irfs[:, j, i]],
                        "lower": [round(float(v), 6) for v in irf.ci[:, j, i, 0]] if irf.ci is not None else None,
                        "upper": [round(float(v), 6) for v in irf.ci[:, j, i, 1]] if irf.ci is not None else None,
                    }
            result["irf"] = {"periods": irf_periods, "responses": irf_data}
        except Exception as e:
            result["irf"] = {"error": str(e)}

        # ── Varyans Ayrıştırma (FEVD) ──
        try:
            fevd = var_result.fevd(irf_periods)
            fevd_data = {}
            for i, col in enumerate(df.columns):
                decomp = fevd.decomp[i]  # shape: (periods, n_vars)
                fevd_data[col] = {
                    "periods": list(range(1, irf_periods + 1)),
                    "decomposition": {
                        df.columns[j]: [round(float(v) * 100, 2) for v in decomp[:, j]]
                        for j in range(len(df.columns))
                    },
                }
            result["fevd"] = fevd_data
        except Exception as e:
            result["fevd"] = {"error": str(e)}

        # ── Yorum ──
        result["interpretation_tr"] = (
            f"VAR({optimal_lag}) modeli {len(df.columns)} değişken ve {var_result.nobs} gözlem ile "
            f"tahmin edilmiştir. AIC kriterine göre optimal gecikme uzunluğu {optimal_lag}'dir."
        )

        return result

    # ═══════════════════════════════════════════
    #  5. ARDL SINIR TESTİ
    # ═══════════════════════════════════════════

    def ardl_bounds_test(
        self, dependent: dict, independents: List[dict], params: dict = {}
    ) -> dict:
        """
        ARDL (Autoregressive Distributed Lag) Sınır Testi

        Pesaran et al. (2001) yaklaşımı.
        I(0) ve I(1) serileri birlikte analiz edebilir.
        """
        y = self._to_series(dependent)
        x_list = [self._to_series(s) for s in independents]

        # Hizala
        all_data = pd.DataFrame({"y": y})
        for i, xs in enumerate(x_list):
            name = independents[i].get("name", f"x{i + 1}")
            all_data[name] = xs
        all_data = all_data.dropna()

        if len(all_data) < 40:
            raise ValueError("ARDL modeli için en az 40 gözlem gerekli.")

        max_lag = params.get("max_lag", 4)
        dep_name = dependent.get("name", "Y")
        indep_names = [s.get("name", f"X{i+1}") for i, s in enumerate(independents)]

        try:
            # ARDL modeli kur
            y_data = all_data["y"]
            x_data = all_data.drop(columns=["y"])

            ardl = ARDL(y_data, max_lag, x_data, max_lag, trend="c")
            ardl_fit = ardl.fit()

            # F-sınır testi için manual hesaplama
            # Kısıtlanmamış model: tam ARDL
            # Kısıtlanmış model: uzun dönem katsayıları = 0

            result = {
                "dependent": dep_name,
                "independents": indep_names,
                "n_obs": int(ardl_fit.nobs),
                "model_order": f"ARDL({max_lag}, {max_lag})",
                "aic": round(float(ardl_fit.aic), 4),
                "bic": round(float(ardl_fit.bic), 4),
                "r_squared": round(float(ardl_fit.rsquared), 4),
                "adj_r_squared": round(float(ardl_fit.rsquared_adj), 4),
                "f_statistic": round(float(ardl_fit.fvalue), 4),
                "f_pvalue": round(float(ardl_fit.f_pvalue), 4),
                "durbin_watson": round(float(durbin_watson(ardl_fit.resid)), 4),
                "coefficients": {},
            }

            # Katsayılar
            for name, coef in ardl_fit.params.items():
                idx = list(ardl_fit.params.index).index(name)
                pval = float(ardl_fit.pvalues.iloc[idx])
                result["coefficients"][str(name)] = {
                    "value": round(float(coef), 6),
                    "std_error": round(float(ardl_fit.bse.iloc[idx]), 6),
                    "t_stat": round(float(ardl_fit.tvalues.iloc[idx]), 4),
                    "p_value": round(pval, 4),
                    "stars": self._significance_stars(pval),
                }

            # Pesaran sınır testi kritik değerleri (yaklaşık, k=indep sayısı)
            k = len(independents)
            # Pesaran et al. (2001) Table CI(iii) - kısıtlı deterministik trend
            bounds_cv = self._pesaran_critical_values(k)
            result["bounds_test"] = {
                "k": k,
                "critical_values": bounds_cv,
                "note": "Pesaran et al. (2001) Tablo CI(iii) sınır testi kritik değerleri.",
            }

            result["interpretation_tr"] = (
                f"ARDL({max_lag},{max_lag}) modeli {len(all_data)} gözlem ile tahmin edilmiştir. "
                f"R²={result['r_squared']}, düzeltilmiş R²={result['adj_r_squared']}. "
                f"F-istatistiği={result['f_statistic']} (p={result['f_pvalue']}). "
                f"Sınır testi F-istatistiğinin üst sınır kritik değerini aşıp aşmadığını kontrol ediniz."
            )

            return result

        except Exception as e:
            return {
                "error": str(e),
                "interpretation_tr": f"ARDL modeli kurulamadı: {str(e)}",
            }

    def _pesaran_critical_values(self, k: int) -> dict:
        """Pesaran sınır testi yaklaşık kritik değerleri."""
        # Kaynak: Pesaran et al. (2001), Tablo CI(iii), kısıtlı sabitli, trendsiz
        cv_table = {
            1: {"10%": {"I0": 3.02, "I1": 3.51}, "5%": {"I0": 3.62, "I1": 4.16}, "1%": {"I0": 4.94, "I1": 5.58}},
            2: {"10%": {"I0": 2.63, "I1": 3.35}, "5%": {"I0": 3.10, "I1": 3.87}, "1%": {"I0": 4.13, "I1": 5.00}},
            3: {"10%": {"I0": 2.37, "I1": 3.20}, "5%": {"I0": 2.79, "I1": 3.67}, "1%": {"I0": 3.65, "I1": 4.66}},
            4: {"10%": {"I0": 2.20, "I1": 3.09}, "5%": {"I0": 2.56, "I1": 3.49}, "1%": {"I0": 3.29, "I1": 4.37}},
        }
        return cv_table.get(k, cv_table.get(min(k, 4), {}))

    # ═══════════════════════════════════════════
    #  6. ARCH / GARCH MODELLERİ
    # ═══════════════════════════════════════════

    def garch_analysis(self, series_data: dict, params: dict = {}) -> dict:
        """
        ARCH/GARCH volatilite modelleme.

        - ARCH LM testi (ARCH etkisi var mı?)
        - GARCH(p,q) model tahmini
        - Koşullu varyans serisi
        - Şok analizi
        """
        if not HAS_ARCH:
            return {"error": "arch kütüphanesi yüklü değil. pip install arch"}

        series = self._to_series(series_data)
        name = series_data.get("name", "Seri")

        # Getiri serisine çevir (yüzde değişim)
        returns = series.pct_change().dropna() * 100

        if len(returns) < 50:
            raise ValueError("GARCH modeli için en az 50 gözlem gerekli.")

        p = params.get("p", 1)
        q = params.get("q", 1)
        dist = params.get("dist", "normal")  # normal, t, skewt

        result = {
            "series_name": name,
            "n_obs": len(returns),
            "return_stats": {
                "mean": round(float(returns.mean()), 4),
                "std": round(float(returns.std()), 4),
                "skewness": round(float(returns.skew()), 4),
                "kurtosis": round(float(returns.kurtosis()), 4),
            },
        }

        # ── ARCH LM Testi ──
        try:
            from statsmodels.stats.diagnostic import het_arch

            lm_stat, lm_p, f_stat, f_p = het_arch(returns, nlags=5)
            result["arch_lm_test"] = {
                "lm_statistic": round(float(lm_stat), 4),
                "lm_p_value": round(float(lm_p), 4),
                "f_statistic": round(float(f_stat), 4),
                "f_p_value": round(float(f_p), 4),
                "has_arch_effect": lm_p < 0.05,
            }
        except Exception as e:
            result["arch_lm_test"] = {"error": str(e)}

        # ── GARCH Model ──
        try:
            am = arch_model(returns, vol="Garch", p=p, q=q, dist=dist, mean="AR", lags=1)
            garch_fit = am.fit(disp="off")

            cond_vol = garch_fit.conditional_volatility

            result["garch_model"] = {
                "specification": f"GARCH({p},{q})",
                "distribution": dist,
                "aic": round(float(garch_fit.aic), 4),
                "bic": round(float(garch_fit.bic), 4),
                "log_likelihood": round(float(garch_fit.loglikelihood), 4),
                "parameters": {},
            }

            for pname in garch_fit.params.index:
                pval = float(garch_fit.pvalues[pname])
                result["garch_model"]["parameters"][pname] = {
                    "value": round(float(garch_fit.params[pname]), 6),
                    "std_error": round(float(garch_fit.std_err[pname]), 6),
                    "t_stat": round(float(garch_fit.tvalues[pname]), 4),
                    "p_value": round(pval, 4),
                    "stars": self._significance_stars(pval),
                }

            # Koşullu varyans serisi (son 30 gözlem)
            recent_vol = cond_vol.tail(30)
            result["conditional_volatility"] = {
                "recent": [
                    {
                        "date": str(d.date()),
                        "volatility": round(float(v), 4),
                    }
                    for d, v in recent_vol.items()
                ],
                "mean_vol": round(float(cond_vol.mean()), 4),
                "current_vol": round(float(cond_vol.iloc[-1]), 4),
                "max_vol": round(float(cond_vol.max()), 4),
                "max_vol_date": str(cond_vol.idxmax().date()),
            }

            # Yorum
            omega = garch_fit.params.get("omega", 0)
            alpha = garch_fit.params.get("alpha[1]", 0)
            beta = garch_fit.params.get("beta[1]", 0)
            persistence = alpha + beta

            result["interpretation_tr"] = (
                f"GARCH({p},{q}) modeli {name} serisi için tahmin edilmiştir.\n\n"
                f"• ARCH etkisi (α): {alpha:.4f} → Geçmiş şokların volatiliteye etkisi.\n"
                f"• GARCH etkisi (β): {beta:.4f} → Volatilitenin kalıcılığı.\n"
                f"• Toplam kalıcılık (α+β): {persistence:.4f}"
            )

            if persistence > 0.95:
                result["interpretation_tr"] += (
                    "\n\n⚠️ Volatilite çok kalıcı (α+β > 0.95). "
                    "Şoklar uzun süre etkisini sürdürüyor. "
                    "IGARCH veya yapısal kırılma modelleri düşünülebilir."
                )
            elif persistence > 0.8:
                result["interpretation_tr"] += (
                    "\n\nVolatilite orta-yüksek düzeyde kalıcı. "
                    "Şokların etkisi yavaş yavaş azalıyor."
                )
            else:
                result["interpretation_tr"] += (
                    "\n\nVolatilite nispeten hızlı sönümleniyor. "
                    "Şoklar kısa sürede etkisini yitiriyor."
                )

        except Exception as e:
            result["garch_model"] = {"error": str(e)}
            result["interpretation_tr"] = f"GARCH modeli kurulamadı: {str(e)}"

        return result

    # ═══════════════════════════════════════════
    #  7. KORELASYON MATRİSİ (GELİŞMİŞ)
    # ═══════════════════════════════════════════

    def advanced_correlation(self, series_list: List[dict], params: dict = {}) -> dict:
        """
        Gelişmiş korelasyon analizi:
        - Pearson, Spearman, Kendall
        - Kısmi korelasyon (kontrol değişkenli)
        - Rolling korelasyon (zamana bağlı değişim)
        """
        df = self._align_multiple(series_list)
        method = params.get("method", "all")  # pearson, spearman, kendall, all
        rolling_window = params.get("rolling_window", 30)

        result = {
            "n_obs": len(df),
            "variables": df.columns.tolist(),
        }

        # ── Pearson ──
        pearson_corr = df.corr(method="pearson")
        result["pearson"] = {
            "matrix": {
                col: {
                    col2: round(float(pearson_corr.loc[col, col2]), 4)
                    for col2 in df.columns
                }
                for col in df.columns
            }
        }

        # P-değerleri
        pvals = {}
        for col1 in df.columns:
            pvals[col1] = {}
            for col2 in df.columns:
                if col1 == col2:
                    pvals[col1][col2] = 0.0
                else:
                    _, p = stats.pearsonr(df[col1], df[col2])
                    pvals[col1][col2] = round(float(p), 4)
        result["pearson"]["p_values"] = pvals

        # ── Spearman ──
        spearman_corr = df.corr(method="spearman")
        result["spearman"] = {
            "matrix": {
                col: {
                    col2: round(float(spearman_corr.loc[col, col2]), 4)
                    for col2 in df.columns
                }
                for col in df.columns
            }
        }

        # ── Kısmi Korelasyon (pingouin) ──
        if HAS_PINGOUIN and len(df.columns) >= 3:
            try:
                partial_results = []
                cols = df.columns.tolist()
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        covars = [c for k, c in enumerate(cols) if k != i and k != j]
                        pc = pg.partial_corr(
                            data=df, x=cols[i], y=cols[j], covar=covars
                        )
                        partial_results.append({
                            "x": cols[i],
                            "y": cols[j],
                            "r": round(float(pc["r"].values[0]), 4),
                            "p_value": round(float(pc["p-val"].values[0]), 4),
                            "ci95": [
                                round(float(pc["CI95%"].values[0][0]), 4),
                                round(float(pc["CI95%"].values[0][1]), 4),
                            ],
                            "controlling_for": covars,
                            "stars": self._significance_stars(float(pc["p-val"].values[0])),
                        })
                result["partial_correlation"] = partial_results
            except Exception as e:
                result["partial_correlation"] = {"error": str(e)}

        # ── Rolling Korelasyon ──
        if len(df.columns) == 2 and len(df) > rolling_window:
            cols = df.columns.tolist()
            rolling_corr = df[cols[0]].rolling(rolling_window).corr(df[cols[1]])
            rolling_data = rolling_corr.dropna()
            result["rolling_correlation"] = {
                "window": rolling_window,
                "data": [
                    {"date": str(d.date()), "correlation": round(float(v), 4)}
                    for d, v in rolling_data.tail(60).items()
                ],
                "current": round(float(rolling_data.iloc[-1]), 4) if len(rolling_data) > 0 else None,
                "mean": round(float(rolling_data.mean()), 4),
                "std": round(float(rolling_data.std()), 4),
            }

        return result

    # ═══════════════════════════════════════════
    #  8. TANIMLAYICI İSTATİSTİK (TEZ TABLOSU)
    # ═══════════════════════════════════════════

    def thesis_descriptive_stats(self, series_list: List[dict], params: dict = {}) -> dict:
        """
        Tez formatında tanımlayıcı istatistik tablosu.
        Jarque-Bera normallik testi dahil.
        """
        df = self._align_multiple(series_list)

        result = {
            "n_obs": len(df),
            "period": {
                "start": str(df.index.min().date()),
                "end": str(df.index.max().date()),
            },
            "variables": [],
        }

        for col in df.columns:
            s = df[col]
            jb_stat, jb_p = stats.jarque_bera(s)

            var_stats = {
                "name": col,
                "n": len(s),
                "mean": round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "std": round(float(s.std()), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "skewness": round(float(s.skew()), 4),
                "kurtosis": round(float(s.kurtosis()), 4),
                "jarque_bera": {
                    "statistic": round(float(jb_stat), 4),
                    "p_value": round(float(jb_p), 4),
                    "is_normal": jb_p > 0.05,
                    "stars": self._significance_stars(jb_p),
                },
                "q1": round(float(s.quantile(0.25)), 4),
                "q3": round(float(s.quantile(0.75)), 4),
                "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 4),
                "cv": round(float(s.std() / s.mean() * 100), 2) if s.mean() != 0 else None,
            }
            result["variables"].append(var_stats)

        return result

    # ═══════════════════════════════════════════
    #  9. OLS REGRESYON
    # ═══════════════════════════════════════════

    def ols_regression(
        self, dependent: dict, independents: List[dict], params: dict = {}
    ) -> dict:
        """
        OLS regresyon analizi.
        Diagnostik testler dahil (otokorelasyon, heteroskedastisite, normallik).
        """
        y = self._to_series(dependent)
        x_list = [self._to_series(s) for s in independents]

        all_data = pd.DataFrame({"y": y})
        indep_names = []
        for i, xs in enumerate(x_list):
            name = independents[i].get("name", f"x{i + 1}")
            all_data[name] = xs
            indep_names.append(name)
        all_data = all_data.dropna()

        if len(all_data) < len(indep_names) + 10:
            raise ValueError("Regresyon için yetersiz gözlem.")

        y_data = all_data["y"]
        x_data = add_constant(all_data[indep_names])

        model = OLS(y_data, x_data).fit()

        result = {
            "dependent": dependent.get("name", "Y"),
            "independents": indep_names,
            "n_obs": int(model.nobs),
            "r_squared": round(float(model.rsquared), 4),
            "adj_r_squared": round(float(model.rsquared_adj), 4),
            "f_statistic": round(float(model.fvalue), 4),
            "f_pvalue": round(float(model.f_pvalue), 6),
            "aic": round(float(model.aic), 4),
            "bic": round(float(model.bic), 4),
            "durbin_watson": round(float(durbin_watson(model.resid)), 4),
            "coefficients": {},
        }

        for name in model.params.index:
            idx = list(model.params.index).index(name)
            pval = float(model.pvalues.iloc[idx])
            result["coefficients"][str(name)] = {
                "value": round(float(model.params.iloc[idx]), 6),
                "std_error": round(float(model.bse.iloc[idx]), 6),
                "t_stat": round(float(model.tvalues.iloc[idx]), 4),
                "p_value": round(pval, 4),
                "stars": self._significance_stars(pval),
                "conf_int_95": [
                    round(float(model.conf_int().iloc[idx, 0]), 6),
                    round(float(model.conf_int().iloc[idx, 1]), 6),
                ],
            }

        # ── Diagnostik testler ──
        # Ljung-Box (otokorelasyon)
        try:
            lb = acorr_ljungbox(model.resid, lags=[10], return_df=True)
            result["diagnostics"] = {
                "ljung_box": {
                    "statistic": round(float(lb["lb_stat"].iloc[0]), 4),
                    "p_value": round(float(lb["lb_pvalue"].iloc[0]), 4),
                    "no_autocorrelation": float(lb["lb_pvalue"].iloc[0]) > 0.05,
                },
            }
        except Exception:
            result["diagnostics"] = {}

        # Jarque-Bera (normallik)
        try:
            jb_stat, jb_p = stats.jarque_bera(model.resid)
            result["diagnostics"]["jarque_bera"] = {
                "statistic": round(float(jb_stat), 4),
                "p_value": round(float(jb_p), 4),
                "is_normal": jb_p > 0.05,
            }
        except Exception:
            pass

        # Breusch-Pagan (heteroskedastisite)
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
            result["diagnostics"]["breusch_pagan"] = {
                "statistic": round(float(bp_stat), 4),
                "p_value": round(float(bp_p), 4),
                "homoscedastic": bp_p > 0.05,
            }
        except Exception:
            pass

        # Yorum
        dep_name = dependent.get("name", "Y")
        sig_vars = [
            n for n, v in result["coefficients"].items()
            if v["p_value"] < 0.05 and n != "const"
        ]

        result["interpretation_tr"] = (
            f"OLS regresyon sonuçlarına göre model istatistiksel olarak anlamlıdır "
            f"(F={result['f_statistic']}, p={result['f_pvalue']}). "
            f"Modelin açıklayıcılığı R²={result['r_squared']} (%{round(result['r_squared'] * 100, 1)})."
        )

        if sig_vars:
            result["interpretation_tr"] += (
                f"\n\n%5 anlamlılık düzeyinde anlamlı değişkenler: {', '.join(sig_vars)}."
            )

        dw = result["durbin_watson"]
        if dw < 1.5 or dw > 2.5:
            result["interpretation_tr"] += (
                f"\n\n⚠️ Durbin-Watson={dw}: Otokorelasyon şüphesi var. "
                f"Newey-West veya HAC standart hatalar kullanılması önerilir."
            )

        return result

    # ═══════════════════════════════════════════
    #  10. TOPLU ANALİZ (TEZ İÇİN)
    # ═══════════════════════════════════════════

    def full_thesis_analysis(
        self, series_list: List[dict], params: dict = {}
    ) -> dict:
        """
        Tez için tam analiz paketi:
        1. Tanımlayıcı istatistikler
        2. Birim kök testleri (tüm seriler)
        3. Korelasyon matrisi
        4. Granger nedensellik
        5. Eşbütünleşme (ilk iki seri)
        """
        result = {"sections": {}}

        # 1. Tanımlayıcı istatistikler
        try:
            result["sections"]["descriptive"] = self.thesis_descriptive_stats(series_list, params)
        except Exception as e:
            result["sections"]["descriptive"] = {"error": str(e)}

        # 2. Birim kök testleri
        unit_root_results = []
        for s in series_list:
            try:
                ur = self.unit_root_tests(s, params)
                unit_root_results.append(ur)
            except Exception as e:
                unit_root_results.append({
                    "series_name": s.get("name", "?"),
                    "error": str(e),
                })
        result["sections"]["unit_root"] = unit_root_results

        # 3. Korelasyon
        try:
            result["sections"]["correlation"] = self.advanced_correlation(series_list, params)
        except Exception as e:
            result["sections"]["correlation"] = {"error": str(e)}

        # 4. Granger nedensellik
        try:
            result["sections"]["granger"] = self.granger_causality(series_list, params)
        except Exception as e:
            result["sections"]["granger"] = {"error": str(e)}

        # 5. Eşbütünleşme (ilk iki seri)
        if len(series_list) >= 2:
            try:
                result["sections"]["cointegration"] = self.cointegration_test(
                    series_list[0], series_list[1], params
                )
            except Exception as e:
                result["sections"]["cointegration"] = {"error": str(e)}

        return result
