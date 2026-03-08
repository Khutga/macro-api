"""
Makroekonomik Dashboard - Python Analiz Mikroservisi v2.0

Yeni: Akademik ekonometrik analiz desteği (Tez modülü)

Endpoints:
──────────
POST /analyze              → Mevcut analiz (korelasyon, trend, istatistik)
POST /econometrics         → Ekonometrik analiz router'ı
GET  /econometrics/methods → Desteklenen ekonometrik yöntemler listesi
GET  /health               → Servis sağlık kontrolü

Ekonometrik Yöntemler:
    unit_root         → ADF, PP, KPSS birim kök testleri
    cointegration     → Engle-Granger, Johansen eşbütünleşme
    granger           → Granger nedensellik testi
    var_model         → VAR modeli + IRF + FEVD
    ardl              → ARDL sınır testi
    garch             → ARCH/GARCH volatilite modeli
    correlation       → Gelişmiş korelasyon (kısmi dahil)
    descriptive       → Tez tablosu formatında tanımlayıcı istatistik
    ols               → OLS regresyon + diagnostik testler
    full_analysis     → Tam tez analiz paketi

Çalıştırma:
    pip install -r requirements.txt
    uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from services.analyzer import AnalyzerService
from services.econometrics import EconometricsService

# =========================================
# UYGULAMA AYARLARI
# =========================================

app = FastAPI(
    title="Macro Dashboard - Analiz & Ekonometri Servisi",
    description="Ekonomik veri analizi + Akademik ekonometrik analiz (tez modülü)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = AnalyzerService()
econometrics = EconometricsService()


# =========================================
# MODELLER
# =========================================

class DataPoint(BaseModel):
    date: str
    value: float


class SeriesInput(BaseModel):
    indicator_id: int
    name: str
    code: str = ""
    unit: str = ""
    data: list[DataPoint]


class AnalyzeRequest(BaseModel):
    type: str
    indicator_ids: list[int]
    period: str = "5y"
    series_data: list[SeriesInput]
    params: Optional[dict] = {}


class EconometricsRequest(BaseModel):
    """Ekonometrik analiz isteği."""
    method: str          # unit_root, cointegration, granger, var_model, ardl, garch, correlation, descriptive, ols, full_analysis
    series_data: list[SeriesInput]
    params: Optional[dict] = {}
    # ARDL / OLS için bağımlı değişken indeksi (series_data listesindeki sıra)
    dependent_index: Optional[int] = 0


# =========================================
# MEVCUT ENDPOINTS (v1 uyumluluk)
# =========================================

@app.get("/health")
def health_check():
    """Servis sağlık kontrolü"""
    libs = {
        "statsmodels": True,
        "scipy": True,
        "pandas": True,
        "numpy": True,
    }
    try:
        from arch import arch_model
        libs["arch"] = True
    except ImportError:
        libs["arch"] = False
    try:
        import pingouin
        libs["pingouin"] = True
    except ImportError:
        libs["pingouin"] = False

    return {
        "status": "ok",
        "service": "macro-dashboard-analyzer",
        "version": "2.0.0",
        "features": ["analysis", "econometrics"],
        "libraries": libs,
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    """Mevcut analiz endpoint'i (v1 uyumluluk)."""
    try:
        series_list = [s.model_dump() for s in request.series_data]
        result = match_analysis_type(
            analysis_type=request.type,
            series_data=series_list,
            params=request.params or {},
        )
        return {"success": True, "analysis_type": request.type, "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")


# =========================================
# EKONOMETRİ ENDPOINTS (v2)
# =========================================

@app.get("/econometrics/methods")
def econometrics_methods():
    """Desteklenen ekonometrik yöntemler listesi."""
    return {
        "methods": [
            {
                "id": "unit_root",
                "name_tr": "Birim Kök Testleri",
                "name_en": "Unit Root Tests",
                "description": "ADF, PP, KPSS — Durağanlık kontrolü",
                "min_series": 1,
                "max_series": 1,
            },
            {
                "id": "cointegration",
                "name_tr": "Eşbütünleşme Testleri",
                "name_en": "Cointegration Tests",
                "description": "Engle-Granger, Johansen — Uzun dönem ilişki",
                "min_series": 2,
                "max_series": 2,
            },
            {
                "id": "granger",
                "name_tr": "Granger Nedensellik",
                "name_en": "Granger Causality",
                "description": "Çift yönlü Granger nedensellik testi",
                "min_series": 2,
                "max_series": 5,
            },
            {
                "id": "var_model",
                "name_tr": "VAR Modeli",
                "name_en": "Vector Autoregression",
                "description": "VAR + IRF + Varyans Ayrıştırma",
                "min_series": 2,
                "max_series": 5,
            },
            {
                "id": "ardl",
                "name_tr": "ARDL Sınır Testi",
                "name_en": "ARDL Bounds Test",
                "description": "Pesaran sınır testi — I(0)/I(1) karışık serilerde",
                "min_series": 2,
                "max_series": 5,
            },
            {
                "id": "garch",
                "name_tr": "ARCH/GARCH",
                "name_en": "GARCH Volatility Model",
                "description": "Volatilite modelleme ve şok analizi",
                "min_series": 1,
                "max_series": 1,
            },
            {
                "id": "correlation",
                "name_tr": "Gelişmiş Korelasyon",
                "name_en": "Advanced Correlation",
                "description": "Pearson, Spearman, Kısmi korelasyon",
                "min_series": 2,
                "max_series": 10,
            },
            {
                "id": "descriptive",
                "name_tr": "Tanımlayıcı İstatistik",
                "name_en": "Descriptive Statistics",
                "description": "Tez tablosu formatında istatistikler",
                "min_series": 1,
                "max_series": 10,
            },
            {
                "id": "ols",
                "name_tr": "OLS Regresyon",
                "name_en": "OLS Regression",
                "description": "Regresyon + diagnostik testler",
                "min_series": 2,
                "max_series": 5,
            },
            {
                "id": "full_analysis",
                "name_tr": "Tam Tez Analizi",
                "name_en": "Full Thesis Analysis",
                "description": "Tanımlayıcı + Birim kök + Korelasyon + Granger + Eşbütünleşme",
                "min_series": 2,
                "max_series": 5,
            },
        ]
    }


@app.post("/econometrics")
def run_econometrics(request: EconometricsRequest):
    """
    Ana ekonometrik analiz endpoint'i.
    
    method parametresine göre doğru analizi çağırır.
    """
    try:
        series_list = [s.model_dump() for s in request.series_data]
        params = request.params or {}
        method = request.method
        dep_idx = request.dependent_index or 0

        result = match_econometrics_method(
            method=method,
            series_data=series_list,
            params=params,
            dependent_index=dep_idx,
        )

        return {
            "success": True,
            "method": method,
            "result": result,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ekonometrik analiz hatası: {str(e)}",
        )


# =========================================
# ROUTER FONKSİYONLARI
# =========================================

def match_analysis_type(analysis_type: str, series_data: list, params: dict) -> dict:
    """Mevcut analiz router'ı (v1)."""
    match analysis_type:
        case "correlation":
            if len(series_data) < 2:
                raise ValueError("Korelasyon için en az 2 gösterge gerekli")
            return analyzer.correlation(series_data[0], series_data[1], params)
        case "trend":
            return analyzer.trend_analysis(series_data[0], params)
        case "statistics":
            results = [analyzer.descriptive_stats(s) for s in series_data]
            return results if len(results) > 1 else results[0]
        case "comparison":
            if len(series_data) < 2:
                raise ValueError("Karşılaştırma için en az 2 gösterge gerekli")
            return analyzer.comparison(series_data[0], series_data[1], params)
        case "moving_avg":
            window = params.get("window", 30)
            return analyzer.moving_average(series_data[0], window=window)
        case _:
            raise ValueError(f"Desteklenmeyen analiz tipi: {analysis_type}")


def match_econometrics_method(
    method: str, series_data: list, params: dict, dependent_index: int = 0
) -> dict:
    """Ekonometrik analiz router'ı (v2)."""

    match method:
        # ── Tek seri analizleri ──
        case "unit_root":
            if len(series_data) == 1:
                return econometrics.unit_root_tests(series_data[0], params)
            # Birden fazla seri varsa hepsini test et
            results = []
            for s in series_data:
                results.append(econometrics.unit_root_tests(s, params))
            return {"series_results": results}

        case "garch":
            return econometrics.garch_analysis(series_data[0], params)

        # ── İki seri analizleri ──
        case "cointegration":
            if len(series_data) < 2:
                raise ValueError("Eşbütünleşme için en az 2 seri gerekli")
            return econometrics.cointegration_test(
                series_data[0], series_data[1], params
            )

        # ── Çok seri analizleri ──
        case "granger":
            if len(series_data) < 2:
                raise ValueError("Granger nedensellik için en az 2 seri gerekli")
            return econometrics.granger_causality(series_data, params)

        case "var_model":
            if len(series_data) < 2:
                raise ValueError("VAR modeli için en az 2 seri gerekli")
            return econometrics.var_model(series_data, params)

        case "correlation":
            if len(series_data) < 2:
                raise ValueError("Korelasyon için en az 2 seri gerekli")
            return econometrics.advanced_correlation(series_data, params)

        case "descriptive":
            return econometrics.thesis_descriptive_stats(series_data, params)

        # ── Bağımlı/bağımsız ayrımı olan analizler ──
        case "ardl":
            if len(series_data) < 2:
                raise ValueError("ARDL için en az 2 seri gerekli (1 bağımlı + 1 bağımsız)")
            dependent = series_data[dependent_index]
            independents = [s for i, s in enumerate(series_data) if i != dependent_index]
            return econometrics.ardl_bounds_test(dependent, independents, params)

        case "ols":
            if len(series_data) < 2:
                raise ValueError("Regresyon için en az 2 seri gerekli")
            dependent = series_data[dependent_index]
            independents = [s for i, s in enumerate(series_data) if i != dependent_index]
            return econometrics.ols_regression(dependent, independents, params)

        # ── Tam analiz ──
        case "full_analysis":
            if len(series_data) < 2:
                raise ValueError("Tam analiz için en az 2 seri gerekli")
            return econometrics.full_thesis_analysis(series_data, params)

        case _:
            raise ValueError(
                f"Desteklenmeyen ekonometrik yöntem: '{method}'. "
                f"Kullanılabilir yöntemler: unit_root, cointegration, granger, "
                f"var_model, ardl, garch, correlation, descriptive, ols, full_analysis"
            )


# =========================================
# ANA ÇALIŞTIRMA
# =========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
