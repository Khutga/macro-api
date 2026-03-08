"""
AnalyzerService - İstatistiksel Analiz Motoru

Pandas ve NumPy kullanarak zaman serisi analizi yapar.
PHP backend'den gelen veri setleri üzerinde çalışır.

Desteklenen Analizler:
- Korelasyon (Pearson & Spearman)
- Trend analizi (Lineer regresyon)
- Tanımlayıcı istatistikler
- Karşılaştırma (iki seri arası)
- Hareketli ortalama
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional


class AnalyzerService:
    """Ekonomik veri analiz servisi"""

    # =========================================
    # VERİ HAZIRLAMA
    # =========================================

    def _to_dataframe(self, series_data: dict) -> pd.DataFrame:
        """
        PHP'den gelen seri verisini Pandas DataFrame'e çevirir
        
        Input format:
        {
            "indicator_id": 1,
            "name": "TÜFE",
            "data": [{"date": "2024-01-01", "value": 64.77}, ...]
        }
        """
        df = pd.DataFrame(series_data["data"])
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df = df.sort_values("date").reset_index(drop=True)
        df.set_index("date", inplace=True)
        return df

    def _align_series(self, series_a: dict, series_b: dict) -> tuple[pd.Series, pd.Series]:
        """
        İki seriyi aynı tarih ekseninde hizalar.
        Farklı frekanslardaki serileri ortak tarihlere getirir.
        """
        df_a = self._to_dataframe(series_a)
        df_b = self._to_dataframe(series_b)

        # Inner join: Sadece her iki seride de veri olan tarihler
        merged = pd.merge(
            df_a[["value"]].rename(columns={"value": "a"}),
            df_b[["value"]].rename(columns={"value": "b"}),
            left_index=True,
            right_index=True,
            how="inner",
        )

        return merged["a"], merged["b"]

    # =========================================
    # KORELASYON ANALİZİ
    # =========================================

    def correlation(self, series_a: dict, series_b: dict, params: dict = {}) -> dict:
        """
        İki gösterge arasındaki korelasyonu hesaplar.
        
        Params:
            lag (int): Gecikme (gün). Örn: Faiz kararının dövize 30 günlük etkisi.
        
        Returns:
            - Pearson korelasyon katsayısı (lineer ilişki)
            - Spearman korelasyon katsayısı (monoton ilişki)
            - p-value (istatistiksel anlamlılık)
            - Veri noktası sayısı
            - Yorum metni
        """
        lag = params.get("lag", 0)

        aligned_a, aligned_b = self._align_series(series_a, series_b)

        # Lag (gecikme) uygula
        if lag > 0:
            aligned_a = aligned_a.iloc[:-lag]
            aligned_b = aligned_b.iloc[lag:]
            # İndeksleri yeniden hizala
            aligned_a = aligned_a.reset_index(drop=True)
            aligned_b = aligned_b.reset_index(drop=True)

        if len(aligned_a) < 10:
            raise ValueError(f"Korelasyon için yetersiz veri noktası ({len(aligned_a)}). En az 10 gerekli.")

        # Pearson korelasyonu (lineer)
        pearson_r, pearson_p = stats.pearsonr(aligned_a, aligned_b)

        # Spearman korelasyonu (sıralama bazlı, non-lineer ilişkileri de yakalar)
        spearman_r, spearman_p = stats.spearmanr(aligned_a, aligned_b)

        # Yorum üret
        interpretation = self._interpret_correlation(pearson_r, pearson_p)

        return {
            "series_a": series_a.get("name", "Seri A"),
            "series_b": series_b.get("name", "Seri B"),
            "lag_days": lag,
            "data_points": len(aligned_a),
            "pearson": {
                "coefficient": round(float(pearson_r), 4),
                "p_value": round(float(pearson_p), 6),
                "significant": pearson_p < 0.05,
            },
            "spearman": {
                "coefficient": round(float(spearman_r), 4),
                "p_value": round(float(spearman_p), 6),
                "significant": spearman_p < 0.05,
            },
            "interpretation": interpretation,
        }

    def _interpret_correlation(self, r: float, p: float) -> dict:
        """Korelasyon katsayısını yorumlar"""
        abs_r = abs(r)

        if abs_r >= 0.8:
            strength = "Çok güçlü"
            strength_en = "Very strong"
        elif abs_r >= 0.6:
            strength = "Güçlü"
            strength_en = "Strong"
        elif abs_r >= 0.4:
            strength = "Orta"
            strength_en = "Moderate"
        elif abs_r >= 0.2:
            strength = "Zayıf"
            strength_en = "Weak"
        else:
            strength = "Çok zayıf / Yok"
            strength_en = "Very weak / None"

        direction = "pozitif" if r > 0 else "negatif"
        direction_en = "positive" if r > 0 else "negative"

        significant = p < 0.05

        return {
            "strength_tr": strength,
            "strength_en": strength_en,
            "direction_tr": direction,
            "direction_en": direction_en,
            "is_significant": significant,
            "summary_tr": f"{strength} {direction} ilişki" + (" (istatistiksel olarak anlamlı)" if significant else " (istatistiksel olarak anlamlı değil)"),
            "summary_en": f"{strength_en} {direction_en} relationship" + (" (statistically significant)" if significant else " (not statistically significant)"),
        }

    # =========================================
    # TREND ANALİZİ
    # =========================================

    def trend_analysis(self, series_data: dict, params: dict = {}) -> dict:
        """
        Lineer trend analizi yapar.
        
        Returns:
            - Eğim (slope): Birim zamandaki değişim
            - R² (determination coefficient): Trendin açıklayıcılığı
            - Trend yönü ve gücü
            - Son N dönemdeki ortalama değişim
        """
        df = self._to_dataframe(series_data)
        values = df["value"].values

        if len(values) < 5:
            raise ValueError("Trend analizi için en az 5 veri noktası gerekli")

        # X ekseni: gün sayısı (0'dan başlayarak)
        x = np.arange(len(values))

        # Lineer regresyon
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        r_squared = r_value ** 2

        # Son 3 ay ve son 1 yıl değişim
        recent_3m = self._period_change(df, months=3)
        recent_1y = self._period_change(df, months=12)

        # Volatilite (standart sapma / ortalama × 100)
        volatility = (values.std() / values.mean() * 100) if values.mean() != 0 else 0

        # Trend yönü
        if slope > 0:
            direction = "yükseliş" if abs(slope) > std_err else "hafif yükseliş"
        else:
            direction = "düşüş" if abs(slope) > std_err else "hafif düşüş"

        return {
            "series_name": series_data.get("name", ""),
            "data_points": len(values),
            "trend": {
                "slope": round(float(slope), 6),
                "intercept": round(float(intercept), 4),
                "r_squared": round(float(r_squared), 4),
                "p_value": round(float(p_value), 6),
                "std_error": round(float(std_err), 6),
                "direction_tr": direction,
                "significant": p_value < 0.05,
            },
            "recent_changes": {
                "last_3_months": recent_3m,
                "last_12_months": recent_1y,
            },
            "volatility_pct": round(float(volatility), 2),
            "current_value": round(float(values[-1]), 4),
            "period_high": round(float(values.max()), 4),
            "period_low": round(float(values.min()), 4),
        }

    def _period_change(self, df: pd.DataFrame, months: int) -> Optional[dict]:
        """Belirli bir dönemdeki değişimi hesaplar"""
        try:
            cutoff = df.index[-1] - pd.DateOffset(months=months)
            period_data = df[df.index >= cutoff]
            if len(period_data) < 2:
                return None

            start_val = period_data["value"].iloc[0]
            end_val = period_data["value"].iloc[-1]
            change = end_val - start_val
            pct_change = (change / start_val * 100) if start_val != 0 else 0

            return {
                "absolute_change": round(float(change), 4),
                "percent_change": round(float(pct_change), 2),
                "start_value": round(float(start_val), 4),
                "end_value": round(float(end_val), 4),
            }
        except (IndexError, KeyError):
            return None

    # =========================================
    # TANIMLAYICI İSTATİSTİKLER
    # =========================================

    def descriptive_stats(self, series_data: dict) -> dict:
        """
        Temel istatistiksel göstergeleri hesaplar.
        """
        df = self._to_dataframe(series_data)
        values = df["value"]

        return {
            "series_name": series_data.get("name", ""),
            "unit": series_data.get("unit", ""),
            "data_points": len(values),
            "date_range": {
                "start": str(df.index.min().date()),
                "end": str(df.index.max().date()),
            },
            "stats": {
                "mean": round(float(values.mean()), 4),
                "median": round(float(values.median()), 4),
                "std": round(float(values.std()), 4),
                "variance": round(float(values.var()), 4),
                "min": round(float(values.min()), 4),
                "max": round(float(values.max()), 4),
                "range": round(float(values.max() - values.min()), 4),
                "q25": round(float(values.quantile(0.25)), 4),
                "q75": round(float(values.quantile(0.75)), 4),
                "iqr": round(float(values.quantile(0.75) - values.quantile(0.25)), 4),
                "skewness": round(float(values.skew()), 4),
                "kurtosis": round(float(values.kurtosis()), 4),
            },
            "latest": {
                "value": round(float(values.iloc[-1]), 4),
                "date": str(df.index[-1].date()),
            },
        }

    # =========================================
    # KARŞILAŞTIRMA
    # =========================================

    def comparison(self, series_a: dict, series_b: dict, params: dict = {}) -> dict:
        """İki seriyi karşılaştırır"""
        stats_a = self.descriptive_stats(series_a)
        stats_b = self.descriptive_stats(series_b)
        corr = self.correlation(series_a, series_b, params)

        return {
            "series_a": stats_a,
            "series_b": stats_b,
            "correlation": corr,
        }

    # =========================================
    # HAREKETLİ ORTALAMA
    # =========================================

    def moving_average(self, series_data: dict, window: int = 30) -> dict:
        """
        Hareketli ortalama hesaplar.
        
        Params:
            window: Pencere boyutu (gün)
        """
        df = self._to_dataframe(series_data)

        # Basit hareketli ortalama (SMA)
        df["sma"] = df["value"].rolling(window=window).mean()

        # Üstel hareketli ortalama (EMA)
        df["ema"] = df["value"].ewm(span=window).mean()

        # NaN'ları temizle
        result_df = df.dropna()

        return {
            "series_name": series_data.get("name", ""),
            "window": window,
            "data_points": len(result_df),
            "data": [
                {
                    "date": str(idx.date()),
                    "value": round(float(row["value"]), 4),
                    "sma": round(float(row["sma"]), 4),
                    "ema": round(float(row["ema"]), 4),
                }
                for idx, row in result_df.iterrows()
            ],
        }