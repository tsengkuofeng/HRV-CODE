import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
import json
import traceback

# ============================================================================
# 基本參數
# ============================================================================
TARGET_FPS = 30.0
DETREND_LAMBDA = 15.0

# 【新增 1】訊號品質指標 (SQI) - 5秒滑動視窗
SQI_WINDOW_SIZE = 150  # 5秒 @ 30fps

# 品質門檻拆成兩道，因為 BPM 和 HRV 對訊號品質的需求差了一個量級。
#
# BPM 是從頻譜的主峰讀的：只要脈動的週期性還在，振幅小一點、雜訊多一點
# 都還抓得到。HRV 要的是「每一拍的時間點」，峰值往旁邊偏 20ms
# 就直接進 RMSSD —— 它需要的訊號品質嚴格得多。
#
# 原本兩者共用 0.20 這一道：品質一掉，連心率都跟著回 0，
# 畫面上心率直接跳成 --，而使用者手指其實好好地放著、波形也還在跑。
# 這兩個門檻是在新的 SQI 量表上定的（見 calculate_sqi 的權重）。
# 合成訊號跑 24 個 (振幅 x 雜訊) 網格點，新 SQI 的分界非常乾淨：
#     可用的最低 0.52 ／ 不可用的最高 0.50
# 取 0.50 當門檻時，24 個點裡誤放行 0 個、誤擋掉 0 個。
SQI_NO_FINGER = 0.30   # 低於此：連週期性都沒有，當作沒有手指
SQI_THRESHOLD = 0.50   # 低於此：這一段訊號的數字全部不可信

# 這個視窗的拍數要夠多，RMSSD 才算得出來。
# 看的是「還剩幾對時間上相鄰的好拍」，不是「剔掉了幾成」——
# 剔除比例對「整段都在輕微抖動」完全沒反應，而那正是最常見的情況。
MIN_PAIRS_IN_WIN = 4
MAX_REJECT_RATIO = 0.50   # 單一視窗自己要報數字時的上限

# 峰值定位抖動的上限（毫秒）。超過就不報 RMSSD。
#
# RMSSD^2 = 真值^2 + 6*sigma^2 —— sigma 20ms 就會憑空加上 49ms 的假變異，
# 比訊號本身還大。合成測試把 sigma 跟實際誤差排在一起看，界線非常乾淨：
#     sigma  5~14ms → 報出來的 RMSSD 誤差在 ±7ms 內
#     sigma 20~25ms → 誤差 +18 ~ +33ms（真值 28~35 被報成 48~61）
# 對教練來說，「RMSSD 61」比「暫時沒有數字」危險得多。
#
# SDNN 不套這道：它只被 2*sigma^2 灌大，同樣的 sigma 下誤差小得多，
# 實測 sigma 25ms 時 SDNN 誤差仍在 16ms 以內。
POOL_JITTER_MAX = 20.0

# 演算法版本。改過清洗規則之後新舊數值不能直接比較，
# 存進紀錄才分得出來哪些是舊算法算的。
ENGINE_VER = 3

# 帶通上限 = BAND_HI_MULT x 基頻（原本寫死 3.0 Hz）。
# HR 100 的基頻 1.67 Hz，二次諧波 3.33 Hz 剛好被 3.0 切掉 ——
# 濾完只剩近乎純弦波，而弦波的頂點最平，峰值定位誤差最大。
# 合成訊號實測：上限 3.0 -> 定位 sigma 29.3ms；跟著心率走 -> 17.8ms。
BAND_HI_MULT = 3.5
BAND_HI_MIN = 3.0        # 不低於原本的固定值，避免低心率反而變窄
MM_W = 3                 # moving mean 視窗（原本 5）

# 綠／藍低於紅的這個比例，就當作「手指壓著鏡頭的純紅畫面」，
# 改用紅光通道算脈搏。實測手指貼緊時 G/R 大約 0.00~0.02，
# 沒貼緊或對著環境時 G/R 通常 0.3 以上，中間空很大。
CONTACT_GB_RATIO = 0.12

# 去偏所需的最少 RR 筆數。12 秒視窗只有約 20 筆，估出來的 sigma
# 跨次標準差高達 12ms，拿去修正只會讓即時數字亂跳 ——
# 所以即時視窗不修正，只有整段分析才修。
DEBIAS_MIN_RR = 30
DEBIAS_MIN_SIGMA = 4.0   # sigma 小於這個值時修正沒有意義


def moving_mean(x, w):
    """移動平均，邊界用端點值延伸。

    原本是 np.convolve(x, kernel, 'same') —— 那個 'same' 在頭尾補的是 0，
    不是訊號。w=3 時第一格會變成 (0 + x0 + x1)/3，也就是真值的三分之二：
    R 的直流大約 211，第一格就掉到 140.7，等於憑空插進一根 −70 的尖峰。

    正規化之後那根尖峰是 0.333，而脈搏本身的標準差只有 0.016（相差 20 倍）。
    它再經過帶通濾波會擴散成前 0.5 秒一整段的暫態，把頻譜的低頻端墊高；
    脈動振幅小的時候，頻譜主峰就被拉到搜尋範圍最低的那一格（0.75Hz），
    心率固定報 45.3 bpm 且完全不隨訊號改變 ——
    又是一個「波形看起來正常、數字卻是錯的」。

    合成訊號實測（真值 104 bpm，振幅由大到小）：
        修好前  振幅 8 以上正確，6 以下一律 45.3
        修好後  振幅 1 都還報 104.0
    三個通道都會被影響，所以 POS 那一路也一樣。
    """
    x = np.asarray(x, dtype=np.float64)
    if w < 2 or len(x) < w:
        return x
    pad = w // 2
    xx = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    return np.convolve(xx, np.ones(w) / float(w), 'valid')[:len(x)]


def smoothness_priors_detrend(z):
    """Tarvainen 平滑先驗法去趨勢。"""
    n = len(z)
    if n < 6:
        return signal.detrend(z)
    try:
        D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
        I = sparse.eye(n)
        A = (I + (DETREND_LAMBDA ** 2) * (D.T @ D)).tocsc()
        trend = spsolve(A, z)
        return z - trend
    except Exception:
        return signal.detrend(z)


# ============================================================================
# 【新增 2】訊號品質指標 (Signal Quality Index - SQI)
# 基於頻域能量集中度的 5 秒滑動窗口檢測
# ============================================================================
def calculate_sqi(signal_data, fs=30.0, window_size=None):
    """
    多規則訊號品質指標 (Multi-rule SQI)。
    四條一起看，任何一條爛掉都算訊號不好：

      振幅範圍       — 沒把手指蓋上去的時候，max-min 幾乎是 0
      頻域能量集中度 — 心跳那個頻段佔總能量的比例
      自相關係數     — 取 lag≈1 拍；沒手指的時候訊號近乎隨機，自相關拉不起來
      前後半段一致性 — 手在動（移動偽影）會讓前半段跟後半段對不起來

    返回值：SQI (0-1)
    """
    if window_size is None:
        # 5 秒，跟著實際取樣率算。寫死 150 格的話，15fps 的機器
        # 要 10 秒才拿得到第一個 SQI，而在那之前一律回 0 = 訊號不足。
        window_size = max(60, int(5.0 * fs))

    n = len(signal_data)
    if n < window_size:
        window_size = n            # 資料還不夠 5 秒就用現有的，不要直接放棄
    if window_size < 45:
        return 0.0

    try:
        recent = np.asarray(signal_data[-window_size:], dtype=np.float64)

        # ── Rule 1: 振幅範圍（無手指 → <5，正常接觸 → >10）──────────────────
        amp_range = float(np.max(recent) - np.min(recent))
        rule1 = float(np.clip(amp_range / 10.0, 0.0, 1.0))  # 手機鏡頭 AC 振幅通常 2–10，以 10 為滿分

        # ── Rule 2: 頻域能量集中度 ──────────────────────────────────────────
        freqs, psd = signal.welch(recent, fs=fs, nperseg=min(256, len(recent)))
        hr_mask = (freqs >= 0.75) & (freqs <= 2.5)
        total_e = np.sum(psd)
        rule2 = float(np.sum(psd[hr_mask]) / total_e) if total_e > 0 else 0.0

        # ── Rule 5: 週期性（在合理心率範圍內取最大自相關）──────────────────
        #
        # 原本 lag 寫死 0.75 秒 —— 那等於假設心率剛好 80 bpm。
        # 心率不是 80 的人會被無故扣分，而且扣得很重：
        #     HR  55 → rule5 0.00      HR  80 → rule5 0.87
        #     HR 102 → rule5 0.20      HR 120 → rule5 0.00
        # 因為 0.75 秒對他們而言落在心動週期的反相位上，自相關是負的。
        # 這是在懲罰「心率不等於 80」，不是在衡量訊號品質。
        #
        # 改成掃描 40–150 bpm 對應的 lag 取最大值：只要有週期性就拿高分，
        # 不管那個週期是多長。實測各心率都落在 0.81–0.92。
        z = (recent - np.mean(recent))
        denom = float(np.dot(z, z))
        best_ac = 0.0
        if denom > 0:
            lag_lo = max(1, int(fs * 60.0 / 150.0))
            lag_hi = min(len(z) - 2, int(fs * 60.0 / 40.0))
            for lag in range(lag_lo, max(lag_lo + 1, lag_hi)):
                ac = float(np.dot(z[:-lag], z[lag:])) / denom
                if ac > best_ac:
                    best_ac = ac
        rule5 = float(np.clip((best_ac + 0.2) / 1.2, 0.0, 1.0))

        # ── Rule 6: 前後半段 RMS 一致性 ────────────────────────────────────
        half = len(recent) // 2
        rms1 = float(np.sqrt(np.mean(recent[:half] ** 2)))
        rms2 = float(np.sqrt(np.mean(recent[half:] ** 2)))
        if rms1 + rms2 > 0:
            rule6 = 1.0 - abs(rms1 - rms2) / (rms1 + rms2)
        else:
            rule6 = 0.0

        # 加權合成。原本是 0.40/0.30/0.20/0.10 —— 振幅最重。那是錯的，
        # 而且錯得會反過來咬人：
        #
        #   rule1 量的是絕對擺幅 (max-min)/10，不是訊噪比。
        #   **雜訊會讓擺幅變大**，所以最吵的訊號反而拿到滿分。
        #   合成訊號實測（心率真值 68）：
        #       小振幅但乾淨   rule1 0.22 → 舊SQI 0.57 → 心率 65.0 ± 0.0  完美
        #       中振幅但很吵   rule1 1.00 → 舊SQI 0.66 → 心率 129.4       錯一倍
        #   壞的比好的高分。而「小振幅但乾淨」正是新手機的樣子 ——
        #   自動曝光與時域降噪會壓抑畫面上的週期變化，訊號乾淨但擺幅小。
        #   旗艦機量不到 HRV，原因就在這裡：被自己的 SQI 擋掉了。
        #
        #   真正分得開的是 rule2（心跳頻段的能量佔比）與 rule5（週期性）：
        #       可用   rule2 0.66~0.74   rule5 0.75~0.82
        #       不可用 rule2 0.22~0.31   rule5 0.27~0.35
        #   中間空一大段，完全不重疊。權重改成以這兩條為主。
        #
        #   rule1 留 0.05：擺幅接近 0 時（完全沒有手指）還是該扣分，
        #   但它不該再主導。rule6 四種情境都接近 1.00，等於沒有資訊。
        sqi = 0.05 * rule1 + 0.50 * rule2 + 0.40 * rule5 + 0.05 * rule6

        return float(np.clip(sqi, 0.0, 1.0))

    except Exception:
        return 0.0


# ============================================================================
# 【新增 3】運動偽影檢測 (Motion Artifact Detection)
# 在 5 秒視窗內檢測突然的訊號跳變
# ============================================================================
def detect_motion_artifact(signal_data, window_size=None):
    """
    檢測是否存在運動偽影（手指晃動）。

    原理：
    - 計算訊號的一階差分（代表變化率）
    - 如果差分過大，代表有突然跳變（運動偽影）
    - 返回 True = 無偽影（訊號穩定），False = 有偽影（拒絕）
    """
    if window_size is None:
        window_size = SQI_WINDOW_SIZE

    n = len(signal_data)
    if n < window_size:
        return True  # 資料不足，暫且接受

    recent_signal = signal_data[-window_size:]

    try:
        x = np.asarray(recent_signal, dtype=np.float64)

        # 先去掉慢速趨勢再判定。
        # 手指壓力微調造成的是「基線緩慢漂移」—— 帶通濾波本來就會濾掉，
        # 對 HRV 沒有實質傷害，不該判成晃動。真正該擋的是「突然跳變」。
        x = signal.detrend(x)

        diff = np.diff(x)
        abs_diff = np.abs(diff)
        if len(abs_diff) < 4:
            return True

        # 用中位數 + MAD 而不是平均 + 標準差。
        # 平均和標準差本身會被離群值拉高 —— 有幾個大跳變時門檻跟著變高
        # 反而擋不住；訊號很平順時標準差趨近 0，門檻低到正常波形都算異常。
        med = float(np.median(abs_diff))
        mad = float(np.median(np.abs(abs_diff - med))) * 1.4826
        threshold = med + 5.0 * max(mad, 1e-9)

        # 再加一道絕對幅度門檻：跳變要跟脈動振幅同量級才算數。
        pulse_amp = float(np.percentile(x, 97.5) - np.percentile(x, 2.5))
        threshold = max(threshold, pulse_amp * 0.25)

        anomaly_ratio = float(np.sum(abs_diff > threshold) / len(abs_diff))

        # 門檻 0.10 → 0.20：偶爾一兩下不該讓整段作廢。
        return bool(anomaly_ratio < 0.20)

    except Exception:
        return True


def msptd_beat_detector(sig_in, fs, plaus_hr_bpm=(40, 180)):
    """MSPTDfast 波峰偵測（保持原有邏輯）。"""
    x = np.asarray(sig_in, dtype=np.float64)
    n = len(x)
    if n < 10:
        return np.array([], dtype=int)

    x = signal.detrend(x)

    max_len = max(int(np.ceil(n / 2)) - 1, 1)
    plaus_hr_hz = np.array(plaus_hr_bpm) / 60.0
    durn = n / fs
    scales = np.arange(1, max_len + 1)
    scale_freqs = (max_len / scales) / durn
    valid_mask = scale_freqs >= plaus_hr_hz[0]
    max_scale = int(np.max(scales[valid_mask])) if np.any(valid_mask) else max_len
    max_scale = max(max_scale, 1)

    m_max = np.zeros((max_scale, n), dtype=bool)
    for k in range(1, max_scale + 1):
        j = np.arange(k, n - k)
        if len(j) == 0:
            continue
        center = x[j]
        m_max[k - 1, j] = (center > x[j - k]) & (center > x[j + k])

    gamma = m_max.sum(axis=1)
    if gamma.sum() == 0:
        return np.array([], dtype=int)

    lambda_scale = int(np.argmax(gamma)) + 1
    m_max_reduced = m_max[:lambda_scale, :]
    is_peak = m_max_reduced.all(axis=0)
    return np.where(is_peak)[0]



def refine_peaks_parabolic(sig, peaks):
    """
    拋物線頂點擬合 — 將整數 sample 位置的 peak 精化到亞採樣精度。
    對每個 peak p，取 sig[p-1], sig[p], sig[p+1] 擬合拋物線 y = ax²+bx+c，
    頂點 x = -b/(2a) 即真實波峰位置（可為小數）。
    精度提升：120fps 下 ±8.3ms → ±<1ms
    """
    refined = []
    for p in peaks:
        if 1 <= p < len(sig) - 1:
            y0, y1, y2 = float(sig[p-1]), float(sig[p]), float(sig[p+1])
            denom = y0 - 2.0*y1 + y2
            offset = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-10 else 0.0
            # 限制修正量在 ±0.5 採樣內，避免噪音干擾
            offset = max(-0.5, min(0.5, offset))
            refined.append(float(p) + offset)
        else:
            refined.append(float(p))
    return np.array(refined, dtype=np.float64)


def estimate_f0_hz(x, fs, lo=0.7, hi=3.0):
    """從訊號本身估心率基頻（Hz），用來決定帶通上限。

    只用來設濾波器頻寬，不參與 BPM 計算 —— 估錯一點只是頻寬稍寬或稍窄，
    不會直接寫進結果。
    """
    try:
        z = np.asarray(x, dtype=np.float64)
        z = z - np.mean(z)
        freqs, psd = signal.welch(z, fs=fs, nperseg=min(512, len(z)))
        m = (freqs >= lo) & (freqs <= hi)
        if not np.any(m) or np.sum(psd[m]) <= 0:
            return None
        return float(freqs[m][np.argmax(psd[m])])
    except Exception:
        return None


def estimate_rr_jitter(rr_ms, min_n=None):
    """估計「每拍獨立的峰值定位誤差」sigma（毫秒）。

    設每拍的定位誤差為 eps（白噪、彼此獨立），則
        RR_i = T_i + eps_{i+1} - eps_i
        c0 = Var(T) + 2*sigma^2
        c1 = Cov(T_i, T_i+1) - sigma^2
        lag >= 2 完全不受 eps 影響
    真 HRV 的自協方差隨 lag 平滑變化（呼吸性竇性心律不整週期約 4 秒），
    量測抖動則只在 lag 0/1 造成尖銳的正負跳動。
    所以用 lag 2..6 外推回 lag 0 與 lag 1，差額就是 eps 的貢獻。

    回傳 None 表示樣本不足或估不出來 —— 此時不做任何修正。

    已知限制：真 HRV 中「逐拍白噪」的成分與定位抖動在數學上無法區分。
    合成測試中，真 RMSSD 90ms 且完全無抖動時會被誤估約 20ms，
    導致低報約兩成。靜息 HRV（RMSSD 10-50）範圍內誤差很小。
    """
    try:
        x = np.asarray(rr_ms, dtype=np.float64)
        n = len(x)
        # 拿來「修正」時樣本要多（DEBIAS_MIN_RR=30），估不準會扣錯量；
        # 拿來「判斷這次量得準不準」時可以放寬 —— 寧可粗略也要有得判。
        if n < (DEBIAS_MIN_RR if min_n is None else min_n):
            return None
        x = x - np.mean(x)
        c = [float(np.dot(x[:n - k], x[k:]) / (n - k)) for k in range(0, 7)]
        lags = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        vals = np.array(c[2:7], dtype=np.float64)
        A = np.vstack([lags, np.ones_like(lags)]).T
        slope, icpt = np.linalg.lstsq(A, vals, rcond=None)[0]
        c1_T = slope * 1.0 + icpt      # 若無抖動，lag1 自協方差應該是多少
        c0_T = icpt                    # 若無抖動，變異數應該是多少
        s2 = 0.5 * ((c1_T - c[1]) + (c[0] - c0_T) / 2.0)
        if not np.isfinite(s2) or s2 <= 0:
            return 0.0
        return float(np.sqrt(s2))
    except Exception:
        return None


def hrv_from_pool(beat_t, beat_rr, debias=False):
    """從『跨視窗累積起來的 RR』算 RMSSD / SDNN。

    為什麼需要這個函式 —— 12 秒的視窗有兩個先天限制：

      1. 只有約 12 拍。RMSSD 由 12 拍算出來，標準誤大約是真值的 20%，
         畫面上的數字會一直跳。
      2. SDNN 要量的慢波（呼吸性竇性心律不整、血壓調節）週期是 7~25 秒，
         12 秒的視窗物理上裝不下。合成訊號實測：真值 50ms 的 SDNN，
         每個視窗只算得出 22~35ms，系統性偏低 20~28ms。
         這不是係數問題，再怎麼平均都補不回來。

    而且原本的做法是「這個視窗剔除超過 20% 就整組作廢」——
    12 拍剔掉 4 拍就觸發，剩下那 8 拍明明是好的也一起丟掉。
    訊號稍差的機型（手電筒離鏡頭遠）幾乎每個視窗都會踩到，
    於是波形一直在跑、數字卻永遠出不來。

    改成：每個視窗只把「通過清洗的那幾拍」丟進池子，
    HRV 從池子裡算。樣本從 12 拍變成 60 拍以上，
    雜訊降低約 2.2 倍，SDNN 也終於裝得下慢波 —— 是同時變靈敏又變準。

    beat_t  每一段 RR 的起點時刻（毫秒，整場錄製的絕對時間）
    beat_rr 對應的 RR 長度（毫秒）
    兩個陣列等長。時間相鄰的判定用 beat_t 自己算，不靠陣列順序，
    因為視窗會重疊、順序不保證乾淨。
    """
    try:
        # JS 傳進來的是 TypedArray／Array，先轉成 Python 物件再給 numpy ——
        # 跟 process_data_from_js 處理 rgb_data 的方式一致。
        if hasattr(beat_t, 'to_py'):
            beat_t = beat_t.to_py()
        if hasattr(beat_rr, 'to_py'):
            beat_rr = beat_rr.to_py()
        t = np.asarray(beat_t, dtype=np.float64)
        r = np.asarray(beat_rr, dtype=np.float64)
        if len(t) != len(r) or len(t) < 3:
            return {"rmssd": 0.0, "sdnn": 0.0, "bpm": 0.0, "n": int(len(t)),
                    "span_sec": 0.0, "pairs": 0}
        # 去重（視窗重疊時同一拍會被送進來好幾次）並按時間排序
        order = np.argsort(t)
        t, r = t[order], r[order]
        keep = np.concatenate([[True], np.diff(t) > 60.0])   # 60ms 內視為同一拍
        t, r = t[keep], r[keep]
        if len(t) < 3:
            return {"rmssd": 0.0, "sdnn": 0.0, "bpm": 0.0, "n": int(len(t)),
                    "span_sec": 0.0, "pairs": 0}

        # 生理界線再擋一次（池子是跨視窗來的，寧可多擋一道）
        ok = (r >= 300) & (r <= 2000)
        med = float(np.median(r[ok])) if np.any(ok) else 0.0
        if med > 0:
            ok = ok & (np.abs(r - med) / med < 0.35)
        rr = r[ok]; tt = t[ok]
        if len(rr) < 3:
            return {"rmssd": 0.0, "sdnn": 0.0, "bpm": 0.0, "n": int(len(rr)),
                    "span_sec": 0.0, "pairs": 0}

        span = float((tt[-1] + rr[-1] - tt[0]) / 1000.0)
        sdnn = float(np.std(rr))
        bpm = float(60000.0 / np.median(rr))

        # RMSSD 只能用「時間上真的相鄰」的配對。
        # 池子裡難免有缺口（某幾拍被剔掉、某個視窗整個沒過），
        # 跨過缺口的一對差值會被灌大 —— 這正是原本用陣列相鄰時的老問題。
        gap = np.abs(tt[1:] - (tt[:-1] + rr[:-1]))
        adjacent = gap < 40.0                      # 40ms 內算接得上
        d = np.abs(np.diff(rr))[adjacent]
        # Malik：相鄰兩拍差超過 20% 視為漏拍／多拍，不是真的 HRV
        if med > 0:
            d = d[d < med * 0.20]
        rmssd = float(np.sqrt(np.mean(d ** 2))) if len(d) > 0 else 0.0
        pairs = int(len(d))

        # 抖動去偏預設關閉。
        #
        # 直覺上池子樣本多、sigma 估得準，應該可以修得更好。實測相反：
        #     真值        去偏開      去偏關
        #     RMSSD 35    -11.3       -7.0
        #     RMSSD 22     -8.8       -5.2
        #     RMSSD 35     -10.1      +6.3   （耦合差的情境）
        #     RMSSD 70     -35.2      -25.6
        # 四個情境全部是「開了更差」，而且一致地往低估跑。
        #
        # 原因 estimate_rr_jitter 自己的註解就寫了：真 HRV 裡「逐拍白噪」
        # 的成分與量測抖動在數學上無法區分。12 秒視窗樣本少、估出來的
        # sigma 小於門檻所以幾乎不作用；池子樣本一多，它就把真的 HRV
        # 當成抖動扣掉了。整段結算那條路徑維持原樣，這裡不開。
        jitter_ms = -1.0
        if debias and len(rr) >= DEBIAS_MIN_RR:
            j = estimate_rr_jitter(rr)
            if j is not None:
                jitter_ms = float(j)
                if j > DEBIAS_MIN_SIGMA:
                    rmssd = float(np.sqrt(max(0.0, rmssd ** 2 - 6.0 * j ** 2)))
                    sdnn = float(np.sqrt(max(0.0, sdnn ** 2 - 2.0 * j ** 2)))

        # 抖動太大時 RMSSD 不報。sigma 是「每拍定位誤差」，
        # 它會以 6*sigma^2 灌進 RMSSD —— 這是代數關係不是經驗值。
        # 這裡不是拿它去修正（修正會把真 HRV 一起扣掉，見上），
        # 而是拿它當「這次量得準不準」的判斷，超過就不報。
        # 實驗過改在「未過濾」的序列上估 sigma（想抓得更兇一點），結果更差：
        # 想留下的『耦合差』從 99% 掉到 29%，想擋掉的『雜訊大』照樣過關。
        # 維持在清洗後的序列上估。
        if jitter_ms < 0 and len(rr) >= 20:
            _j = estimate_rr_jitter(rr, min_n=20)
            if _j is not None:
                jitter_ms = float(_j)
        if jitter_ms >= POOL_JITTER_MAX:
            rmssd = 0.0

        # 生理合理性：靜息 RMSSD/RR 典型 2~5%，運動員約 8%。
        # 超過 15% 幾乎確定是假影。只擋 RMSSD ——
        # SDNN 對逐拍抖動的敏感度只有 RMSSD 的三分之一
        # （SDNN^2 = 真值^2 + 2*sigma^2，RMSSD^2 = 真值^2 + 6*sigma^2），
        # 被 RMSSD 連坐一起歸零是過度懲罰。
        if med > 0 and rmssd > med * 0.15:
            rmssd = 0.0

        return {"rmssd": round(rmssd, 1), "sdnn": round(sdnn, 1),
                "bpm": round(bpm, 1), "n": int(len(rr)),
                "span_sec": round(span, 1), "pairs": pairs,
                "jitter_ms": round(jitter_ms, 1)}
    except Exception:
        return {"rmssd": 0.0, "sdnn": 0.0, "bpm": 0.0, "n": 0,
                "span_sec": 0.0, "pairs": 0}


class WebProfileManager:
    def __init__(self):
        self.profiles = {"Public Mode": {"c_R": 3.0, "c_G": 2.0, "baseline_bpm": 75.0, "trained": False}}
        self.current_user = "Public Mode"
        self.bpm_contexts = {}

    def get_ctx(self, name):
        if name not in self.bpm_contexts:
            self.bpm_contexts[name] = {'history': [], 'pending': None}
        return self.bpm_contexts[name]

    def update_learning(self, bpm_truth, r_mean, g_mean):
        p = self.profiles.get(self.current_user)
        if not p or not p.get("trained", False): return
        opt_R = r_mean / g_mean if g_mean > 0 else 3.0
        opt_R = max(2.5, min(3.5, opt_R))
        p["c_R"] = (p["c_R"] * 0.99) + (opt_R * 0.01)
        p["baseline_bpm"] = (p["baseline_bpm"] * 0.95) + (bpm_truth * 0.05)


engine = WebProfileManager()


def sync_profiles(json_str):
    try:
        data = json.loads(json_str)
        engine.profiles.update(data)
    except:
        pass


def export_profiles():
    return json.dumps(engine.profiles)


# ============================================================================
# 【修正】主處理函數 - 加入 SQI 和運動偽影檢測
# ============================================================================


def process_data_from_js(rgb_data, timestamps, fps, polar_bpm, profile_name, is_training, is_training_active):
    try:
        engine.current_user = profile_name
        ctx = engine.get_ctx(profile_name)

        if hasattr(rgb_data, 'to_py'):
            rgb_data = rgb_data.to_py()
        if hasattr(timestamps, 'to_py'):
            timestamps = timestamps.to_py()

        buffer = np.asarray(rgb_data, dtype=np.float64).reshape(-1, 3)

        # 門檻用時間不用格數。寫死 150 格的話，15fps 的機器要 10 秒才開始，
        # 而 JS 那邊 3 秒就開始呼叫了 —— 中間那 7 秒一律回「緩衝資料不足」。
        min_frames = max(60, int(3.0 * (fps if fps and fps > 0 else 30.0)))
        if len(buffer) < min_frames:
            return {"bpm": 0, "rmssd": 0, "sdnn": 0, "error": "緩衝資料不足", "peaks": 0, "sqi": 0, "motion_free": True}

        # ====================================================================
        # 【修正 1】以真實 timestamp 重新取樣到均勻時間網格
        # ====================================================================
        real_fps = fps
        ts = None
        if timestamps is not None:
            try:
                ts = np.asarray(timestamps, dtype=np.float64)
            except Exception:
                ts = None

        if ts is not None and len(ts) == len(buffer) and ts[-1] > ts[0]:
            t_raw = (ts - ts[0]) / 1000.0
            duration = t_raw[-1]
            real_fps = (len(ts) - 1) / duration if duration > 0 else fps

            n_uniform = max(int(duration * TARGET_FPS), 150)
            t_uniform = np.linspace(0, duration, n_uniform)

            R = np.interp(t_uniform, t_raw, buffer[:, 0])
            G = np.interp(t_uniform, t_raw, buffer[:, 1])
            B = np.interp(t_uniform, t_raw, buffer[:, 2])
            fps = TARGET_FPS
        else:
            R, G, B = buffer[:, 0], buffer[:, 1], buffer[:, 2]

        # 「影像全黑」要三個通道都黑才算。
        #
        # 原本只看 G，那是照臉部 rPPG 的假設寫的。手指壓在鏡頭上、手電筒開著
        # 的時候畫面是純紅的：R 飽和在 200 以上，G 和 B 掉到 1 以下 ——
        # 那是「訊號最好」的樣子，不是全黑。JS 那邊的接觸判定本來就允許
        # （只要求 R/(G+1) > 2），於是就出現：波形漂亮、心率卻永遠是 --。
        #
        # 症狀是「一開始有數字，手沒動就暴跌到 0% 然後測不到」——
        # 因為手指剛放上去時還沒壓穩、自動曝光也還沒收斂，G 還有值；
        # 壓穩、曝光收斂之後 G 跌破 1，這一行就把整組結果清成 0。
        mean_R0, mean_G0, mean_B0 = float(np.mean(R)), float(np.mean(G)), float(np.mean(B))
        if max(mean_R0, mean_G0, mean_B0) < 1:
            return {"bpm": 0, "rmssd": 0, "sdnn": 0, "error": "影像全黑（三個通道都沒有亮度）",
                    "peaks": 0, "sqi": 0, "motion_free": False}

        # ====================================================================
        # 【新增】SQI 和運動偽影檢測 - 在濾波前先評估訊號品質
        # ====================================================================
        sqi = calculate_sqi(R, fs=fps)
        motion_free = detect_motion_artifact(R)

        # 沒有手指（或鏡頭被完全遮住／完全沒遮）——這種才是真的什麼都不能報
        if sqi < SQI_NO_FINGER:
            return {
                "bpm": 0, "rmssd": 0, "sdnn": 0,
                "error": f"訊號品質不足 SQI={sqi:.2f}（請確認手指貼緊鏡頭）",
                "peaks": 0, "sqi": round(sqi, 2), "motion_free": motion_free
            }

        # 品質不到門檻：整組都不報，心率也一樣。
        #
        # 上一版我把它拆開，讓品質不足時仍然回報 BPM ——
        # 理由是「BPM 從頻譜主峰讀，比逐拍時間耐雜訊」。那個推論是錯的，
        # 而且錯得剛好相反：合成測試顯示低品質時**崩掉的正是心率**
        # （真值 68 報成 129~158），RMSSD 只是漸進地被灌大。
        # 一個穩定顯示、卻錯了一倍的心率，比顯示「訊號不足」危險得多。
        if sqi < SQI_THRESHOLD:
            return {
                "bpm": 0, "rmssd": 0, "sdnn": 0,
                "error": f"訊號品質不足 SQI={sqi:.2f}（請調整手指位置或開燈）",
                "peaks": 0, "sqi": round(sqi, 2), "motion_free": motion_free
            }

        # 手指在動：逐拍時間不可信，但這一段的心率還算得出來。
        # （動作偽影是「突然跳變」，跟整體訊噪比是兩回事。）
        hrv_allowed = motion_free

        # ====================================================================
        # 【修正 2】POS 演算法（原有邏輯保持）
        # ====================================================================
        # Moving mean。原本 w=5，在 30fps 下是 167ms 的方波，
        # 對 3Hz 已衰減到 0.65、5Hz 只剩 0.20 —— 等於再砍一次諧波。
        # 諧波正是讓波峰變尖、定位變準的東西，w 縮到 3。
        R = moving_mean(R, MM_W)
        G = moving_mean(G, MM_W)
        B = moving_mean(B, MM_W)

        mean_R, mean_G, mean_B = np.mean(R), np.mean(G), np.mean(B)

        # POS 是為「臉部遠距 rPPG」設計的：它靠三個通道對血液容積變化的
        # 不同反應去抵銷光照變動，前提是三個通道都有東西。
        #
        # 手指接觸式（手電筒透過指腹）不是那個情境：紅光穿得過組織、
        # 綠藍幾乎全被吸收，畫面就是純紅。這時候 Gn = G/mean_G 是拿雜訊
        # 除以接近零的數，POS 出來的是放大的雜訊，不是脈搏。
        # 紅光通道自己就是最乾淨的訊號 —— 接觸式 PPG 本來就這樣做。
        #
        # 判準用「綠藍相對紅有多弱」，不用絕對值：不同機型的曝光差很多。
        red_dominant = (mean_R > 0 and
                        mean_G < CONTACT_GB_RATIO * mean_R and
                        mean_B < CONTACT_GB_RATIO * mean_R)

        # profile_name == 'finger' 就是「手指壓在鏡頭上」這個量測模式本身，
        # 不必再從畫面去猜。前端的接觸判定已經要求 R/(G+1) > 2，
        # 進得到這裡的每一格畫面本來就是紅光主導的。
        # red_dominant 留著給其他呼叫端（臉部／遠距）自動判斷用。
        contact_mode = (profile_name == 'finger') or red_dominant

        if contact_mode:
            # 扣掉平均值只是讓量綱跟 POS 那一路一致（兩邊都在 0 附近）。
            # smoothness priors 對常數項免疫（D @ 1 = 0，常數會原封不動
            # 被當成 trend 減掉），加不加這一步結果一樣。
            Rn = (R - mean_R) / mean_R if mean_R > 0 else (R - np.mean(R))
            # 負號：血液容積上升會吸收更多光，紅光通道是往下凹的。
            # POS 的 S2 = -2Rn + ... 也是同一個號誌，接下來的波峰偵測
            # 才不用分兩套。
            bvp = smoothness_priors_detrend(-Rn)
            src = 'red'
        else:
            Rn = R / mean_R if mean_R > 0 else R
            Gn = G / mean_G if mean_G > 0 else G
            Bn = B / mean_B if mean_B > 0 else B

            S1 = Gn - Bn
            S2 = -2 * Rn + Gn + Bn

            std_S2 = np.std(S2)
            if std_S2 == 0: std_S2 = 1e-6
            alpha = np.std(S1) / std_S2

            bvp = smoothness_priors_detrend(S1 + alpha * S2)
            src = 'pos'

        # 帶通濾波
        nyq = fps / 2.0
        low_cut = 0.5 / nyq   # 放寬下限（允許較慢心率）

        # 上限跟著心率走。寫死 3.0 Hz 對 HR 60 沒問題（諧波在 2/3 Hz），
        # 但 HR 100 的二次諧波是 3.33 Hz，會被整個切掉。
        f0 = estimate_f0_hz(R, fps)
        hi_hz = 3.0 if f0 is None else max(BAND_HI_MIN, BAND_HI_MULT * f0)
        hi_hz = min(hi_hz, nyq * 0.90)
        high_cut = hi_hz / nyq
        plaus_hr_bpm = (45.0, 150.0)

        if is_training and polar_bpm > 30:
            target_hz = polar_bpm / 60.0
            low_cut = max(0.5, target_hz - 0.3) / nyq
            high_cut = min(3.0, target_hz + 0.3) / nyq
            plaus_hr_bpm = (max(30.0, polar_bpm - 25), min(220.0, polar_bpm + 25))
            if is_training_active:
                engine.update_learning(polar_bpm, np.mean(R), np.mean(G))

        b, a = signal.butter(4, [low_cut, high_cut], btype='band')
        f_bvp = signal.filtfilt(b, a, bvp)

        # ====================================================================
        # 【修正 3】亞像素波峰分析
        # ====================================================================
        interp_factor = 4
        new_fps = fps * interp_factor
        t = np.arange(len(f_bvp))
        t_new = np.linspace(0, len(f_bvp) - 1, len(f_bvp) * interp_factor)
        cs = CubicSpline(t, f_bvp)
        f_bvp_interp = cs(t_new)

        # ====================================================================
        # 【修正 4】MSPTDfast 波峰偵測
        # ====================================================================
        peaks = msptd_beat_detector(f_bvp_interp, new_fps, plaus_hr_bpm=plaus_hr_bpm)

        # ── 拋物線頂點擬合：整數 sample → 亞採樣精度 ─────────────────────────
        refined_peaks = refine_peaks_parabolic(f_bvp_interp, peaks)

        bpm, rmssd, sdnn = 0, 0, 0

        if len(peaks) >= 4:
            rr = np.diff(refined_peaks) * (1000.0 / new_fps)

            # ================================================================
            # RR 清洗：用「遮罩」而不是「刪除」
            # ================================================================
            #  原本是連續三段過濾，每一段都直接把元素從陣列刪掉：
            #    1. rr 落在 400~1500 之外 → 刪
            #    2. 與「前一筆」差超過 20% → 刪
            #    3. 超出 1.5×IQR → 刪
            #  三段都在削掉變異度，而變異度就是 HRV 本身。
            #  尤其第 3 段：SDNN 量的就是這個分布的離散程度，
            #  把尾巴修掉再算標準差，等於先把答案改小再去量它。
            #
            #  刪除還有一個更隱蔽的後果：陣列相鄰 ≠ 時間相鄰。
            #  刪掉第 i 筆之後，rr[i-1] 和 rr[i+1] 在陣列裡變成鄰居，
            #  但它們在時間上中間隔了一拍。RMSSD 算的是「相鄰兩拍的差」，
            #  拿跨過空隙的一對去算，值會被灌大 ——
            #  這正好解釋為什麼兩位受試者的 RMSSD 一個偏低一個偏高。
            #
            #  改成保留原長度 + 布林遮罩：
            #    SDNN  取所有可信的 RR（不做統計修剪）
            #    RMSSD 只取「兩筆都可信而且時間上相鄰」的配對
            ok = (rr >= 300) & (rr <= 2000)          # 只擋生理上不可能的值

            if np.sum(ok) >= 3:
                # 與「整段中位數」比較，不跟前一筆比。
                # 跟前一筆比會連鎖：一拍抓錯，後面那拍也跟著被判為異常。
                med = np.median(rr[ok])
                if med > 0:
                    # 第一關：離整段中位數太遠的直接剔除（漏拍、多拍）
                    ok = ok & (np.abs(rr - med) / med < 0.35)

                # ── 第二關：Malik 準則 ──────────────────────────────
                # 逐筆跟「前一筆已接受的間期」比，差超過 20% 就剔除。
                #
                # 為什麼中位數過濾不夠：HR 102 時中位數 588ms，±35% 是
                # 382–794ms —— 相鄰兩筆可以差到 400ms 還雙雙過關，
                # 那種抖動全部進 RMSSD。實測把 10ms 灌成 110ms，
                # 而 HR 102（交感主導）的 RMSSD 生理上應該低於 20ms。
                #
                # 中位數過濾擋的是「離群值」，Malik 擋的是「逐拍抖動」，
                # 兩者針對的東西不一樣，要一起用。
                #
                # 跟「前一筆已接受」的比而不是前一筆原始值 ——
                # 才不會一拍抓錯就連鎖判掉後面一整串。
                # 原本 prev 用「第一筆通過的 rr」起頭。如果那一筆本身就是
                # 漏拍造成的長間期，後面每一筆都會跟它比而全部被判掉 ——
                # 實測會出現 n_beats=1、整段不回報。改成用中位數起頭
                # （中位數對單一壞值免疫），並在連續兩筆被判掉時重新同步，
                # 這樣真的心率變化（例如運動後恢復）不會被卡住。
                prev = float(med)
                misses = 0
                for i in range(len(rr)):
                    if not ok[i]:
                        continue
                    if prev > 0 and abs(rr[i] - prev) / prev > 0.20:
                        ok[i] = False
                        misses += 1
                        if misses >= 2:
                            prev = rr[i]
                            misses = 0
                    else:
                        prev = rr[i]
                        misses = 0

                valid_rr = rr[ok]

                if len(valid_rr) >= 2:
                    raw_bpm = 60000.0 / np.median(valid_rr)

                    # 【新增】頻譜驗證（確保波峰計數準確）
                    freqs, psd = signal.periodogram(f_bvp, fs=fps)
                    band_mask = (freqs >= 0.75) & (freqs <= 2.5)

                    fft_bpm = 0.0
                    if np.any(band_mask) and np.sum(psd[band_mask]) > 0:
                        band_freqs = freqs[band_mask]
                        band_psd = psd[band_mask]
                        peak_freq = float(band_freqs[np.argmax(band_psd)])
                        fft_bpm = peak_freq * 60.0

                    # 剔除比例：剔太多代表峰值偵測本身不穩，
                    # 這種情況下的 HRV 不該當成有效數值回報。
                    reject_ratio = 1.0 - (float(np.sum(ok)) / float(len(rr)))

                    # 可靠度不再只看剔除比例。
                    # 12 拍剔掉 4 拍（0.33）就超過原本的 0.20 門檻，
                    # 但剩下 8 拍其實完全夠算 —— 訊號稍差的機型幾乎每個
                    # 視窗都會踩到，於是波形一直跑、數字永遠出不來。
                    # 真正決定 RMSSD 算不算得出來的是「還剩幾對時間上相鄰
                    # 而且兩筆都可信的拍」，不是丟掉了幾成。
                    pair_mask = ok[:-1] & ok[1:]
                    n_pairs = int(np.sum(pair_mask))
                    rr_reliable = (n_pairs >= MIN_PAIRS_IN_WIN and
                                   reject_ratio <= MAX_REJECT_RATIO)

                    # 頻譜驗證要考慮諧波。PPG 波形有重搏波，低心率時
                    # 二次諧波的能量常常比基頻還高 —— 實測 HR 55 時
                    # 諧波/基頻 = 1.8，argmax 直接抓到 1.83 Hz。
                    # 原本的寫法會因此把 BPM 覆蓋成 110（正好兩倍），
                    # 而且順手把 rr_reliable 設成 False，
                    # 於是靜息心率低的人不但看到錯的心率，還永遠拿不到 HRV。
                    #
                    # 改成：raw_bpm 與 fft_bpm 成簡單整數倍時視為一致，
                    # 並且相信 RR 序列（時域數拍不會把一拍數成兩拍）。
                    # 只有連諧波關係都對不上，才判定波峰計數有問題。
                    if fft_bpm > 0:
                        ratio = raw_bpm / fft_bpm
                        harmonic_ok = False
                        for k in (1.0, 0.5, 2.0, 1.0 / 3.0, 3.0):
                            if abs(ratio - k) / k < 0.12:
                                harmonic_ok = True
                                break
                        if not harmonic_ok:
                            raw_bpm = fft_bpm
                            rr_reliable = False

                    # 【修正】生理連續性檢查（原有邏輯）
                    accept = True
                    if len(ctx['history']) >= 3:
                        recent_median = np.median(ctx['history'])
                        if recent_median > 0 and abs(raw_bpm - recent_median) / recent_median > 0.25:
                            if (ctx['pending'] is not None and
                                    abs(raw_bpm - ctx['pending']) / max(ctx['pending'], 1e-6) < 0.15):
                                ctx['history'] = [ctx['pending'], raw_bpm]
                                ctx['pending'] = None
                            else:
                                ctx['pending'] = raw_bpm
                                accept = False
                        else:
                            ctx['pending'] = None

                    if accept:
                        ctx['history'].append(raw_bpm)

                    if len(ctx['history']) > 5:
                        ctx['history'] = ctx['history'][-5:]

                    if ctx['history']:
                        bpm = np.median(ctx['history'])

                    rr_for_hrv = valid_rr

                    # 時域 HRV：只在波峰可靠時計算
                    if rr_reliable and len(valid_rr) > 2:
                        # SDNN：直接用所有可信的 RR。
                        # 不再做 IQR 修剪 —— 那是把要量的東西先削掉再量。
                        sdnn = np.std(valid_rr)

                        # RMSSD：只取時間上真的相鄰、而且兩筆都可信的配對。
                        # ok[:-1] & ok[1:] 就是「這一對在原始序列裡連續」。
                        pair = ok[:-1] & ok[1:]
                        if np.sum(pair) > 0:
                            d = np.abs(rr[1:] - rr[:-1])[pair]
                            # 不再用固定 150ms 上限截斷。
                            # 真正高 HRV 的人，相鄰兩拍差 150ms 以上很常見，
                            # 固定門檻等於專門懲罰 HRV 好的人。
                            # 改成相對門檻：超過中位數 40% 才視為漏拍/多拍。
                            # 差值上限跟 Malik 對齊（0.20）。
                            # 原本 0.40 太寬，正是 RMSSD 被灌大的直接原因。
                            lim = med * 0.20 if med > 0 else 200.0
                            d = d[d < lim]
                            if len(d) > 0:
                                rmssd = np.sqrt(np.mean(d ** 2))

                        # ── 抖動去偏 ──────────────────────────────────
                        # 峰值定位誤差 eps 會以固定倍率灌大兩個指標：
                        #     RMSSD^2 = 真值^2 + 6*sigma^2
                        #     SDNN^2  = 真值^2 + 2*sigma^2
                        # 這是代數關係，不是經驗公式；合成訊號實測
                        # 預測值與實際值差在 1ms 內。
                        # 所以只要估得出 sigma，就可以把它的貢獻扣掉。
                        #
                        # 只在整段分析（RR 夠多）時才修 —— 12 秒即時視窗
                        # 樣本太少，估出來的 sigma 自己就在跳。
                        jitter = estimate_rr_jitter(valid_rr)
                        if jitter is not None:
                            jitter_ms = jitter
                            if jitter > DEBIAS_MIN_SIGMA:
                                rmssd = float(np.sqrt(max(0.0, rmssd ** 2 - 6.0 * jitter ** 2)))
                                sdnn = float(np.sqrt(max(0.0, sdnn ** 2 - 2.0 * jitter ** 2)))

                        # ── 生理合理性把關 ────────────────────────────
                        # RR 清洗擋得住離群值，擋不住「整段都在抖」——
                        # 峰值定位每拍抖 ±60ms 時，剔除比例只有 14%（低於門檻），
                        # 但 RMSSD 仍被灌大四倍。剔除比例不是好的可靠度代理。
                        #
                        # 直接用生理界線：靜息狀態下 RMSSD 很少超過 RR 的 20%。
                        # 心率越快交感越主導、RMSSD 應該越低 —— 高心率配高 RMSSD
                        # 是自相矛盾的組合，那幾乎一定是假影而不是真的自律神經活性。
                        #
                        # 超過就整組不回報。寧可說「這次量不準」，
                        # 也不要給一個看起來正常、實際上錯四倍的數字。
                        # 門檻 15%：靜息 RMSSD/RR 典型值 2–5%，高 HRV 運動員約 8%。
                        # 15% 是「幾乎確定是假影」的界線 —— 只擋災難級的爆量，
                        # 不會誤傷真的 HRV 好的人。
                        #
                        # 但它擋不住「中度灌大」：實測峰值抖動 ±60ms 時
                        # RMSSD 從 14.5 被灌到 56（RR 的 9.6%），仍會通過。
                        # 那種只能從源頭修峰值偵測，清洗規則救不回來。
                        # 只擋 RMSSD。SDNN 對逐拍抖動的敏感度只有 RMSSD 的
                        # 三分之一（SDNN^2=真值^2+2*sigma^2，RMSSD^2=真值^2+6*sigma^2），
                        # 因為 RMSSD 爆掉就把 SDNN 一起歸零是過度懲罰 ——
                        # 而且 SDNN 歸零之後畫面上那一格就跳回 --，
                        # 看起來像整個量測失敗。
                        if med > 0 and rmssd > med * 0.15:
                            rmssd = 0

        # 品質不到 HRV 的門檻：逐拍時間不可信，但頻譜讀到的心率還是可信的
        if not hrv_allowed:
            rmssd = 0
            sdnn = 0

        # ── 把這個視窗的好拍交出去，讓上層累積 ──────────────────────────
        # 只有在「時域數出來的心率」與「頻譜看到的心率」是 1:1 對得上時才交。
        # 對不上通常代表峰值被數成兩倍或一半 —— 那種 RR 的中位數自己也是錯的，
        # 中位數過濾擋不住它，放進池子會整池被帶歪。
        beat_t, beat_rr = [], []
        if (hrv_allowed and 'ok' in locals() and 'rr' in locals()
                and 'refined_peaks' in locals()
                and int(locals().get('n_pairs', 0)) >= MIN_PAIRS_IN_WIN):
            try:
                # 諧波關係要跟上面 raw_bpm 那一段用同一套判準，不能更嚴。
                # 只認 1:1 的話，靜息心率低的人（頻譜常常鎖到二次諧波）
                # 會整場一拍都交不出來 —— 實測 HR 55 的案例就是這樣，
                # 加了池子卻完全沒有改善，就是被這裡擋掉的。
                # 真的被數成兩倍時，池子自己的 ±35% 中位數過濾會擋下來。
                _fb = float(locals().get('fft_bpm', 0.0))
                _rb = 60000.0 / float(np.median(rr[ok])) if np.any(ok) else 0.0
                same_beat = _fb <= 0 or _rb <= 0
                if not same_beat:
                    _ratio = _rb / _fb
                    for _k in (1.0, 0.5, 2.0, 1.0/3.0, 3.0):
                        if abs(_ratio - _k) / _k < 0.12:
                            same_beat = True; break
                if same_beat:
                    # refined_peaks 的單位是內插後的取樣點；換回毫秒，
                    # 再加上這個視窗第一格的絕對時刻，上層才接得起來
                    t0 = float(ts[0]) if ts is not None and len(ts) else 0.0
                    pt = np.asarray(refined_peaks, dtype=np.float64) * (1000.0 / new_fps) + t0
                    for i in range(len(rr)):
                        if ok[i]:
                            beat_t.append(round(float(pt[i]), 1))
                            beat_rr.append(round(float(rr[i]), 1))
            except Exception:
                beat_t, beat_rr = [], []

        return {
            "bpm": round(bpm, 1),
            "rmssd": round(rmssd, 1),
            "sdnn": round(sdnn, 1),
            "error": "",
            "peaks": len(peaks),
            "beat_t": beat_t,
            "beat_rr": beat_rr,
            "hrv_allowed": bool(hrv_allowed),
            "real_fps": round(real_fps, 1),
            "sqi": round(sqi, 2),
            "motion_free": bool(motion_free),
            # 診斷用：出問題時看得到證據，不用重現
            "rr_reject": round(float(locals().get('reject_ratio', 0.0)), 3),
            "hrv_ok": bool(locals().get('rr_reliable', False)),
            "n_beats": int(np.sum(locals().get('ok', np.array([])))) if 'ok' in locals() else 0,
            "engine_ver": ENGINE_VER,
            # 這一段是用哪個通道算的：red = 手指接觸（純紅畫面），pos = 三通道
            "src": locals().get('src', ''),
            "mean_rgb": [round(mean_R0, 1), round(mean_G0, 1), round(mean_B0, 1)],
            # 估到的每拍定位抖動（毫秒）。這是判斷「這次量得準不準」
            # 最直接的數字 —— 比剔除比例可靠，剔除比例對整段都在抖的
            # 情況完全沒反應。
            "jitter_ms": round(float(locals().get('jitter_ms', -1.0)), 1),
            "band_hi": round(float(locals().get('hi_hz', 0.0)), 2),
        }

    except Exception as e:
        return {
            "bpm": 0, "rmssd": 0, "sdnn": 0,
            "error": str(traceback.format_exc()),
            "peaks": 0,
            "sqi": 0,
            "motion_free": False
        }