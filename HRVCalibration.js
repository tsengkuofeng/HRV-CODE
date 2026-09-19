// ══════════════════════════════════════════════════════════
//  HRVCalibration.js — 相機 vs 胸帶的個人化校正
// ══════════════════════════════════════════════════════════
//  抽出來獨立成模組，是因為現在有兩個地方要用：
//    HRVScreen  —— 量測時套用校正、BLE 連線時自動記錄對照
//    PolarScreen —— 手動輸入對照、檢視與刪除紀錄
//  各寫一份的話，哪天改了門檻或係數上限只會改到一邊，
//  然後兩個畫面顯示的校正狀態就對不起來。
//
//  設計原則
//   1. 記錄的必須是「未校正」的相機原始值。存校正後的值會讓迴歸
//      在自己的輸出上再擬合，係數震盪而永遠收斂不到真值。
//   2. 校正綁在「這台裝置 + 這個人」：膚色、指腹厚度、按壓力道
//      差異很大，混不同人的資料擬合出來的對誰都不準。
//   3. RMSSD / SDNN 不能用線性 a*x+b。誤差主要來自逐拍計時抖動，
//      屬加性雜訊，要在變異數域擬合：polar² ≈ a2 * cam² + b2
//   4. 留每一筆明細而不只是累加值 —— 手動輸入一定會有打錯的時候，
//      只有明細刪得掉、重算得回來。
// ══════════════════════════════════════════════════════════

import AsyncStorage from '@react-native-async-storage/async-storage';

export const LOCAL_CAL_KEY = '@hrv_local_cal_v1';
export const LAST_CAM_KEY  = '@hrv_last_cam_raw';

export const MIN_PAIRS_BPM   = 5;   // BPM 是線性擬合，樣本需求較低
export const MIN_PAIRS_RMSSD = 8;   // 變異數域擬合要更多樣本才穩
export const MIN_PAIRS_SDNN  = 8;

export const blankCal = () => ({
  bpm:   { n:0, sx:0, sy:0, sxx:0, sxy:0 },
  rmssd: { n:0, sx:0, sy:0, sxx:0, sxy:0 },   // x/y 已是平方值
  sdnn:  { n:0, sx:0, sy:0, sxx:0, sxy:0 },
  diffsBpm: [], diffsRmssd: [], diffsSdnn: [],
  records: [],
});

// 最小平方法；資料不足或退化時回傳恆等式（等於不校正）
export const fitLS = (m, minN) => {
  if (!m || m.n < minN) return { a: 1, b: 0, n: m ? m.n : 0 };
  const d = m.n * m.sxx - m.sx * m.sx;
  if (Math.abs(d) < 1e-9) return { a: 1, b: 0, n: m.n };
  const a = (m.n * m.sxy - m.sx * m.sy) / d;
  return { a, b: (m.sy - a * m.sx) / m.n, n: m.n };
};

// 偏差與 95% 一致性界限（Bland-Altman）。
// n 大不代表準 —— 要看差距落在哪個範圍。
export const agreement = (diffs) => {
  if (!diffs || diffs.length < 3) return null;
  const n = diffs.length;
  const bias = diffs.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(diffs.reduce((s, d) => s + (d - bias) ** 2, 0) / Math.max(1, n - 1));
  return {
    n, bias: +bias.toFixed(2), sd: +sd.toFixed(2),
    loLoA: +(bias - 1.96 * sd).toFixed(2),
    hiLoA: +(bias + 1.96 * sd).toFixed(2),
  };
};

// 從明細重建所有累加值。刪掉一筆之後一定要呼叫。
export const rebuildCal = (lc) => {
  lc.bpm   = { n:0, sx:0, sy:0, sxx:0, sxy:0 };
  lc.rmssd = { n:0, sx:0, sy:0, sxx:0, sxy:0 };
  lc.sdnn  = { n:0, sx:0, sy:0, sxx:0, sxy:0 };
  lc.diffsBpm = []; lc.diffsRmssd = []; lc.diffsSdnn = [];
  (lc.records || []).forEach(r => {
    if (r.cam?.bpm > 0 && r.polar?.bpm > 0) {
      const m = lc.bpm, x = r.cam.bpm, y = r.polar.bpm;
      m.n++; m.sx += x; m.sy += y; m.sxx += x * x; m.sxy += x * y;
      lc.diffsBpm.push(+(x - y).toFixed(2));
    }
    [['rmssd', 'diffsRmssd'], ['sdnn', 'diffsSdnn']].forEach(([k, dk]) => {
      if (r.cam?.[k] > 0 && r.polar?.[k] > 0) {
        const m = lc[k], x = r.cam[k] ** 2, y = r.polar[k] ** 2;
        m.n++; m.sx += x; m.sy += y; m.sxx += x * x; m.sxy += x * y;
        lc[dk].push(+(r.cam[k] - r.polar[k]).toFixed(2));
      }
    });
  });
  lc.agree = {
    bpm:   agreement(lc.diffsBpm),
    rmssd: agreement(lc.diffsRmssd),
    sdnn:  agreement(lc.diffsSdnn),
  };
  return lc;
};

export const loadCal = async () => {
  try {
    const raw = await AsyncStorage.getItem(LOCAL_CAL_KEY);
    const lc = raw ? JSON.parse(raw) : blankCal();
    // 舊版存檔沒有 records / sdnn，補齊避免呼叫端到處判斷 null
    if (!lc.records) lc.records = [];
    if (!lc.sdnn) lc.sdnn = { n:0, sx:0, sy:0, sxx:0, sxy:0 };
    if (!lc.diffsSdnn) lc.diffsSdnn = [];
    return lc;
  } catch { return blankCal(); }
};

export const saveCal = async (lc) => {
  try { await AsyncStorage.setItem(LOCAL_CAL_KEY, JSON.stringify(lc)); } catch {}
};

// 新增一組對照。cam 必須是未校正的原始值。
export const addPair = (lc, cam, polar, source) => {
  if (!lc.records) lc.records = [];
  lc.records.push({
    at: Date.now(),
    src: source || 'ble',
    cam:   { bpm: +cam.bpm   || 0, rmssd: +cam.rmssd   || 0, sdnn: +cam.sdnn   || 0 },
    polar: { bpm: +polar.bpm || 0, rmssd: +polar.rmssd || 0, sdnn: +polar.sdnn || 0 },
  });
  if (lc.records.length > 50) lc.records.shift();
  return rebuildCal(lc);
};

export const removePair = (lc, at) => {
  lc.records = (lc.records || []).filter(r => r.at !== at);
  return rebuildCal(lc);
};

// ── 套用校正 ────────────────────────────────────────────
// 樣本不足一律回傳原值 —— 用兩三組資料擬出來的係數比不校正更糟。

export const correctBpm = (lc, camBpm) => {
  if (!camBpm || camBpm <= 0 || !lc) return camBpm;
  const f = fitLS(lc.bpm, MIN_PAIRS_BPM);
  if (f.n < MIN_PAIRS_BPM) return camBpm;
  const a = Math.max(0.7, Math.min(1.3, f.a));
  const b = Math.max(-15, Math.min(15, f.b));
  return Math.round(a * camBpm + b);
};

// 變異數域：polar² ≈ a2 * cam² + b2，算完開根號還原
const correctVar = (m, camVal, minN, aMax, outMax) => {
  if (!camVal || camVal <= 0 || !m) return camVal;
  const f = fitLS(m, minN);
  if (f.n < minN) return camVal;
  const a2 = Math.max(0.5, Math.min(aMax, f.a));
  const v = a2 * camVal * camVal + f.b;
  if (!(v > 0)) return camVal;                     // 扣過頭就不校正
  const out = Math.sqrt(v);
  // 只做溫和修正，避免單一極端配對把值拉飛
  return Math.round(Math.max(camVal * 0.5, Math.min(camVal * outMax, out)));
};

export const correctRmssd = (lc, camRmssd) =>
  correctVar(lc?.rmssd, camRmssd, MIN_PAIRS_RMSSD, 2.0, 2.0);

// SDNN 的偏差幅度通常比 RMSSD 大，上限放寬一點
export const correctSdnn = (lc, camSdnn) =>
  correctVar(lc?.sdnn, camSdnn, MIN_PAIRS_SDNN, 3.0, 2.5);

// 給畫面顯示用的整體狀態
export const calStatus = (lc) => ({
  bpm:   { n: lc?.bpm?.n   || 0, need: MIN_PAIRS_BPM,   fit: fitLS(lc?.bpm,   MIN_PAIRS_BPM) },
  rmssd: { n: lc?.rmssd?.n || 0, need: MIN_PAIRS_RMSSD, fit: fitLS(lc?.rmssd, MIN_PAIRS_RMSSD) },
  sdnn:  { n: lc?.sdnn?.n  || 0, need: MIN_PAIRS_SDNN,  fit: fitLS(lc?.sdnn,  MIN_PAIRS_SDNN) },
  agree: lc?.agree || null,
  records: lc?.records || [],
});

// 最近一次量測的相機原始值（跨畫面用）
export const saveLastCam = async (camRaw) => {
  try { await AsyncStorage.setItem(LAST_CAM_KEY, JSON.stringify(camRaw)); } catch {}
};
export const loadLastCam = async () => {
  try {
    const v = await AsyncStorage.getItem(LAST_CAM_KEY);
    return v ? JSON.parse(v) : null;
  } catch { return null; }
};

// 輸入把關：校正是累積的，一筆壞資料會影響之後每一次量測
export const validPolarInput = (bpm, rmssd, sdnn) => {
  const bad = [];
  if (bpm   && (bpm < 30   || bpm > 220)) bad.push('心率');
  if (rmssd && (rmssd < 1  || rmssd > 300)) bad.push('RMSSD');
  if (sdnn  && (sdnn < 1   || sdnn > 300)) bad.push('SDNN');
  return bad;
};
