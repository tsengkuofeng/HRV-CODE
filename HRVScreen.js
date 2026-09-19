/**
 * HRVScreen.js
 * 單鏡頭 HRV 量測（手指 PPG 版本）
 * - 後鏡頭（圓圈顯示）：手指接觸式 PPG
 * - PPG 波形區塊：實時波形顯示
 * - 訊號品質指標（SQI）和運動偵測
 * - Polar 選配：連線後雙面板滑動
 */
import React from 'react';
import {
  View, Text, TextInput, Platform,
  ScrollView, ActivityIndicator, Animated, PanResponder,
} from 'react-native';
import { TouchableOpacity } from './SoundTouchable';
import { sfx } from './AudioManager';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_URL } from './constants';
import { saveLastCam } from './HRVCalibration';

const AURA_ENGINE_URL = 'assets/hrv_engine.py';
const PYODIDE_URL     = 'https://cdn.jsdelivr.net/pyodide/v0.23.4/full/pyodide.js';
const HISTORY_KEY     = '@hrv_history';

// ─── 全域 Pyodide 快取（module 層級，mount/unmount 不會釋放）────────────────
let _pyCache        = null;   // { py, processData, hrvFromPool }
let _pyCachePromise = null;   // 載入中的 Promise，防止重複初始化

const getPyodide = async (engineUrl, onProgress) => {
  if (_pyCache) {
    onProgress?.('快取命中，略過重新載入', 95);
    return _pyCache;
  }
  if (_pyCachePromise) return _pyCachePromise;
  _pyCachePromise = (async () => {
    onProgress?.('載入 Pyodide…', 10);
    if (!document.querySelector(`script[src="${PYODIDE_URL}"]`)) {
      await new Promise((res, rej) => {
        const s = document.createElement('script');
        s.src = PYODIDE_URL; s.onload = res;
        s.onerror = () => rej(new Error('Pyodide 載入失敗'));
        document.head.appendChild(s);
      });
    }
    onProgress?.('載入 numpy / scipy…', 30);
    const py = await window.loadPyodide();
    await py.loadPackage(['numpy', 'scipy']);
    onProgress?.('載入 AI 引擎…', 60);
    const code = await (await fetch(engineUrl)).text();
    await py.runPythonAsync(code);
    _pyCache = {
      py,
      processData: py.globals.get('process_data_from_js'),
      // 跨視窗累積的 RR 拿來算 HRV。舊版引擎沒有這個函式，
      // 取不到就是 undefined —— 下面用不到它時會自動退回單視窗的值。
      hrvFromPool: py.globals.get('hrv_from_pool'),
    };
    _pyCachePromise = null;
    return _pyCache;
  })();
  return _pyCachePromise;
};


// ─── 取樣與分析參數 ──────────────────────────────────────────────────────────
//  全部用「時間」定義，不用畫格數 —— 取樣率會因機型、相機解析度、
//  發熱降頻而變動，用畫格數等於讓行為隨著硬體漂移。
const PROC_W = 160, PROC_H = 120;   // 取樣畫布尺寸，跟相機解析度脫鉤
const BUF_SECONDS      = 12;        // 分析視窗長度（秒）
const MIN_ANALYSIS_SEC = 3.0;       // 累積多久就開始出數字
const ANALYSIS_EVERY_MS = 400;      // 分析頻率
const SLOW_WARN_MS     = 8000;      // 超過這麼久還沒數字就說明原因
/* 低於這個取樣率才算「取樣率偏低」。
   原本沒有這個門檻，30fps 也會被寫成「取樣率偏低（30 fps）」——
   30fps 是手機相機的正常值，那句話是誤導。
   PPG 要抓到 20ms 等級的峰值位置，20fps 是實務下限。 */
const MIN_OK_FPS       = 20;
// 心率算不出來時先保留上一個值多久才清成 --。
// 單一個 12 秒視窗品質差一點是常態，不該讓畫面上的心率一直閃。
const BPM_HOLD_MS      = 4000;
// 還沒按開始時，逐拍池只留最近這麼多秒（預覽用，要能反映當下）
const POOL_PREVIEW_SEC = 60;
// 池子的 HRV 多久重算一次。它是幾十秒的累積量，不需要跟著每一格畫面跑。
const POOL_EVERY_MS    = 1500;
// 池子最長保留多久（計時中也適用）
const POOL_MAX_SEC     = 300;

/* ── Android：自動挑一顆真的量得到的後鏡頭 ──────────────────────────
   「S23 可以、S25 S26 測不太到」這種乾淨的世代切分不像硬體極限 ——
   手電筒離鏡頭遠造成的訊噪比下降是連續的，症狀會是「越新越差」，
   不會是「這一代完全正常、下一代完全不行」。

   實際的落差在程式裡：iOS 那條路會把 facingMode:'environment' 拿到的
   虛擬多鏡頭裝置換成實體廣角（見 startCameras 裡 isVirtual 那段），
   Android 從來沒有做這件事 —— 因為 label 認不出來。
   iOS 給的是 "Back Camera" / "Back Dual Wide Camera"，
   Android Chrome 給的是 "camera2 0, facing back"，字面上分不出哪顆是哪顆。

   既然分不出來就不要用猜的：每顆都開來試幾秒，看哪顆的 SQI 最高。
   量測本身就是最可靠的判準，而且不必假設任何一家廠商的命名規則。
   挑到之後把 deviceId 記在本機，下次直接開那顆，不用再試。

   探測只在「手指確實貼著、訊號卻一直起不來」時啟動；
   錄製中絕對不換鏡頭 —— 中途換會把整段訊號切成兩截。 */
const LENS_KEY         = '@hrv_lens_v1';
const PROBE_OK_SQI     = 0.50;   // 跟引擎的 SQI_THRESHOLD 對齊
const PROBE_TRIGGER_MS = 8000;   // 手指貼著這麼久還是不行，才開始試別顆
const PROBE_SETTLE_MS  = 1400;   // 剛換鏡頭，等自動曝光／白平衡穩定
const PROBE_SAMPLE_MS  = 2600;   // 每顆實際採計的時間

// ─── 裝置偵測 ────────────────────────────────────────────────────────────────
const isAndroid = typeof navigator !== 'undefined' && /android/i.test(navigator.userAgent);
// iPadOS 13 之後 iPad 的 UA 會偽裝成 Mac，只能靠「Mac + 有觸控」再認一次。
const isIOS = typeof navigator !== 'undefined' && (
  /iphone|ipad|ipod/i.test(navigator.userAgent) ||
  (/macintosh/i.test(navigator.userAgent) && (navigator.maxTouchPoints || 0) > 1)
);
// 從主畫面圖示開的（iOS 用 navigator.standalone，其他平台用 display-mode）。
// 這件事會影響「去哪裡改權限」，所以錯誤訊息要分開寫。
const isHomeApp = typeof window !== 'undefined' && (
  window.navigator?.standalone === true ||
  (window.matchMedia && window.matchMedia('(display-mode: standalone)').matches)
);

// ─── 自繪圖示 ────────────────────────────────────────────────────────────────
// 用 SVG 不用 emoji：emoji 的樣式由作業系統決定，同一顆按鈕在 iOS、
// Android、各家瀏覽器長得都不一樣，而且沒辦法跟著主題色走。
const IconLensSwap = ({ size = 20, color = '#333' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    stroke={color} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3.5 12a8.5 8.5 0 0 1 14.2-6.3" />
    <path d="M20.5 12a8.5 8.5 0 0 1-14.2 6.3" />
    <path d="M18 2.2v3.9h-3.9" />
    <path d="M6 21.8v-3.9h3.9" />
    <circle cx="12" cy="12" r="3.1" />
  </svg>
);

const IconTorch = ({ size = 20, color = '#333', on = false }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
    stroke={color} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3.2a5.8 5.8 0 0 0-3.4 10.5c.6.4.9 1.1.9 1.8v.4h5v-.4c0-.7.3-1.4.9-1.8A5.8 5.8 0 0 0 12 3.2z" />
    <path d="M9.5 18.4h5" />
    <path d="M10.4 20.8h3.2" />
    {/* 亮起時才畫光線 —— 開關狀態用形狀差異表示，不是只有顏色，
        色盲使用者也分得出來。 */}
    {on && (
      <>
        <path d="M12 0.8v1.2" />
        <path d="M3.6 5.1l1 .6" />
        <path d="M20.4 5.1l-1 .6" />
        <path d="M2.4 12.4h1.2" />
        <path d="M21.6 12.4h-1.2" />
      </>
    )}
  </svg>
);

// ─── 非 Web 佔位 ─────────────────────────────────────────────────────────────
const HRVPlaceholder = ({ theme, onClose }) => (
  <View style={{ flex:1, backgroundColor:theme.bg, justifyContent:'center', alignItems:'center', padding:32 }}>
    <Text style={{ fontSize:48, marginBottom:16 }}>❤️</Text>
    <Text style={{ color:theme.textMain, fontSize:16, fontWeight:'700', marginBottom:8 }}>HRV 量測</Text>
    <Text style={{ color:theme.textSub, fontSize:13, textAlign:'center', marginBottom:28 }}>
      請用手機瀏覽器開啟網頁版以使用此功能
    </Text>
    <TouchableOpacity onPress={onClose}
      style={{ paddingHorizontal:24, paddingVertical:10, borderRadius:12, borderWidth:1.5, borderColor:theme.primary }}>
      <Text style={{ color:theme.primary, fontWeight:'600' }}>← 返回</Text>
    </TouchableOpacity>
  </View>
);

const HRVScreen = ({ theme, onClose, onNavigate }) => {
  if (Platform.OS !== 'web') return <HRVPlaceholder theme={theme} onClose={onClose} />;
  return <HRVWebContent theme={theme} onClose={onClose} onNavigate={onNavigate} />;
};

// ─── Web 內容 ─────────────────────────────────────────────────────────────────
const HRVWebContent = ({ theme, onClose, onNavigate }) => {

  // ── State ──────────────────────────────────────────────────────────────────
  const [loadStep,       setLoadStep]       = React.useState('loading');
  const [loadMsg,        setLoadMsg]        = React.useState('準備中...');
  const [screen,         setScreen]         = React.useState('dashboard');
  const [bpm,            setBpm]            = React.useState(0);
  const [rmssd,          setRmssd]          = React.useState(0);
  const [sdnn,           setSdnn]           = React.useState(0);
  const [realFps,        setRealFps]        = React.useState(0);
  const [fingerStatus,   setFingerStatus]   = React.useState({ msg:'等待手指...', err:false });
  const [polarBpm,       setPolarBpm]       = React.useState(0);
  const [polarOk,        setPolarOk]        = React.useState(false);
  const [polarRmssd,     setPolarRmssd]     = React.useState(0);
  const [polarSdnn,      setPolarSdnn]      = React.useState(0);
  const [panel,          setPanel]          = React.useState(0);
  const [timerMode,      setTimerMode]      = React.useState('idle');
  const [timerLeft,      setTimerLeft]      = React.useState(0);
  // 改成正數計時。倒數會給人「還要撐 1:38」的壓力，而且時間到才有結果 ——
  // 撐不住就整段作廢，這是使用者反映撐不久的真正原因。
  // 正數 ＋ 里程碑：隨時可以結束，但看得到現在的數值算不算數。
  const [timerElapsed,   setTimerElapsed]   = React.useState(0);
  const [reportData,     setReportData]     = React.useState(null);
  const [calN,           setCalN]           = React.useState(0);
  const [torchOn,        setTorchOn]        = React.useState(false);
  // 手電筒是否可用。iOS Safari 不支援 torch 約束 ——
  // 顯示一顆按不動的按鈕比沒有更糟，所以偵測後才渲染。
  const [torchAvail,     setTorchAvail]     = React.useState(false);
  const [loadPct,        setLoadPct]        = React.useState(5);
  const [eduIdx,         setEduIdx]         = React.useState(0);
  const [timerSecs,      setTimerSecs]      = React.useState(180);
  // 時長手打模式：點數字切成輸入框，跟 ParamSlider 同一個手勢
  const [durEditing,     setDurEditing]     = React.useState(false);
  const [durText,        setDurText]        = React.useState('');
  const [ppgWaveform,    setPpgWaveform]    = React.useState([]);
  const [sqi,            setSqi]            = React.useState(0);
  const [motionFree,     setMotionFree]     = React.useState(true);
  const [cameras,        setCameras]        = React.useState([]);
  const [lensIdx,        setLensIdx]        = React.useState(0);

  // ── Refs ───────────────────────────────────────────────────────────────────
  const rearVideoRef       = React.useRef(null);
  const rearHiddenCvRef    = React.useRef(null);
  const overlayCvRef       = React.useRef(null);
  const ppgWaveformCvRef   = React.useRef(null);
  const pyProcessRef       = React.useRef(null);
  const rearStreamRef      = React.useRef(null);
  const rafRef             = React.useRef(null);
  const fingerRgbBuf       = React.useRef([]);
  const fingerTsBuf        = React.useRef([]);
  const lastFrameT         = React.useRef(0);
  const frameN             = React.useRef(0);
  const lastAnalysisT      = React.useRef(0);   // 上次跑分析的時間（毫秒）
  const analysisStartT     = React.useRef(0);   // 開始累積訊號的時間，用來判斷「等太久」
  const pyErrRef           = React.useRef('');  // 最近一次分析的錯誤，給畫面提示用
  const lastGoodT          = React.useRef(0);   // 上次真的算出數值的時間
  // 整段錄製的原始訊號。即時顯示用 12 秒滾動視窗（要反應快），
  // 但結算必須用整段 —— 見 genReport 裡的說明。
  const sessionRgbBuf      = React.useRef([]);
  const sessionTsBuf       = React.useRef([]);
  const [slowHint, setSlowHint] = React.useState('');
  // 相機打不開時的實際原因。
  //
  // 原本 startCameras 的四個 fallback 全部是 `catch {}` —— 不管失敗原因是什麼
  // 都被吞掉，畫面上什麼都不會顯示。使用者只會看到「圓圈是黑的」，
  // 而系統設定裡因為根本沒發出過權限請求，也不會有東西可以改。
  // 三星機上回報的「權限通知出不來、設定顯示未允許任何權限、沒地方可改」
  // 就是這個狀態。
  const [camErr, setCamErr] = React.useState('');
  // 最近一次請求失敗的技術細節（錯誤名稱、耗時、permissions 狀態）。
  // 使用者只能回報「沒反應」，那三個字沒辦法除錯；這一行可以被截圖。
  const [camDiag, setCamDiag] = React.useState('');
  // 鏡頭自動挑選的進度／結果。平常是空字串。
  // camDiag 只有在相機完全打不開時才渲染，這一行是「相機正常、但訊號起不來」
  // 的情況 —— 遠端測試的人可以直接截這一行回報最後選到哪一顆。
  const [lensNote, setLensNote] = React.useState('');
  const [calCount, setCalCount] = React.useState({ bpm:0, rmssd:0, sdnn:0 });
  // 最近一次量測的相機原始值。寫進 AsyncStorage 交給 Polar 頁讀 ——
  // 使用者是「量完 → 切到 Polar 頁 → 輸入」，中間換過畫面。
  const [lastCam, setLastCam] = React.useState(null);
  // rawBpm：未經校正的鏡頭原始值。提交校正時必須用它，
  //         用校正後的值會形成回饋迴路，係數永遠震盪不收斂。
  // goodQ  ：該筆樣本當下的訊號品質是否過關（SQI + 無晃動）
  const sessionStats       = React.useRef({ bpm:[], rawBpm:[], rmssd:[], rawRmssd:[], sdnn:[], polarRaw:[], polarRmssd:[], polarSdnn:[], goodQ:[] });
  const timerRef           = React.useRef(null);
  const timerActiveRef     = React.useRef(false);
  const polarBpmRef        = React.useRef(0);
  const polarRRBuf         = React.useRef([]);
  const calRef             = React.useRef({ a:1.0, b:0.0 });
  // 本機（每台裝置）校正：膚色、鏡頭、按壓方式因人而異，
  // 個人化係數比全域平均可靠得多；全域值只在本機樣本不足時當備援。
  const localCalRef        = React.useRef(null);   // { bpm:{a,b,n}, rmssd:{a2,b2,n}, agree:{...} }
  const swipeAnim          = React.useRef(new Animated.Value(0)).current;
  const ppgHistoryRef      = React.useRef([]);
  const recordWaveBuf      = React.useRef([]);  // 全程錄製波型（計時期間）
  const waveformScrollRef  = React.useRef(0);
  const lensIdxRef         = React.useRef(0);
  const noContactRef       = React.useRef(0);   // 連續無接觸幀數
  const contactFramesRef   = React.useRef(0);   // 連續接觸幀數（防抖）
  const totalSamplesRef    = React.useRef(0);   // 累計樣本數（秒數標籤用）
  const lockedDeviceIdRef  = React.useRef(null); // 鎖定的相機 deviceId
  const mountedRef         = React.useRef(true); // 元件是否仍掛載
  const processActiveRef   = React.useRef(false); // rAF loop 是否運行
  const sqiHistRef         = React.useRef([]);  // SQI 平滑歷史
  const engErrRef          = React.useRef('');  // 引擎自己回報的原因（不是丟出來的例外）
  const pyPoolRef          = React.useRef(null); // hrv_from_pool（舊引擎沒有）
  /* 鏡頭清單同時放一份在 ref 裡。
     processFrame 是自己 requestAnimationFrame 自己（見它第二行），
     所以它永遠是**第一次排程時那個 closure** —— 裡面讀到的 state 會
     停在初次 render 的值。相機清單是開頁好幾秒後才 enumerate 到的，
     從那個 closure 讀 cameras 只會拿到空陣列。這就是為什麼這個檔案
     的 rAF 迴圈幾乎只用 ref。 */
  const camerasRef         = React.useRef([]);
  const lastSampleRef      = React.useRef({ r:0, g:0, b:0 });  // 最近一格的 RGB 平均，診斷用
  const rememberedLensRef  = React.useRef(null);  // 上次挑中的鏡頭 { deviceId, label, score }
  const lensProbeRef       = React.useRef({
    state: 'idle',      // idle → watch → probe → done ／ off（手動換過或只有一顆）
    contactStart: 0,    // 手指開始連續貼著的時間
    cand: [], i: -1,    // 候選鏡頭與目前測到第幾顆
    phaseT: 0,          // 這一顆開始測的時間
    best: null,         // { deviceId, label, score }
    cur: 0,             // 這一顆目前看過的最高 SQI
    switching: false,   // 正在換鏡頭，這段期間不要下判斷
  });

  /* ── 逐拍累積池 ────────────────────────────────────────────────────
     即時 HRV 原本是「12 秒視窗自己算一次」。那個視窗只有約 12 拍，
     而且引擎只要剔掉超過 20% 就整組作廢 —— 12 拍剔掉 4 拍就觸發。
     訊號稍差的機型（手電筒離鏡頭遠）幾乎每個視窗都會踩到，
     於是波形一直在跑、RMSSD/SDNN 卻出不來，或出現一下就沒了。

     改成：每個視窗只把「通過清洗的那幾拍」丟進這個池子，
     HRV 從池子裡算。合成訊號實測（耦合差的情境）：
         出現率  RMSSD 53% → 99%，SDNN 53% → 99%
         誤差    RMSSD 9.1 → 6.3ms，SDNN 21.4 → 10.2ms
     出現率和準度同時變好，不是取捨。

     SDNN 的改善特別大是有原因的：它要量的慢波週期是 7~25 秒，
     12 秒的視窗物理上裝不下，怎麼平均都補不回來。

     池子的長度：計時開始之後就一路累積不丟（大部分人只量 1 分鐘，
     所以池子＝整場，即時看到的數字會一路收斂到結算的那個數字）；
     還沒開始計時的預覽階段則維持 60 秒滾動。 */
  const rrPoolRef          = React.useRef({ t: [], rr: [], lastT: 0 });
  const poolHrvRef         = React.useRef({ rmssd:0, sdnn:0, n:0, span:0, pairs:0, jitter:-1 });
  const lastPoolT          = React.useRef(0);
  const lastBpmT           = React.useRef(0);   // 上次拿到有效心率的時間

  // ── Helpers ────────────────────────────────────────────────────────────────
  const fmt = (s) => `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`;

  // ── 量測時長 ──────────────────────────────────────────────────────────────
  // MIN_VALID_SEC 是「數值開始可信」的門檻，不是最短錄製時間。
  // RMSSD 大約 60 秒以上才穩定；低於這個值不是不能存，
  // 而是要讓使用者知道那筆數字不能拿來跟別次比較。
  const MIN_VALID_SEC = 60;
  const DUR_STEP      = 30;    // 加減一次 30 秒
  const DUR_MIN       = 60;
  const DUR_MAX       = 600;
  const UNLIMITED     = 0;     // timerSecs === 0 代表不限時

  // 手打的時長：只收數字，用「碼表式」解讀 —— 最後兩位是秒，前面是分。
  //   100 → 1:00    130 → 1:30    300 → 3:00    1000 → 10:00
  //   45  → 0:45（會被下限夾到 1:00）
  //
  // 不要求使用者打冒號。手機數字鍵盤上根本沒有冒號，
  // 切到符號鍵盤只為了打一個字元，是很差的體驗。
  const parseDur = (t) => {
    const d = String(t || '').replace(/\D/g, '');
    if (!d) return null;
    if (d.length <= 2) return parseInt(d, 10);                    // 純秒
    const sec = parseInt(d.slice(-2), 10);
    const min = parseInt(d.slice(0, -2), 10);
    return min * 60 + sec;
  };

  const stepDur = (delta) => {
    setTimerSecs(cur => {
      if (cur === UNLIMITED) return delta > 0 ? DUR_MIN : UNLIMITED;
      const next = cur + delta;
      if (next < DUR_MIN) return UNLIMITED;      // 減到底 = 不限時
      return Math.min(DUR_MAX, next);
    });
  };

  const commitDur = () => {
    const v = parseDur(durText);
    if (v !== null) setTimerSecs(Math.max(DUR_MIN, Math.min(DUR_MAX, v)));
    setDurEditing(false);
  };

  const post = (body) => fetch(API_URL, {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)
  });

  const loadScript = (src) => new Promise((res, rej) => {
    if (document.querySelector(`script[src="${src}"]`)) { res(); return; }
    const s = document.createElement('script');
    s.src = src; s.onload = res;
    s.onerror = () => rej(new Error(`載入失敗: ${src}`));
    document.head.appendChild(s);
  });

  // ── Calibration ────────────────────────────────────────────────────────────
  //
  // 設計說明
  //  1. 提交的 rppg 必須是「未校正」的原始值。送校正後的值會讓迴歸在自己的
  //     輸出上再擬合一次，係數會震盪而永遠收斂不到真值。
  //  2. 校正以「本機」為主：膚色、鏡頭、按壓力道因人而異，全域平均可能比
  //     不校正更糟。本機樣本不足時才退回伺服器的全域值當備援。
  //  3. RMSSD 不能用線性 a*x+b。誤差主要來自逐拍時間抖動，屬加性雜訊，
  //     必須在「變異數域」擬合：rmssd_polar² ≈ a2 * rmssd_cam² + b2
  //     （引擎已做 4x 內插與拋物線頂點擬合，另有 <150ms 截斷與 IQR 過濾，
  //      會壓低高 HRV 者的值，所以 a2/b2 一律由實測配對學出，不用理論值。）
  //  4. 一致性以偏差與 95% 一致性界限（LoA）記錄，n 大不等於準。

  const LOCAL_CAL_KEY = '@hrv_local_cal_v1';
  const MIN_PAIRS_BPM   = 5;    // 本機配對達此數才啟用個人化 BPM 校正
  const MIN_PAIRS_RMSSD = 8;    // RMSSD 變異數擬合需要更多樣本才穩定
  const MIN_PAIRS_SDNN  = 8;    // SDNN 同理
  const SQI_MIN         = 0.6;  // 訊號品質閘門

  const blankCal = () => ({
    bpm:   { n:0, sx:0, sy:0, sxx:0, sxy:0 },
    rmssd: { n:0, sx:0, sy:0, sxx:0, sxy:0 },   // 這裡的 x/y 已是平方值
    // SDNN 原本完全沒有校正路徑 —— 但它是唯一呈現「系統性」偏差的指標
    // （實測兩位受試者都偏低，而 RMSSD 一高一低屬於隨機誤差）。
    // 有方向的偏差才校正得動，所以 SDNN 反而最該做。
    // 同樣走變異數域：SDNN 也是標準差，代數形式跟 RMSSD 一致。
    sdnn:  { n:0, sx:0, sy:0, sxx:0, sxy:0 },
    diffsBpm: [], diffsRmssd: [], diffsSdnn: [],   // 供偏差 / LoA 計算
    // 每一筆對照的明細。累加值（sx/sxy…）刪不掉單筆，
    // 而手動輸入一定會有打錯的時候 —— 留明細才能刪掉重算。
    records: [],
  });

  // 從明細重建所有累加值。刪除一筆之後呼叫。
  const rebuildCal = (lc) => {
    lc.bpm = { n:0, sx:0, sy:0, sxx:0, sxy:0 };
    lc.rmssd = { n:0, sx:0, sy:0, sxx:0, sxy:0 };
    lc.sdnn = { n:0, sx:0, sy:0, sxx:0, sxy:0 };
    lc.diffsBpm = []; lc.diffsRmssd = []; lc.diffsSdnn = [];
    (lc.records || []).forEach(r => {
      if (r.cam?.bpm > 0 && r.polar?.bpm > 0) {
        const m = lc.bpm, x = r.cam.bpm, y = r.polar.bpm;
        m.n++; m.sx += x; m.sy += y; m.sxx += x*x; m.sxy += x*y;
        lc.diffsBpm.push(+(x - y).toFixed(2));
      }
      [['rmssd','diffsRmssd'], ['sdnn','diffsSdnn']].forEach(([k, dk]) => {
        if (r.cam?.[k] > 0 && r.polar?.[k] > 0) {
          const m = lc[k], x = r.cam[k]*r.cam[k], y = r.polar[k]*r.polar[k];
          m.n++; m.sx += x; m.sy += y; m.sxx += x*x; m.sxy += x*y;
          lc[dk].push(+(r.cam[k] - r.polar[k]).toFixed(2));
        }
      });
    });
    lc.agree = {
      bpm: agreement(lc.diffsBpm),
      rmssd: agreement(lc.diffsRmssd),
      sdnn: agreement(lc.diffsSdnn),
    };
    return lc;
  };

  // 最小平方法；資料不足或退化時回傳恆等式
  const fitLS = (m, minN) => {
    if (!m || m.n < minN) return { a:1, b:0, n:m ? m.n : 0 };
    const d = m.n*m.sxx - m.sx*m.sx;
    if (Math.abs(d) < 1e-9) return { a:1, b:0, n:m.n };
    const a = (m.n*m.sxy - m.sx*m.sy) / d;
    return { a, b: (m.sy - a*m.sx)/m.n, n:m.n };
  };

  // 偏差（bias）與 95% 一致性界限：n 大不代表準，要看差距落在哪
  const agreement = (diffs) => {
    if (!diffs || diffs.length < 3) return null;
    const n = diffs.length;
    const bias = diffs.reduce((a,b)=>a+b,0)/n;
    const sd = Math.sqrt(diffs.reduce((s,d)=>s+(d-bias)**2,0)/Math.max(1,n-1));
    return { n, bias:+bias.toFixed(2), sd:+sd.toFixed(2),
             loLoA:+(bias-1.96*sd).toFixed(2), hiLoA:+(bias+1.96*sd).toFixed(2) };
  };

  const loadLocalCal = async () => {
    try {
      const raw = await AsyncStorage.getItem(LOCAL_CAL_KEY);
      localCalRef.current = raw ? JSON.parse(raw) : blankCal();
    } catch { localCalRef.current = blankCal(); }
    applyBestCal();
  };

  // 本機樣本夠就用本機係數，否則沿用伺服器全域值
  const applyBestCal = () => {
    const lc = localCalRef.current;
    if (!lc) return;
    const f = fitLS(lc.bpm, MIN_PAIRS_BPM);
    if (f.n >= MIN_PAIRS_BPM) {
      calRef.current = { a: Math.max(0.7, Math.min(1.3, f.a)),
                         b: Math.max(-15, Math.min(15, f.b)) };
      setCalN(f.n);
    }
  };

  // RMSSD 校正：變異數域擬合後開根號還原
  // SDNN 校正，與 correctRmssd 同一套算法
  const correctSdnn = (camSdnn) => {
    const lc = localCalRef.current;
    if (!camSdnn || camSdnn <= 0 || !lc || !lc.sdnn) return camSdnn;
    const f = fitLS(lc.sdnn, MIN_PAIRS_SDNN);
    if (f.n < MIN_PAIRS_SDNN) return camSdnn;
    const a2 = Math.max(0.5, Math.min(3.0, f.a));   // 上限放寬：SDNN 偏低幅度較大
    const v  = a2 * camSdnn * camSdnn + f.b;
    if (!(v > 0)) return camSdnn;
    const out = Math.sqrt(v);
    return Math.round(Math.max(camSdnn * 0.5, Math.min(camSdnn * 2.5, out)));
  };

  const correctRmssd = (camRmssd) => {
    const lc = localCalRef.current;
    if (!camRmssd || camRmssd <= 0 || !lc) return camRmssd;
    const f = fitLS(lc.rmssd, MIN_PAIRS_RMSSD);
    if (f.n < MIN_PAIRS_RMSSD) return camRmssd;
    const a2 = Math.max(0.5, Math.min(2.0, f.a));
    const v  = a2 * camRmssd * camRmssd + f.b;
    if (!(v > 0)) return camRmssd;                    // 扣過頭就不校正
    const out = Math.sqrt(v);
    // 只做溫和修正，避免單一極端配對把值拉飛
    return Math.round(Math.max(camRmssd*0.5, Math.min(camRmssd*2.0, out)));
  };

  const fetchCal = async () => {
    try {
      const r = await post({ action:'getCalibration' });
      const d = await r.json();
      // 全域值只當備援：本機樣本足夠時 applyBestCal 會覆蓋掉它
      if (d.a !== undefined) {
        const lc = localCalRef.current;
        const localN = lc ? fitLS(lc.bpm, MIN_PAIRS_BPM).n : 0;
        if (localN < MIN_PAIRS_BPM) { calRef.current = { a:d.a, b:d.b }; setCalN(d.n||0); }
      }
    } catch {}
    applyBestCal();
  };

  // fingerBpm 必須是原始值（未經 calRef 校正）
  const submitCal = async (fingerBpm, polarBpm, camRmssd, polarRmssdVal, camSdnn, polarSdnnVal, source) => {
    const lc = localCalRef.current || blankCal();

    if (fingerBpm > 0 && polarBpm > 0) {
      const m = lc.bpm;
      m.n++; m.sx += fingerBpm; m.sy += polarBpm;
      m.sxx += fingerBpm*fingerBpm; m.sxy += fingerBpm*polarBpm;
      lc.diffsBpm.push(+(fingerBpm - polarBpm).toFixed(2));
      if (lc.diffsBpm.length > 100) lc.diffsBpm.shift();
    }

    // RMSSD：在變異數域累積（x=cam², y=polar²）
    if (camRmssd > 0 && polarRmssdVal > 0) {
      const x = camRmssd*camRmssd, y = polarRmssdVal*polarRmssdVal;
      const m = lc.rmssd;
      m.n++; m.sx += x; m.sy += y; m.sxx += x*x; m.sxy += x*y;
      lc.diffsRmssd.push(+(camRmssd - polarRmssdVal).toFixed(2));
      if (lc.diffsRmssd.length > 100) lc.diffsRmssd.shift();
    }

    // SDNN：同樣在變異數域累積
    if (camSdnn > 0 && polarSdnnVal > 0) {
      const x = camSdnn*camSdnn, y = polarSdnnVal*polarSdnnVal;
      if (!lc.sdnn) lc.sdnn = { n:0, sx:0, sy:0, sxx:0, sxy:0 };
      if (!lc.diffsSdnn) lc.diffsSdnn = [];
      const m = lc.sdnn;
      m.n++; m.sx += x; m.sy += y; m.sxx += x*x; m.sxy += x*y;
      lc.diffsSdnn.push(+(camSdnn - polarSdnnVal).toFixed(2));
      if (lc.diffsSdnn.length > 100) lc.diffsSdnn.shift();
    }

    lc.agree = {
      bpm: agreement(lc.diffsBpm),
      rmssd: agreement(lc.diffsRmssd),
      sdnn: agreement(lc.diffsSdnn || []),
    };
    if (!lc.records) lc.records = [];
    lc.records.push({
      at: Date.now(),
      src: source || 'ble',
      cam:   { bpm: +fingerBpm || 0, rmssd: +camRmssd || 0, sdnn: +camSdnn || 0 },
      polar: { bpm: +polarBpm  || 0, rmssd: +polarRmssdVal || 0, sdnn: +polarSdnnVal || 0 },
    });
    if (lc.records.length > 50) lc.records.shift();
    localCalRef.current = lc;
    try { await AsyncStorage.setItem(LOCAL_CAL_KEY, JSON.stringify(lc)); } catch {}
    applyBestCal();

    // 仍上傳一份到全域，供尚未累積個人資料的新裝置當起點
    try { await post({ action:'submitCalibration', rppg: fingerBpm, polar: polarBpm }); } catch {}
  };

  // ── Init ───────────────────────────────────────────────────────────────────
  //
  // 相機請求一定要放在所有 await 之前。
  //
  // Chrome for Android 只在「使用者剛剛做過動作」的時候才會跳權限視窗。
  // 使用者點進 HRV 頁的那一下就是那個動作，而這個 effect 就跟著那一下跑 ——
  // 所以在這裡第一行送出請求，手勢還在，視窗會跳。
  //
  // 一旦先 await getPyodide()（下載加編譯 WASM，動輒好幾秒）才要相機，
  // 手勢早就過期，Chrome 不會跳視窗，使用者只看到黑圓圈。
  // iOS Safari 不管這件事，所以 iPhone 兩種寫法都正常 —— 這是 Android 專屬的坑。
  //
  // 這不是新發現：a5f9758（7/26）就是這樣寫的，註解寫著
  // 「提前請求鏡頭權限，不等 AI 引擎」，同一天稍晚的 26745df 又改回串行。
  //
  // 並行是安全的：拿到的 stream 存進 rearStreamRef，畫面上的 <video>
  // 是 rearContainerRef 掛載時才建立的，那個 callback 會把 ref 裡的
  // stream 補接上去（見下面 rearContainerRef）。
  React.useEffect(() => {
    let cancelled = false;
    const init = async () => {
      // ↓ 第一行，不 await。startCameras 內部自己處理所有錯誤，
      //   不會把 init 的 try 拉進 catch。
      const camPromise = startCameras().catch(() => {});
      try {
        const pyResult = await getPyodide(
          AURA_ENGINE_URL,
          (msg, pct) => { if (!cancelled) { setLoadMsg(msg); setLoadPct(pct); } }
        );
        if (cancelled) return;
        pyProcessRef.current = pyResult.processData;
        pyPoolRef.current    = pyResult.hrvFromPool || null;
        await loadLocalCal();   // 先讀本機個人化係數
        // 最多等 4 秒。抓不到只是少了全域備援係數（本機係數照常運作），
        // 但無限等下去會讓載入頁永遠停在那裡。
        await Promise.race([
          fetchCal(),                                  // 再抓全域值當備援
          new Promise(r => setTimeout(r, 4000)),
        ]);
        if (!cancelled) { setLoadPct(100); setLoadStep('ready'); }
        await camPromise;       // 不是為了等它，是為了不留下沒人接的 promise
      } catch (e) {
        if (!cancelled) { setLoadStep('error'); setLoadMsg('載入失敗：' + e.message); }
      }
    };
    init();
    return () => { mountedRef.current = false; cancelled = true; cleanup(); };
  }, []);

  // 衛教卡片輪播
  React.useEffect(() => {
    if (loadStep === 'ready') return;
    const id = setInterval(() => setEduIdx(i => (i + 1) % 4), 4000);
    return () => clearInterval(id);
  }, [loadStep]);

  // 頁面重置
  const resetToMeasure = () => {
    setBpm(0);
    setRmssd(0);
    setSdnn(0);
    setTimerMode('idle');
    setTimerLeft(0);
    setTimerElapsed(0);
    setReportData(null);
    setPanel(0);
    setFingerStatus({ msg:'等待手指...', err:false });
    setSqi(0);
    setMotionFree(true);
    fingerRgbBuf.current  = [];
    fingerTsBuf.current   = [];
    ppgHistoryRef.current = [];
    recordWaveBuf.current  = [];
    totalSamplesRef.current = 0;
    setPpgWaveform([]);
    sessionStats.current = { bpm:[], rawBpm:[], rmssd:[], rawRmssd:[], sdnn:[], polarRaw:[], polarRmssd:[], polarSdnn:[], goodQ:[] };
    rrPoolRef.current = { t: [], rr: [], lastT: 0 };
    poolHrvRef.current = { rmssd:0, sdnn:0, n:0, span:0, pairs:0, jitter:-1 };
    lastPoolT.current = 0; lastBpmT.current = 0;
    setScreen('dashboard');
  };

  // ── Torch + 手動對焦（Android / iOS 雙平台）────────────────────────────────
  const applyTorchAndFocus = async (track) => {
    try {
      const cap = track.getCapabilities?.() || {};
      // Android Chrome 不在 getCapabilities 回報 torch，直接強制嘗試
      try {
        await track.applyConstraints({ advanced: [{ torch: true }] });
        setTorchOn(true);
        setTorchAvail(true);
      } catch {
        // 裝置不支援 torch（iOS Safari 就是這種）。
        // 記下來讓畫面不要渲染那顆按鈕 —— 按不動的按鈕比沒有更糟。
        setTorchAvail(false);
      }
      // iOS：手動對焦，而且要對到「遠」不是「近」。
      //
      // 原本寫 focusDistance: 0.0 —— 那是最近距離，也就是微距。
      // 手指貼在鏡頭上時，這等於主動告訴系統「我要拍很近的東西」，
      // 而 iPhone 對這個情境的標準反應就是切到超廣角鏡頭做 Macro。
      // 我們反而是在幫它下決定切鏡頭。
      //
      // PPG 只需要平均亮度，畫面糊掉完全沒關係（甚至更好，等於空間平均）。
      // 所以把對焦推到最遠，讓系統沒有理由啟動微距。
      if (!isAndroid && cap.focusMode?.includes?.('manual')) {
        try {
          const far = cap.focusDistance?.max;
          await track.applyConstraints({
            advanced: [far != null ? { focusMode: 'manual', focusDistance: far }
                                   : { focusMode: 'manual' }],
          });
        } catch {}
      }
    } catch {}
  };

  // ── 手電筒開關 ────────────────────────────────────────────────────────────
  // 注意：手指 PPG 幾乎一定要開燈。sampleFingerROI 的接觸判定要求
  // R > 100 且紅光主導，沒有補光的話大多數手機根本偵測不到手指。
  // 這顆按鈕主要是給「鏡頭發燙」「反光過曝」時排除問題用的。
  const toggleTorch = async () => {
    const track = rearStreamRef.current?.getVideoTracks?.()[0];
    if (!track) return;
    const next = !torchOn;
    try {
      await track.applyConstraints({ advanced: [{ torch: next }] });
      setTorchOn(next);
    } catch {
      setTorchAvail(false);
    }
  };

  // ── 記住「哪一顆鏡頭量得到」────────────────────────────────────────────────
  //
  // 讀取用 localStorage 的同步介面，不用 AsyncStorage —— startCameras 必須是
  // 使用者手勢之後的第一件事（見 Init 的長註解），前面不能插入任何 await，
  // 插了手勢就過期、Chrome 不跳權限視窗。
  // AsyncStorage 那一份是給沒有 localStorage 的原生殼用的備援，開頁後補讀。
  const readLensSync = () => {
    try {
      const raw = (typeof window !== 'undefined' && window.localStorage)
        ? window.localStorage.getItem(LENS_KEY) : null;
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  };
  const rememberLens = (deviceId, label, score) => {
    if (!deviceId) return;
    const rec = { deviceId, label: label || '', score: score || 0, ts: Date.now() };
    rememberedLensRef.current = rec;
    const json = JSON.stringify(rec);
    try { if (typeof window !== 'undefined' && window.localStorage) window.localStorage.setItem(LENS_KEY, json); } catch {}
    try { AsyncStorage.setItem(LENS_KEY, json).catch(() => {}); } catch {}
  };
  React.useEffect(() => {
    if (rememberedLensRef.current) return;   // localStorage 已經讀到了
    AsyncStorage.getItem(LENS_KEY)
      .then(raw => { if (raw && !rememberedLensRef.current) rememberedLensRef.current = JSON.parse(raw); })
      .catch(() => {});
  }, []);

  // ── 後鏡頭啟動 ────────────────────────────────────────────────────────────
  const startCameras = async () => {
    // 上次探測挑中的那一顆優先。只給 deviceId、不帶 facingMode ——
    // 兩個一起給的時候，瀏覽器可能又解析回別的裝置（iOS 那段有同樣的註解）。
    // deviceId 會在清除網站資料後失效，那時這一條會丟 OverconstrainedError，
    // 下面的迴圈自然往後退到一般的 facingMode 條件，不必特別處理。
    const remembered = rememberedLensRef.current || readLensSync();
    rememberedLensRef.current = remembered;
    const rearConstraints = [
      ...((isAndroid && remembered && remembered.deviceId) ? [{
        video:{ deviceId:{ exact: remembered.deviceId },
                width:{ideal:640}, height:{ideal:480}, frameRate:{ideal:30} },
        audio:false,
      }] : []),
      { video:{ facingMode:{ exact:'environment' }, width:{exact:640}, height:{exact:480}, frameRate:{ideal:30,max:30} }, audio:false },
      { video:{ facingMode:{ exact:'environment' }, width:{ideal:640}, height:{ideal:480}, frameRate:{ideal:30} }, audio:false },
      { video:{ facingMode:'environment' }, audio:false },
      { video:true, audio:false },
    ];
    setCamErr('');
    setCamDiag('');

    // 問瀏覽器「這個權限現在是什麼狀態」，但**不等它**。
    //
    // granted 卻拿不到串流 = 相機被別的程式佔用；
    // denied = 之前被拒過或被瀏覽器擋掉；prompt = 應該要跳視窗。
    // 這個值是唯一能分辨這三種情況的東西，一定要留著。
    //
    // 但它不能擋在 getUserMedia 前面。使用者手勢（transient user activation）
    // 是有時效的，任何排在 getUserMedia 之前的 await 都是在花那個時效。
    // permissions.query 通常一兩毫秒就回來，理論上不會花完 —— 但這種
    // 「理論上沒問題」的東西不值得留著當一個可能性。
    // 改成先發問、最後才收結果：getUserMedia 變成手勢之後的第一件事。
    let permBefore = 'unknown';
    const permProbe = (async () => {
      try { return (await navigator.permissions.query({ name: 'camera' })).state; }
      catch (e) { return 'n/a'; }
    })();
    permProbe.then(v => { permBefore = v; });

    // ── 先擋掉「連問都不會問」的兩種情況 ────────────────────────────────
    //
    // getUserMedia 只在安全內容（HTTPS 或 localhost）下存在。
    // 用區網 IP 開發（http://192.168.x.x:8081）時 navigator.mediaDevices
    // 直接是 undefined —— 呼叫會丟 TypeError，權限請求從來沒發出去，
    // 所以系統設定裡當然「未允許任何權限」，而且沒有東西可以改。
    // 這不是使用者拒絕，是瀏覽器根本不讓問。
    if (typeof window !== 'undefined' && window.isSecureContext === false) {
      setCamErr('這個網址不是 HTTPS，瀏覽器不允許使用相機，也不會跳出權限詢問。\n'
        + '請改用 https 的網址開啟（手機上用區網 IP 測試會踩到這個限制）。');
      return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setCamErr('這個瀏覽器沒有提供相機介面。\n'
        + '如果是從 App 內建瀏覽器（LINE、FB、IG）開啟的，請改用 Chrome 或 Safari。');
      return;
    }

    let rearStream = null;
    let lastErr = null;

    // getUserMedia 有一個容易被忽略的狀態：它可以永遠不回來。
    // 權限視窗被擱置時，Promise 既不 resolve 也不 reject，下面的 catch
    // 從來不會執行 —— 畫面就停在黑圓圈，連錯誤訊息都沒有。
    //
    // 這個計時器只負責讓畫面有話可說。它不會把原本的請求砍掉 ——
    // 使用者有可能只是慢慢才按「允許」，砍掉會把已經到手的權限丟掉。
    const hangTimer = setTimeout(() => {
      if (!mountedRef.current || rearStream) return;
      setCamDiag('pending >12s / perm=' + permBefore);
      setCamErr('相機請求送出去了，但瀏覽器沒有跳出詢問視窗，也沒有回應。'
        + '\n\n請點下面那顆按鈕再試一次 —— 由你點擊觸發的請求一定會跳視窗。');
    }, 12000);

    // 耗時是判斷「有沒有真的跳過視窗」的線索：
    // 使用者真的看到視窗才按拒絕，一定是好幾秒；
    // 幾十毫秒就回 NotAllowedError，代表根本沒問，是瀏覽器自己擋掉的。
    const t0 = Date.now();
    for (const c of rearConstraints) {
      try { rearStream = await navigator.mediaDevices.getUserMedia(c); break; }
      catch (e) { lastErr = e; }
    }
    const elapsed = Date.now() - t0;
    clearTimeout(hangTimer);
    // 拖過 12 秒但最後還是拿到了 —— 把剛剛那段話收掉，不要留在畫面上嚇人
    if (rearStream) { setCamErr(''); setCamDiag(''); }

    // 四個 constraint 都失敗才判斷原因。分開講，因為使用者要做的事完全不同。
    if (!rearStream) {
      const n = lastErr?.name || '';
      // 現在才收 permissions.query 的結果 —— 請求已經送出去了，
      // 等它多久都不影響手勢。
      permBefore = await permProbe;
      setCamDiag((n || '未知') + ' / ' + elapsed + 'ms / perm=' + permBefore);

      // 沒跳視窗就被拒 = 瀏覽器自己擋掉的（Chrome 的 embargo）。
      // 這種封鎖不會在設定的相機清單裡留下項目，所以照一般的
      // 「去設定開權限」指示走，使用者永遠找不到那一項可以改。
      if (n === 'NotAllowedError' && elapsed < 400) {
        setCamErr('瀏覽器把相機請求擋掉了，沒有跳出詢問視窗。'
          + '\n\n先點下面那顆按鈕試一次。還是不行的話：'
          + '\n' + (isHomeApp
              ? '設定 → 應用程式 → 這個 App → 儲存空間 → 清除資料，然後重開。'
              : 'Chrome ⋮ → 設定 → 網站設定 → 全部網站 → 找到這個網址 → 清除並重設。')
          + '\n\n（這種封鎖不會出現在「網站設定 → 相機」的清單裡，'
          + '所以在那裡找不到東西可以改是正常的。）');
        return;
      }

      if (n === 'NotAllowedError' || n === 'SecurityError') {
        // 「去哪裡開權限」三個平台完全不一樣，寫錯路徑比不寫還糟 ——
        // 使用者會照著找，找不到就以為 App 壞了。
        let how;
        if (isIOS && isHomeApp) {
          how = '設定 → 找到這個 App → 相機 → 開啟。\n'
              + '（從主畫面圖示開啟時，權限屬於這個 App，不在 Safari 的設定裡）';
        } else if (isIOS) {
          how = '網址列左邊的「ᴀA」→ 網站設定 → 相機 → 允許。\n'
              + '或：設定 → Safari → 相機 → 允許。';
        } else if (isAndroid) {
          // 三個位置由上往下查。第一個是全域開關 —— 它被設成「封鎖」時，
          // 每個網站都會直接被拒、而且不會在清單裡留下任何一筆，
          // 所以在「網站設定 → 相機」裡找不到這個網址是正常的。
          how = '① Chrome ⋮ → 設定 → 網站設定 → 相機\n'
              + '　　要是「先詢問」，不能是「已封鎖」\n\n'
              + '② Android 設定 → 應用程式 → Chrome → 權限 → 相機 → 允許\n'
              + '　　（是 Chrome 的權限，不是這個 App 的）\n\n'
              + '③ 如果 ① 的清單裡有這個網址被封鎖，點進去改成允許\n\n'
              + '這個 App 是「加入主畫面」的捷徑，不是獨立 App，'
              + '所以在「設定 → 應用程式」裡找不到它是正常的。';
        } else {
          how = '網址列左邊的鎖頭圖示 → 權限 → 相機 → 允許，然後重新整理。';
        }
        setCamErr('相機權限被拒絕。\n' + how);
      } else if (n === 'NotFoundError' || n === 'OverconstrainedError') {
        setCamErr('找不到可用的後鏡頭。\n'
          + '如果這台裝置沒有後鏡頭，或相機正被其他 App 佔用，都會是這個結果。');
      } else if (n === 'NotReadableError') {
        setCamErr('相機被其他程式佔用中。\n'
          + '請關掉其他正在用相機的 App（相機、視訊、掃碼）再試一次。');
      } else {
        setCamErr(`相機無法啟動${n ? `（${n}）` : ''}。\n`
          + '請重新整理頁面再試一次。');
      }
      return;
    }
    // ── iOS：把虛擬多鏡頭裝置換成實體廣角鏡頭 ──────────────────────────
    //
    // facingMode:'environment' 在 iPhone 上拿到的是「虛擬」多鏡頭裝置
    // （label 會是 Back Dual Wide / Back Triple Camera）。
    // 會自動在廣角與超廣角之間切換的，就是這個虛擬裝置本身 ——
    // 它的設計目的就是「幫使用者選鏡頭」，所以任何 constraint 都擋不住。
    //
    // iOS 16.3 之後 enumerateDevices 會列出各個實體鏡頭，其中 label 為
    // 純粹「Back Camera」的那一顆就是實體廣角，它不會自己切換。
    //
    // 只有在偵測到目前拿到的是虛擬裝置時才重開一次。
    // 權限此時已經授予，不會再跳一次提示（交接文件裡那個「權限跳兩次」
    // 是因為在授權流程還沒結束就重開，這裡是在拿到串流之後才判斷）。
    if (rearStream && !isAndroid) {
      try {
        const cur = rearStream.getVideoTracks()[0];
        const label = cur?.label || '';
        const isVirtual = /dual|triple|雙|三/i.test(label);
        if (isVirtual) {
          const devs = await navigator.mediaDevices.enumerateDevices();
          const wide = devs.find(d =>
            d.kind === 'videoinput' &&
            /back camera|後置相機|背面相機/i.test(d.label) &&
            !/ultra|wide angle|dual|triple|tele|超廣角|望遠/i.test(d.label));
          if (wide && wide.deviceId) {
            const alt = await navigator.mediaDevices.getUserMedia({
              // 只給 deviceId，不要再帶 facingMode ——
              // 兩個一起給的時候，瀏覽器可能又解析回虛擬裝置。
              video: { deviceId: { exact: wide.deviceId },
                       width: { ideal: 640 }, height: { ideal: 480 },
                       frameRate: { ideal: 30 } },
              audio: false,
            });
            rearStream.getTracks().forEach(t => t.stop());
            rearStream = alt;
          }
        }
      } catch {
        // 換不成就沿用原本那條串流，功能不受影響
      }
    }

    if (rearStream) {
      // 只記錄 deviceId，不要再取一次串流。
      //
      // 原本 iOS 這條會 stop() 掉剛拿到的串流、再用 exact deviceId 重新
      // getUserMedia 一次 —— 那就是「權限跳兩次、相機指示燈閃兩下」的來源。
      // 而且它拿回來的是同一顆鏡頭（deviceId 本來就是從第一個串流讀的），
      // 等於付出兩次權限提示的代價換一個已經成立的結果。
      //
      // 防 Macro 自動切換的保護沒有消失：setupLensLock() 監聽 track.onended，
      // iOS 真的強制切換時會用記下來的 deviceId 自動重連。
      // 那條路才是實際在擋的機制，這裡的預先重鎖是多餘的。
      if (!isAndroid) {
        try {
          const track = rearStream.getVideoTracks()[0];
          const lockedId = track?.getSettings?.().deviceId;
          if (lockedId) lockedDeviceIdRef.current = lockedId;
        } catch {}
      } else {
        // Android：直接記錄 deviceId，不重鎖
        try {
          const t = rearStream.getVideoTracks()[0];
          const id = t?.getSettings?.().deviceId;
          if (id) lockedDeviceIdRef.current = id;
        } catch {}
      }
      // ── torch + focusMode 套在最終 stream 上 ────────────────────────────
      if (rearStream) {
        const track = rearStream.getVideoTracks()[0];
        if (track) await applyTorchAndFocus(track);
        setupLensLock(rearStream);
      }
    }
    // 若使用者已返回，立即停掉剛取得的 stream
    if (!mountedRef.current) {
      rearStream?.getTracks().forEach(t => t.stop());
      return;
    }
    rearStreamRef.current = rearStream;
    if (rearVideoRef.current && rearStream) {
      rearVideoRef.current.srcObject = rearStream;
      rearVideoRef.current.play().catch(()=>{});
    }
    fingerRgbBuf.current = [];
    fingerTsBuf.current = [];
    enumerateRearCameras();
    processActiveRef.current = true;
    rafRef.current = requestAnimationFrame(processFrame);
  };

  const cleanup = () => {
    processActiveRef.current = false;
    if (rafRef.current) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    // srcObject = null 是釋放相機硬體的關鍵
    if (rearVideoRef.current) {
      try { rearVideoRef.current.pause(); } catch {}
      rearVideoRef.current.srcObject = null;
    }
    if (rearStreamRef.current) {
      rearStreamRef.current.getTracks().forEach(t => {
        // 先明確關閉 torch，再 stop()，確保手電筒熄滅
        try {
          const cap = t.getCapabilities?.();
          if (cap?.torch) t.applyConstraints({ advanced: [{ torch: false }] }).catch(() => {});
        } catch {}
        try { t.stop(); } catch {}
      });
      rearStreamRef.current = null;
    }
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    timerActiveRef.current = false;
  };

  // ── 鏡頭鎖定：監聽 onended，iOS Macro 強制切換時自動重連 ──────────────────
  const setupLensLock = (stream) => {
    const track = stream.getVideoTracks()[0];
    if (!track) return;
    track.onended = async () => {
      if (!mountedRef.current || !processActiveRef.current) return;
      const id = lockedDeviceIdRef.current;
      if (!id) return;
      try {
        const newStream = await navigator.mediaDevices.getUserMedia({
          video: { deviceId:{ exact: id }, width:{ideal:640}, height:{ideal:480}, frameRate:{ideal:30} },
          audio: false,
        });
        if (!mountedRef.current) { newStream.getTracks().forEach(t=>t.stop()); return; }
        rearStreamRef.current = newStream;
        if (rearVideoRef.current) { rearVideoRef.current.srcObject = newStream; rearVideoRef.current.play().catch(()=>{}); }
        try {
          const t2 = newStream.getVideoTracks()[0];
          const cap = t2.getCapabilities?.();
          await applyTorchAndFocus(t2);
        } catch {}
        setupLensLock(newStream);  // 重新掛載 onended
      } catch {}
    };
  };
  // ── 鏡頭枚舉（取得後鏡頭列表，permission 後呼叫一次）─────────────────────
  const enumerateRearCameras = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const rear = devices.filter(d => d.kind === 'videoinput' && d.label &&
        /back|rear|environment/i.test(d.label));
      const list = rear.length > 0 ? rear : devices.filter(d => d.kind === 'videoinput');
      if (list.length > 0) {
        camerasRef.current = list;
        setCameras(list);
        // 目前用的是清單裡的哪一顆。按鈕上的 1/N 要對得起來，
        // 自動探測也是用這個索引在報進度。
        const idx = list.findIndex(d => d.deviceId === lockedDeviceIdRef.current);
        if (idx >= 0) { lensIdxRef.current = idx; setLensIdx(idx); }
      }
    } catch {}
  };

  // ── 換到指定鏡頭 ──────────────────────────────────────────────────────────
  //
  // 兩件事原本的 cycleLens 沒做，換鏡頭卻一定要做：
  //
  // ① 先解掉舊 track 的 onended。setupLensLock 掛的那個 handler 的工作是
  //    「串流被系統中斷時，用記住的 deviceId 重新連回去」—— 但我們自己
  //    呼叫 stop() 也會觸發它，於是它會拿**舊的** deviceId 把剛換好的
  //    鏡頭又搶回去。原本的寫法是先 stop 再開新的，正好踩在這個順序上。
  //
  // ② 拿到新串流之後才停舊的。失敗時舊的還活著，功能不會整個掉。
  //
  // 換鏡頭等於換了訊號源，短緩衝與 SQI 平滑歷史一律丟掉；
  // 逐拍池只有在「沒在錄製」時才清 —— 錄製中清掉等於把已經收到的拍丟了。
  const switchToDeviceId = async (deviceId) => {
    if (!deviceId) return false;
    const old = rearStreamRef.current;
    try {
      if (old) old.getTracks().forEach(t => { try { t.onended = null; } catch {} });
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId:{ exact: deviceId }, width:{ideal:640}, height:{ideal:480}, frameRate:{ideal:30} },
        audio: false,
      });
      if (!mountedRef.current) { stream.getTracks().forEach(t => t.stop()); return false; }
      if (old) old.getTracks().forEach(t => { try { t.stop(); } catch {} });
      try {
        const track = stream.getVideoTracks()[0];
        if (track) await applyTorchAndFocus(track);
      } catch {}
      const newId = stream.getVideoTracks()[0]?.getSettings?.().deviceId || deviceId;
      lockedDeviceIdRef.current = newId;
      rearStreamRef.current = stream;
      if (rearVideoRef.current) {
        rearVideoRef.current.srcObject = stream;
        rearVideoRef.current.play().catch(()=>{});
      }
      setupLensLock(stream);

      fingerRgbBuf.current  = [];
      fingerTsBuf.current   = [];
      ppgHistoryRef.current = [];
      setPpgWaveform([]);
      sqiHistRef.current = [];
      setSqi(0);
      contactFramesRef.current = 0;
      analysisStartT.current = 0; lastGoodT.current = 0;
      if (!timerActiveRef.current) {
        rrPoolRef.current  = { t: [], rr: [], lastT: 0 };
        poolHrvRef.current = { rmssd:0, sdnn:0, n:0, span:0, pairs:0, jitter:-1 };
        lastPoolT.current  = 0;
      }

      const idx = (camerasRef.current || []).findIndex(c => c.deviceId === newId);
      if (idx >= 0) { lensIdxRef.current = idx; setLensIdx(idx); }
      return true;
    } catch {
      // 換失敗：舊串流還在（我們是拿到新的才停舊的），
      // 只是剛剛把 onended 解掉了，補掛回去。
      try { if (rearStreamRef.current) setupLensLock(rearStreamRef.current); } catch {}
      return false;
    }
  };

  // ── 循環切換鏡頭（單一按鈕）────────────────────────────────────────────────
  const cycleLens = async () => {
    const list = camerasRef.current || [];
    if (list.length < 2) return;
    const nextIdx = (lensIdxRef.current + 1) % list.length;
    const cam = list[nextIdx];
    // 使用者自己動手就停掉自動探測 —— 他的選擇優先，不要被程式改回去
    lensProbeRef.current.state = 'off';
    const ok = await switchToDeviceId(cam.deviceId);
    if (ok) {
      lensIdxRef.current = nextIdx;
      setLensIdx(nextIdx);
      rememberLens(cam.deviceId, cam.label || '');
      setLensNote(`已切換到鏡頭 ${nextIdx + 1}／${list.length}，請重新把手指貼上`);
    }
  };

  // ── Android：自動找出真的量得到訊號的那一顆鏡頭 ──────────────────────────
  const finishLensProbe = async () => {
    const P = lensProbeRef.current;
    P.state = 'done';
    const best = P.best;
    if (best && best.deviceId && best.deviceId !== lockedDeviceIdRef.current) {
      P.switching = true;
      await switchToDeviceId(best.deviceId);
      P.switching = false;
    }
    if (best && best.deviceId) rememberLens(best.deviceId, best.label, best.score);
    const pct = Math.round((best ? best.score : 0) * 100);
    const list = camerasRef.current || [];
    const idx  = list.findIndex(c => best && c.deviceId === best.deviceId);
    if (pct >= PROBE_OK_SQI * 100) {
      setLensNote(`已自動選用鏡頭 ${(idx < 0 ? 0 : idx) + 1}／${list.length || 1}（訊號 ${pct}%），下次會直接用這顆`);
    } else {
      // 全部都不行的時候不要假裝解決了 —— 到這一步問題多半已經不在鏡頭上。
      // 紅光平均值可以直接分辨兩種最常見的原因：
      //   逼近 255 → 過曝。感光元件被打飽和，脈動那幾個 LSB 全被削掉，
      //              波形看起來還在跑（那是雜訊），但振幅是假的。
      //              新機的主鏡頭進光量更大，這件事只會更容易發生。
      //   剛過 100 → 光不夠，或手指沒有蓋滿鏡頭。
      const R = Math.round(lastSampleRef.current.r || 0);
      const why = R >= 248
        ? '目前紅光 ' + R + '／255，已經過曝 —— 請把手指稍微抬高一點點、'
          + '或先關掉手電筒（右下角）再試，讓畫面不要全白。'
        : '目前紅光 ' + R + '／255。請讓指腹完全蓋住鏡頭與旁邊的燈、'
          + '輕輕貼著不要用力壓（壓太用力會把微血管壓扁）。';
      setLensNote(`已試過 ${P.cand.length} 顆鏡頭，最好的一顆訊號只有 ${pct}%。` + why
        + '仍不行的話可用左下角按鈕手動換鏡頭。');
    }
  };

  const nextLensCandidate = async () => {
    const P = lensProbeRef.current;
    P.i++;
    if (P.i >= P.cand.length) { await finishLensProbe(); return; }
    const c = P.cand[P.i];
    P.cur = 0;
    P.switching = true;
    setLensNote(`正在測試鏡頭 ${P.i + 1}／${P.cand.length}…　手指請保持不動`);
    const ok = await switchToDeviceId(c.deviceId);
    P.switching = false;
    P.phaseT = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    if (!ok) { await nextLensCandidate(); }   // 這顆開不起來，跳過
  };

  /* 每次分析（約 ANALYSIS_EVERY_MS 一次）呼叫一次。
     只在「手指確實貼著」時計時 —— 沒有手指的低 SQI 不是鏡頭的錯，
     拿它當證據會在使用者還沒把手放上去的時候就開始亂換鏡頭。 */
  const driveLensProbe = (now, smoothSqi, contactOk) => {
    if (!isAndroid) return;
    const P = lensProbeRef.current;
    if (P.state === 'done' || P.state === 'off') return;
    if (timerActiveRef.current) return;   // 錄製中不換鏡頭，會把訊號切成兩截
    if (P.switching) return;

    if (P.state === 'probe') {
      const dt = now - P.phaseT;
      if (dt < PROBE_SETTLE_MS) return;          // 曝光／白平衡還沒穩，這段不算
      if (!contactOk) { P.phaseT = now - PROBE_SETTLE_MS; return; }  // 手指離開 → 這顆重新計時
      P.cur = Math.max(P.cur, smoothSqi);
      if (dt < PROBE_SETTLE_MS + PROBE_SAMPLE_MS) return;

      const c = P.cand[P.i];
      if (c && P.cur > (P.best ? P.best.score : 0)) P.best = { ...c, score: P.cur };
      // 已經夠好就不必把剩下的試完 —— 每多試一顆就是多四秒的等待
      if (c && P.cur >= PROBE_OK_SQI) { finishLensProbe(); return; }
      nextLensCandidate();
      return;
    }

    if (!contactOk) { P.contactStart = 0; return; }

    // 目前這顆就量得到 —— 記住它，這台手機以後不用再試
    if (smoothSqi >= PROBE_OK_SQI) {
      P.state = 'done';
      const id = lockedDeviceIdRef.current;
      const cam = (camerasRef.current || []).find(c => c.deviceId === id);
      if (id) rememberLens(id, cam ? cam.label : '', smoothSqi);
      setLensNote('');
      return;
    }

    if (!P.contactStart) { P.contactStart = now; P.state = 'watch'; return; }
    if (now - P.contactStart < PROBE_TRIGGER_MS) return;

    const list = (camerasRef.current || []).filter(c => c.deviceId);
    if (list.length < 2) { P.state = 'off'; return; }   // 只枚舉到一顆，沒得挑
    // 目前這顆已經知道不行，排到最後再試（它的分數用現在這個值就好）
    const curId = lockedDeviceIdRef.current;
    P.cand = [...list.filter(c => c.deviceId !== curId),
              ...list.filter(c => c.deviceId === curId)]
             .map(c => ({ deviceId: c.deviceId, label: c.label || '' }));
    P.best = curId ? { deviceId: curId, label: '', score: smoothSqi } : null;
    P.i = -1;
    P.state = 'probe';
    nextLensCandidate();
  };


  // ── 手指取樣：中央大面積平均 ─────────────────────────────────────────────
  //
  // 原本是 5 個 2x2 點 = 20 個像素。那 20 個像素決定了每一格的雜訊：
  // 單像素雜訊約 2 LSB 時，平均 20 個 → 每格約 0.5 LSB，
  // 而指尖 PPG 的脈動振幅只有 2-10 LSB。訊噪比一差，波峰頂點就找不準，
  // 而 RMSSD 是以 sqrt(真值^2 + 6*sigma^2) 被放大的 ——
  // sigma 是每拍的定位誤差。合成訊號實測：
  //     平均   20 像素 -> sigma 29.3ms -> RMSSD 74.9（真值 20.5）
  //     平均  320 像素 -> sigma  9.6ms -> RMSSD 25.0
  //     平均 5120 像素 -> sigma  8.6ms -> RMSSD 22.7
  // 像素數是整條鏈上最便宜的槓桿：不動演算法，只是別丟掉已經畫好的資料。
  //
  // 取中央區塊而不是整張畫面：邊角有暗角，脈動振幅小，
  // 平均進來只會稀釋訊號。
  // 不做飽和像素遮罩：被遮掉的像素集合會逐格改變，那本身就是新的雜訊源，
  // 比「讓飽和像素穩定地稀釋一點振幅」更糟。
  const sampleFingerROI = (ctx, cw, ch) => {
    const rx = Math.floor(cw * 0.15), ry = Math.floor(ch * 0.15);
    const rw = Math.max(1, cw - rx * 2), rh = Math.max(1, ch - ry * 2);
    let r=0,g=0,b=0,n=0;
    try {
      const d = ctx.getImageData(rx, ry, rw, rh).data;
      for (let i=0;i<d.length;i+=4) { r+=d[i]; g+=d[i+1]; b+=d[i+2]; n++; }
    } catch { return null; }
    if (n === 0) return null;
    const ar = r/n, ag = g/n, ab = b/n;
    // Red-channel dominance = finger contact (prevent false positives)
    // 手電筒透過手指：紅光高度主導（R>100, R/G>2.0, R/B>2.5）
    // 桌面/牆壁在室內燈下無法同時滿足三條件，防止假陽性
    if (ar < 100 || ar/(ag+1) < 2.0 || ar/(ab+1) < 2.5) return null;
    return { r:ar, g:ag, b:ab };
  };

  // ── PPG 波形（每秒一格，即時捲動）──────────────────────────────────────
  const WAVE_WINDOW = 150;  // 5秒 @ 30fps
  const WAVE_FPS    = 30;

  const drawPpgWaveform = (cvRef, data, absoluteTotal = 0) => {
    const canvas = cvRef?.current;
    if (!canvas || !data || data.length < 2) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.offsetWidth || 300;
    const h = canvas.offsetHeight || 100;
    if (canvas.width  !== w) canvas.width  = w;
    if (canvas.height !== h) canvas.height = h;

    const PAD = { top:6, bottom:20, left:4, right:34 };
    const cw = w - PAD.left - PAD.right;
    const ch = h - PAD.top - PAD.bottom;

    // 即時模式：永遠顯示最後 WAVE_WINDOW 個樣本
    const start = Math.max(0, data.length - WAVE_WINDOW);
    const slice = data.slice(start);
    if (slice.length < 2) return;

    const mn  = Math.min(...slice);
    const mx  = Math.max(...slice);
    const rng = (mx - mn) || 1;

    ctx.fillStyle = theme.bg;
    ctx.fillRect(0, 0, w, h);

    // 水平輔助線
    ctx.strokeStyle = theme.card || '#2a2a2a';
    ctx.lineWidth = 0.5;
    [0, 0.5, 1].forEach(f => {
      const y = PAD.top + ch * f;
      ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + cw, y); ctx.stroke();
    });

    // 垂直秒刻線（每 1 秒）
    const pxPerSample = cw / WAVE_WINDOW;
    const offsetInSec = start % WAVE_FPS;
    const firstGridI  = (WAVE_FPS - offsetInSec) % WAVE_FPS;
    ctx.setLineDash([3, 3]);
    for (let i = firstGridI; i <= WAVE_WINDOW; i += WAVE_FPS) {
      const x = PAD.left + i * pxPerSample;
      ctx.beginPath(); ctx.moveTo(x, PAD.top); ctx.lineTo(x, PAD.top + ch); ctx.stroke();
    }
    ctx.setLineDash([]);

    /* 波形（y 軸翻過來畫）

       data 裡放的是相機每一格的平均亮度（R 通道原始值，見 ppgHistoryRef）。
       亮度跟脈搏是反的：收縮期血多 → 吸掉更多光 → 畫面變暗 → 數值變小。
       原本直接照數值畫（大的在上），畫出來就是脈搏波上下顛倒 ——
       看起來像慢慢爬升然後陡降，重搏切跡會跑到上升支的下段變成一個小台階。
       教科書、血氧機、任何一張 PPG 圖都是相反的：陡升、尖峰、緩降，
       切跡在下降支上。上課拿這張圖去對 Wiggers 圖會對不起來。

       所以這裡把 y 軸翻過來，讓「血多＝往上」。只動畫圖：
       送進 hrv_engine 的是 fingerRgbBuf，這裡完全沒碰
       （引擎本來就自己取 -R，見 hrv_engine.py 的 contact_mode）。 */
    ctx.beginPath();
    ctx.strokeStyle = '#ff4d4d';
    ctx.lineWidth = 1.8;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    slice.forEach((v, i) => {
      const x = PAD.left + i * pxPerSample;
      const y = PAD.top + ((v - mn) / rng) * ch;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // 時間標籤（底部，用累計樣本數計算真實秒數）
    ctx.fillStyle = theme.textSub || '#888';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    const windowAbsStart = absoluteTotal - slice.length;
    const absOffsetInSec = windowAbsStart % WAVE_FPS;
    const absFirstGridI  = (WAVE_FPS - absOffsetInSec) % WAVE_FPS;
    for (let i = absFirstGridI; i <= WAVE_WINDOW; i += WAVE_FPS) {
      const x   = PAD.left + i * pxPerSample;
      const sec = Math.floor((windowAbsStart + i) / WAVE_FPS);
      ctx.fillText(`${sec}s`, x, h - 4);
    }

    /* 振幅標籤（右側）。上面翻了 y 軸，這兩個數字要跟著換位置，
       不然標的是相反的邊。數字本身還是原始亮度值（不是翻過來的），
       因為判斷訊號夠不夠強時看的就是這兩個數的差。 */
    ctx.textAlign = 'left';
    ctx.fillStyle = theme.textSub || '#888';
    const rx = PAD.left + cw + 3;
    ctx.fillText(mn.toFixed(0),          rx, PAD.top + 10);
    ctx.fillText(mx.toFixed(0),          rx, PAD.top + ch - 2);
  };

  // ── Frame loop ─────────────────────────────────────────────────────────────
  const processFrame = (now) => {
    if (!processActiveRef.current) return;  // cleanup 後立即停止
    rafRef.current = requestAnimationFrame(processFrame);
    if (!processActiveRef.current) {  // 雙重檢查，防止競態
      cancelAnimationFrame(rafRef.current); rafRef.current = null; return;
    }
    if (!now) now = performance.now();
    // 門檻不能設 33.3。60Hz 螢幕的 rAF 間隔是 16.667ms，兩格 = 33.333ms，
    // 只比 33.3 多 0.03ms —— performance.now() 稍微抖一下就低於門檻，
    // 這一格被丟掉，要再等一格才過，那一拍就變成 50ms（20fps）。
    //
    // 實測（收 400 個真實 rAF 時間戳，離線套用不同門檻）：
    //     門檻 33.3 → 25.0 fps，41% 的格子跳過一整輪
    //     門檻 32 以下 → 30.1 fps，0% 跳格
    // 設 30 在 60Hz（2 格 33.3ms）與 120Hz（4 格 33.3ms）都穩定拿到 30fps。
    if (now - lastFrameT.current < 30) return;
    lastFrameT.current = now;
    frameN.current++;

    const ocv  = overlayCvRef.current;
    const octx = ocv ? ocv.getContext('2d') : null;

    if (ocv) {
      const ow = ocv.offsetWidth  || 260;
      const oh = ocv.offsetHeight || 260;
      if (ocv.width  !== ow) ocv.width  = ow;
      if (ocv.height !== oh) ocv.height = oh;
    }
    if (octx) octx.clearRect(0, 0, ocv.width, ocv.height);

    // ── 後鏡頭：手指接觸式 PPG
    const rVid = rearVideoRef.current;
    const rHcv = rearHiddenCvRef.current;
    if (rearStreamRef.current && rVid && rHcv && rVid.readyState >= 2 && rVid.videoWidth) {
      // 畫布固定小尺寸，不跟著相機解析度走。
      // 原本是 rHcv.width = rVid.videoWidth —— 手機給 1080p 或 4K 時，
      // 每一格都要畫一整張大圖，高階機（相機解析度更高）反而更慢，
      // 取樣率掉到十幾 fps。而 PPG 只需要幾個像素的平均亮度，
      // 大圖完全沒有幫助。
      if (rHcv.width !== PROC_W)  rHcv.width  = PROC_W;
      if (rHcv.height !== PROC_H) rHcv.height = PROC_H;
      const ctx = rHcv.getContext('2d', { willReadFrequently:true });
      ctx.drawImage(rVid, 0, 0, PROC_W, PROC_H);

      const sample = sampleFingerROI(ctx, rHcv.width, rHcv.height);

      if (sample) {
        noContactRef.current = 0;
        lastSampleRef.current = sample;   // 診斷用：探測全失敗時要報這個數字
        contactFramesRef.current = Math.min(contactFramesRef.current + 1, 30);
        fingerRgbBuf.current.push([sample.r, sample.g, sample.b]);
        fingerTsBuf.current.push(now);

        const rValue = sample.r;
        ppgHistoryRef.current.push(rValue);
        if (ppgHistoryRef.current.length > 300) {
          ppgHistoryRef.current.shift();
        }
        setPpgWaveform([...ppgHistoryRef.current]);
        totalSamplesRef.current++;
        if (timerActiveRef.current) {
          recordWaveBuf.current.push(rValue);
          sessionRgbBuf.current.push([sample.r, sample.g, sample.b]);
          sessionTsBuf.current.push(now);
        }

        if (octx && ocv) {
          const cx = ocv.width/2, cy = ocv.height/2;
          const cr = Math.min(ocv.width, ocv.height)/2 - 15;
          octx.beginPath();
          octx.arc(cx, cy, cr, 0, Math.PI*2);
          octx.strokeStyle = 'rgba(255,255,255,0.35)';
          octx.lineWidth = 2;
          octx.setLineDash([6,4]);
          octx.stroke();
          octx.setLineDash([]);
        }
        if (contactFramesRef.current >= 8) {
          setFingerStatus({ msg: timerActiveRef.current ? '✅ 後鏡頭記錄中' : '✅ 手指訊號良好', err:false });
        }
      } else {
        contactFramesRef.current = 0;  // 離開立即重置
        noContactRef.current++;
        if (noContactRef.current >= 30) {  // ~1s 持續無接觸才清空
          fingerRgbBuf.current  = [];
          fingerTsBuf.current   = [];
          ppgHistoryRef.current = [];
          setPpgWaveform([]);
          /* 畫面上的數字也要一起收掉。
             緩衝清空之後 spanSec 會變成 0，下面那段分析就整個被跳過 ——
             於是 setSqi / setBpm 再也不會被呼叫，畫面停在手指離開前
             的最後一次結果（實測會一直顯示「訊號良好 86%」）。
             那比顯示 0% 危險：使用者以為還在量，其實什麼都沒有。

             逐拍池刻意不清。錄製中途手指滑開再放回來是常見的事，
             已經收到的拍是真的，沒有理由丟掉；池子的相鄰判定本來就是
             看時間對不對得上，中間的缺口不會被算成一對。 */
          sqiHistRef.current = [];
          setSqi(0); setMotionFree(true);
          setBpm(0); setRmssd(0); setSdnn(0);
          setSlowHint(''); pyErrRef.current = '';
          analysisStartT.current = 0; lastGoodT.current = 0; lastBpmT.current = 0;
          // 鏡頭探測的「連續貼著多久」也要跟著歸零，
          // 不然手指離開再放回來會直接跳過 8 秒的觀察期。
          if (lensProbeRef.current.state === 'watch') lensProbeRef.current.contactStart = 0;
        }
        setFingerStatus({ msg:'⚠️ 請將食指覆蓋後鏡頭', err:true });
      }

      // 緩衝區用「秒數」修剪，不用筆數。
      // 原本固定留 300 筆 —— 30fps 時是 10 秒，60fps 時只剩 5 秒，
      // 心跳間隔數量直接砍半，HRV 算不出來。
      while (fingerTsBuf.current.length > 2 &&
             (now - fingerTsBuf.current[0]) > BUF_SECONDS * 1000) {
        fingerRgbBuf.current.shift();
        fingerTsBuf.current.shift();
      }
    }

    // ── Python：BPM/RMSSD/SDNN/SQI/運動狀態
    //
    // 門檻與節奏一律用時間，不用畫格數。
    // 原本是「累積 150 格」才開始、「每 30 格」算一次 ——
    // 那假設了穩定的 30fps。實際上取樣率會因為機型、相機解析度、
    // 發熱降頻而差很多：15fps 的機器要等 10 秒才看得到第一個數字，
    // 而且傳給引擎的取樣率永遠寫死 30.0，跟真實值對不上時
    // 帶通濾波的頻段就整個偏掉，波形正常但抓不到心跳。
    const tsBuf = fingerTsBuf.current;
    const spanSec = tsBuf.length > 1 ? (now - tsBuf[0]) / 1000 : 0;
    if (spanSec > 0 && !analysisStartT.current) analysisStartT.current = now;
    if (spanSec <= 0) analysisStartT.current = 0;

    if (spanSec >= MIN_ANALYSIS_SEC &&
        now - lastAnalysisT.current >= ANALYSIS_EVERY_MS &&
        pyProcessRef.current) {
      lastAnalysisT.current = now;
      try {
        const flat = Float64Array.from(fingerRgbBuf.current.flat());
        const ts   = Float64Array.from(tsBuf);
        // 實測取樣率：用時間戳自己算，不猜
        const fpsNow = Math.max(5, Math.min(60, (tsBuf.length - 1) / spanSec));
        /* PyProxy 一定要自己釋放。
           下面那個池子的呼叫早就有 destroy 了，這一個沒有 ——
           而這一個每 0.4 秒跑一次，是池子的四倍頻率。
           漏掉的物件會一直卡在 Pyodide 的堆裡，量久了 fps 就往下掉。 */
        const _resPx = pyProcessRef.current(flat, ts, fpsNow, polarBpmRef.current,
                                            'finger', false, false);
        const res  = _resPx.toJs();
        try { _resPx.destroy && _resPx.destroy(); } catch {}
        const get  = (k) => res.get ? res.get(k) : res[k];

        /* 引擎自己攔下來的錯誤要讀出來。
           它內部 try/except 之後是「正常回傳一個 error 欄位」，不是丟例外，
           所以下面那個 catch 永遠接不到 —— 畫面上只會看到
           「訊號不足 0%」加一句猜的「取樣率偏低」，完全看不出真正的原因。
           這次「波形漂亮但心率一直是 --」就是這樣藏了整整一輪才被發現。 */
        const engErr = String(get('error') || '');
        const rawBpm = get('bpm');
        const rv   = get('rmssd');
        const sv   = get('sdnn');
        const fps  = get('real_fps');
        const sqi_val = get('sqi') || 0;
        const motion = get('motion_free') !== false;

        if (fps) setRealFps(fps);
        // 引擎說得出原因就用引擎的，不要自己猜
        engErrRef.current = engErr;
        // SQI 平滑：取最近 5 筆平均，避免單幀抖動
        sqiHistRef.current.push(sqi_val);
        if (sqiHistRef.current.length > 5) sqiHistRef.current.shift();
        const smoothSqi = sqiHistRef.current.reduce((a,b)=>a+b,0) / sqiHistRef.current.length;
        setSqi(smoothSqi);
        setMotionFree(motion);

        /* 鏡頭自動挑選（只在 Android 上動作）。
           放在這裡是因為 smoothSqi 只有這裡算得出來，而它就是判準本身。
           contactFramesRef >= 8 是既有的接觸防抖門檻，沿用同一個，
           不要另外定義一個「手指有沒有貼著」的標準。 */
        driveLensProbe(now, smoothSqi, contactFramesRef.current >= 8);

        // 品質閘門：訊號差或有晃動的樣本不納入校正，避免污染迴歸
        const qualityOk = smoothSqi >= SQI_MIN && motion;

        if (rawBpm > 0) {
          const { a, b } = calRef.current;
          const calibrated = Math.round(a*rawBpm + b);
          setBpm(calibrated);
          lastBpmT.current = now;
          if (timerActiveRef.current) {
            sessionStats.current.bpm.push(calibrated);
            // 原始值另存一份 —— 提交校正時要用這個，不能用 calibrated
            sessionStats.current.rawBpm.push(rawBpm);
            sessionStats.current.goodQ.push(qualityOk);
            if (polarBpmRef.current > 30) {
              sessionStats.current.polarRaw.push(polarBpmRef.current);
            }
          }
        } else if (now - lastBpmT.current > BPM_HOLD_MS) {
          // 只有連續好幾秒都算不出來才把心率清掉。
          // 原本是「這個視窗沒算出來就立刻 setBpm(0)」——
          // 手指好好放著、波形也在跑，心率卻一直在數字和 -- 之間閃，
          // 看起來像量測壞掉，其實只是某一個 12 秒視窗品質差了一點。
          setBpm(0);
        }

        /* ── 逐拍進池 ──────────────────────────────────────────────
           引擎交出來的是「這個視窗裡通過清洗的那幾拍」，
           帶著整場錄製的絕對時刻，所以視窗重疊送重複的拍沒關係，
           hrv_from_pool 會依時間去重。 */
        const bt = get('beat_t'), br = get('beat_rr');
        const pool = rrPoolRef.current;
        if (bt && br && bt.length === br.length && bt.length > 0) {
          /* 進池之前先去重 —— 這一步不能省。
             視窗是 12 秒、每 0.4 秒算一次，所以連續兩次分析有 96% 的內容
             是同一段訊號，同一拍會被交出來大約 30 次。
             不去重的話池子每秒長 30 筆而不是 1 筆，一分鐘就是 1800 筆；
             每次再整包丟給 Python 排序去重，畫面的 fps 會直接垮掉
             （實測 iOS 上約 10 秒後就崩，然後因為時間戳變得不規則，
             連帶被判成「偵測到運動偽影」而且 SQI 掉到 0 —— 一個原因，
             三個看起來不相干的症狀）。 */
          for (let i = 0; i < bt.length; i++) {
            const tb = +bt[i];
            if (tb > pool.lastT + 60) {      // 60ms 內視為同一拍
              pool.t.push(tb); pool.rr.push(+br[i]); pool.lastT = tb;
            }
          }
          // 沒在計時的預覽階段只留最近 POOL_PREVIEW_SEC 秒；
          // 計時中一路累積不丟 —— 池子就是整場，即時值會收斂到結算值。
          if (!timerActiveRef.current && pool.t.length) {
            const cut = now - POOL_PREVIEW_SEC * 1000;
            let k = 0; while (k < pool.t.length && pool.t[k] < cut) k++;
            if (k > 0) { pool.t.splice(0, k); pool.rr.splice(0, k); }
          }
          // 計時中也要有上限。大部分人量 1 分鐘，池子＝整場；
          // 但萬一有人量很久，陣列不能無限長。
          if (pool.t.length) {
            const cutMax = now - POOL_MAX_SEC * 1000;
            let k2 = 0; while (k2 < pool.t.length && pool.t[k2] < cutMax) k2++;
            if (k2 > 0) { pool.t.splice(0, k2); pool.rr.splice(0, k2); }
          }
        }

        // 池子算得出來就用池子的，算不出來才退回這個視窗自己的值
        let rvUse = rv, svUse = sv;
        /* 池子的 HRV 不用每 0.4 秒算一次 —— 它是幾十秒的累積量，
           跳動本來就慢，而每一次呼叫都要把整包陣列送進 Pyodide。
           節流到 POOL_EVERY_MS，中間沿用上一次的結果。 */
        if (pyPoolRef.current && pool.t.length >= 3 &&
            now - lastPoolT.current >= POOL_EVERY_MS) {
          lastPoolT.current = now;
          let px = null;
          try {
            // 傳 Float64Array：Pyodide 對 TypedArray 是零複製，
            // 普通 Array 每個元素都要包一層
            px = pyPoolRef.current(Float64Array.from(pool.t), Float64Array.from(pool.rr));
            const h = px.toJs();
            const hg = (k) => (h.get ? h.get(k) : h[k]);
            poolHrvRef.current = {
              rmssd: hg('rmssd') || 0, sdnn: hg('sdnn') || 0,
              n: hg('n') || 0, span: hg('span_sec') || 0,
              pairs: hg('pairs') || 0, jitter: hg('jitter_ms') ?? -1,
            };
          } catch { /* 舊引擎或呼叫失敗：維持上一次的結果 */ }
          // PyProxy 一定要自己釋放，不然每 1.5 秒漏一個，
          // 量個幾分鐘記憶體就爬上去了（fps 也會跟著掉）
          finally { try { px && px.destroy && px.destroy(); } catch {} }
        }
        const ph = poolHrvRef.current;
        if (ph.rmssd > 0) rvUse = ph.rmssd;
        if (ph.sdnn  > 0) svUse = ph.sdnn;

        if (rvUse > 0) {
          // RMSSD 走變異數域校正（樣本不足時原值回傳）
          const rvCal = correctRmssd(rvUse);
          setRmssd(rvCal);
          if (timerActiveRef.current) {
            sessionStats.current.rmssd.push(rvCal);
            // 原始 RMSSD 另存：擬合必須用它，否則會和 BPM 一樣形成回饋迴路
            sessionStats.current.rawRmssd.push(rvUse);
          }
        }

        if (svUse > 0) {
          setSdnn(svUse);
          if (timerActiveRef.current) {
            sessionStats.current.sdnn.push(svUse);
          }
        }

        // 有算出東西就把提示清掉
        if (rawBpm > 0 || rvUse > 0) {
          pyErrRef.current = '';
          engErrRef.current = '';
          lastGoodT.current = now;
          setSlowHint('');
        }
      } catch (e) {
        // 原本是 catch {} —— 分析每次都拋錯的話畫面完全看不出來，
        // 只會覺得「有波形但永遠沒數字」。記下來，下面決定要不要講。
        pyErrRef.current = String(e && e.message ? e.message : e).slice(0, 120);
      }
    }

    // 訊號累積夠久卻始終沒有數字 → 說明原因，不要讓使用者對著波形乾等
    const quietSince = Math.max(analysisStartT.current, lastGoodT.current);
    if (analysisStartT.current && now - quietSince > SLOW_WARN_MS) {
      const fpsNow = spanSec > 0 ? (tsBuf.length - 1) / spanSec : 0;
      /* 原本不管什麼原因都寫「取樣率偏低」，連 30fps 也照寫 ——
         那是把唯一一個看得到的數字拿來當代罪羔羊。
         真的低才講取樣率；引擎講得出原因就用引擎的話。 */
      const eng = engErrRef.current;
      setSlowHint(
        pyErrRef.current      ? `分析發生錯誤：${pyErrRef.current}`
      : eng                   ? `分析沒有結果：${eng.split('\n').filter(Boolean).slice(-1)[0].slice(0, 90)}`
      : fpsNow < MIN_OK_FPS   ? `取樣率偏低（${fpsNow.toFixed(0)} fps），訊號不足以判讀心跳。請壓穩手指、避免移動。`
      :                         `取樣率正常（${fpsNow.toFixed(0)} fps）但抓不到穩定的心跳，請把手指壓穩、蓋滿鏡頭。`);
    }
  };

  // ── 整段重算 ────────────────────────────────────────────────────────────
  //  即時畫面用的是 12 秒滾動視窗，而結算不能拿那些視窗值去平均。
  //
  //  SDNN 量的是「整段錄製的總變異」，包含週期 7～25 秒的慢波
  //  （呼吸性竇性心律不整、血壓調節）。12 秒的視窗物理上裝不下這些成分，
  //  所以每個視窗算出來的 SDNN 都偏低，再怎麼平均也補不回來 ——
  //  這不是校正係數的問題，是把短窗的量拿去對照長窗的量。
  //  實測 12 秒窗約只有 60 秒窗的四成，跟回報的 24 vs 60 吻合。
  //
  //  RMSSD 受影響小得多（它只看相鄰兩拍的差），所以偏差沒那麼大。
  //
  //  正確做法是拿整段訊號重算一次，跟 Polar 的算法對齊。
  const analyseWholeSession = () => {
    try {
      const n = sessionTsBuf.current.length;
      if (n < 60 || !pyProcessRef.current) return null;
      const spanSec = (sessionTsBuf.current[n-1] - sessionTsBuf.current[0]) / 1000;
      if (spanSec < 10) return null;
      const fps = Math.max(5, Math.min(60, (n - 1) / spanSec));
      const flat = Float64Array.from(sessionRgbBuf.current.flat());
      const ts   = Float64Array.from(sessionTsBuf.current);
      const res  = pyProcessRef.current(flat, ts, fps, polarBpmRef.current, 'finger', false, false).toJs();
      const get  = (k) => res.get ? res.get(k) : res[k];
      return { bpm: get('bpm') || 0, rmssd: get('rmssd') || 0, sdnn: get('sdnn') || 0, spanSec };
    } catch (e) { return null; }
  };

  // Polar 同理：polarRRBuf 一直累積整段的 RR，
  // 但原本回報的是「每次計算結果的平均」—— 早期那幾次只有少少幾拍，
  // 算出來的 SDNN 偏低，把它們平均進去等於自己把數字拉低。
  // 直接用整段 RR 重算才是 Polar 官方口徑。
  const polarWholeSession = () => {
    const valid = polarRRBuf.current.filter(r => r >= 333 && r <= 2000);
    if (valid.length < 8) return null;
    const mean  = valid.reduce((a,b)=>a+b,0)/valid.length;
    const sdnn_ = Math.round(Math.sqrt(valid.reduce((s,r)=>s+(r-mean)**2,0)/valid.length));
    const diffs = valid.slice(1).map((r,i)=>r-valid[i]);
    const rmssd_= Math.round(Math.sqrt(diffs.reduce((s,d)=>s+d*d,0)/diffs.length));
    return { rmssd: rmssd_, sdnn: sdnn_, bpm: Math.round(60000/mean), beats: valid.length };
  };

  // ── Timer ──────────────────────────────────────────────────────────────────
  //
  // 結算：自動到時與手動停止走同一條路。
  //
  // 原本只有「倒數歸零」那條分支會提交校正配對，手動按停止完全不會 ——
  // 使用者提早結束的那些量測，Polar 配對就這樣默默丟掉了。
  // 抽成同一個函式之後兩條路徑行為一致。
  const finishTimer = () => {
    sfx('complete');            // 自動到時與手動停止都會走這裡
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    timerActiveRef.current = false;
    setTimerMode('done');
    setTimerLeft(0);

    const ss = sessionStats.current;
    const avg = arr => arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0;
    const rawGood = ss.rawBpm.filter((_, i) => ss.goodQ[i] !== false);

    // 未達可用時長的量測不提交校正配對。
    //
    // 改成「隨時可結束」之後多了一條路：使用者量 20 秒就按結束。
    // 那種長度的 RMSSD/SDNN 本來就不穩，拿去跟 Polar 配對等於
    // 餵雜訊給迴歸 —— 而且是看不出來的那種，係數會慢慢歪掉。
    //
    // 原本只有倒數自然歸零才會提交，時長天然有保障；現在要自己擋。
    if (timerElapsed >= MIN_VALID_SEC && ss.polarRaw.length >= 5 && rawGood.length >= 5) {
      // 配對一律用「整段」的值。
      // 原本送的是各視窗結果的平均 —— 那等於拿短窗的量去對照 Polar 的長窗值，
      // 迴歸學到的會是「視窗長度差異」而不是「相機 vs 胸帶的差異」，
      // 校正係數因此永遠對不準。
      const camWhole   = analyseWholeSession();
      const polarWhole = polarWholeSession();
      const camRawR = ss.rawRmssd.filter((_, i) => ss.goodQ[i] !== false);
      const camR    = camWhole?.rmssd || avg(camRawR.length ? camRawR : ss.rawRmssd);
      const camS    = camWhole?.sdnn  || avg(ss.sdnn);
      const polarR  = polarWhole?.rmssd || avg(ss.polarRmssd);
      const polarS  = polarWhole?.sdnn  || avg(ss.polarSdnn);
      const camB    = camWhole?.bpm   || avg(rawGood);
      const polarB  = polarWhole?.bpm || avg(ss.polarRaw);
      submitCal(camB, polarB, camR, polarR, camS, polarS).then(fetchCal);
    }
  };

  // secs === 0（UNLIMITED）代表不設終點，由使用者自己按結束。
  const startTimer = (secs) => {
    if (timerRef.current) clearInterval(timerRef.current);
    sessionStats.current = { bpm:[], rawBpm:[], rmssd:[], rawRmssd:[], sdnn:[], polarRaw:[], polarRmssd:[], polarSdnn:[], goodQ:[] };
    polarRRBuf.current = [];
    fingerRgbBuf.current = [];
    fingerTsBuf.current = [];
    ppgHistoryRef.current = [];
    recordWaveBuf.current  = [];  // 重置全程錄製 buffer
    sessionRgbBuf.current  = [];
    sessionTsBuf.current   = [];
    setPpgWaveform([]);
    // 逐拍池也重來。預覽期間累積的拍是「還沒正式開始」的心跳，
    // 混進來會讓這一場的 HRV 帶著前面調整手指時的抖動。
    rrPoolRef.current = { t: [], rr: [], lastT: 0 };
    poolHrvRef.current = { rmssd:0, sdnn:0, n:0, span:0, pairs:0, jitter:-1 };
    lastPoolT.current = 0;
    timerActiveRef.current = true;
    setTimerMode('running');
    setTimerElapsed(0);
    setTimerLeft(secs);

    let elapsed = 0;
    timerRef.current = setInterval(() => {
      elapsed += 1;
      setTimerElapsed(elapsed);
      if (secs > 0) setTimerLeft(Math.max(0, secs - elapsed));
      if (secs > 0 && elapsed >= secs) finishTimer();
    }, 1000);
  };

  // 讀目前累積了幾組配對，讓使用者知道還差幾組
  const refreshCalCount = React.useCallback(() => {
    const lc = localCalRef.current;
    setCalCount({
      bpm:   lc?.bpm?.n   || 0,
      rmssd: lc?.rmssd?.n || 0,
      sdnn:  lc?.sdnn?.n  || 0,
    });
  }, []);



  const genReport = async () => {
    const ss = sessionStats.current;
    if (!ss.bpm.length && !ss.rmssd.length) return;
    const avg = arr => arr.length ? Math.round(arr.reduce((a,b)=>a+b,0)/arr.length) : 0;
    const whole      = analyseWholeSession();
    const polarWhole = polarWholeSession();
    const usePolarHRV = !!polarWhole || ss.polarRmssd.length >= 3;
    // 心率同樣以 Polar 為優先：它和 RMSSD/SDNN 一樣是黃金標準，
    // 原本只有 HRV 用 Polar、心率卻仍用相機估計值，兩者來源不一致。
    // 注意這不影響校正 —— 校正用的是另存的 ss.rawBpm。
    const usePolarBpm = ss.polarRaw.length >= 3;

    // 優先序：Polar 整段 > Polar 視窗平均 > 相機整段 > 相機視窗平均。
    // 「整段」永遠優於「平均」—— 平均是把短窗的偏低值攤開，不是估計值。
    const camRmssd = whole && whole.rmssd > 0 ? correctRmssd(whole.rmssd) : avg(ss.rmssd);
    const camSdnn  = whole && whole.sdnn  > 0 ? correctSdnn(whole.sdnn)    : avg(ss.sdnn);
    const camBpm   = whole && whole.bpm   > 0
      ? Math.round(calRef.current.a * whole.bpm + calRef.current.b)
      : avg(ss.bpm);

    const reportData = {
      timestamp: Date.now(),
      // 這一段量了多久。未滿 MIN_VALID_SEC 的紀錄不該拿來跟別次比較，
      // 存下來才有辦法在歷史頁標示出來。
      durationSec: timerElapsed,
      valid: timerElapsed >= MIN_VALID_SEC,
      bpm:   usePolarBpm  ? (polarWhole?.bpm   || avg(ss.polarRaw))   : camBpm,
      rmssd: usePolarHRV  ? (polarWhole?.rmssd || avg(ss.polarRmssd)) : camRmssd,
      sdnn:  usePolarHRV  ? (polarWhole?.sdnn  || avg(ss.polarSdnn))  : camSdnn,
      hrvSource: usePolarHRV ? 'Polar' : '手指',
      // 未校正的整段相機值。手動輸入 Polar 數值做校正時要用這個 ——
      // 送校正後的值會讓迴歸在自己的輸出上再擬合，係數永遠收斂不到真值。
      camRaw: whole
        ? { bpm: whole.bpm || 0, rmssd: whole.rmssd || 0, sdnn: whole.sdnn || 0 }
        : { bpm: avg(ss.rawBpm), rmssd: avg(ss.rawRmssd), sdnn: avg(ss.sdnn) },
      // 診斷用：整段 vs 視窗平均的差距，之後回頭查數值偏低時看得到證據
      wholeVsWindow: {
        camWhole:  whole      ? { rmssd: whole.rmssd, sdnn: whole.sdnn, sec: Math.round(whole.spanSec) } : null,
        camWindow: { rmssd: avg(ss.rawRmssd), sdnn: avg(ss.sdnn) },
        polarWhole: polarWhole ? { rmssd: polarWhole.rmssd, sdnn: polarWhole.sdnn, beats: polarWhole.beats } : null,
        polarWindow: { rmssd: avg(ss.polarRmssd), sdnn: avg(ss.polarSdnn) },
      },
      // 校正與一致性資訊：不改動畫面，僅隨紀錄存起來供日後檢視
      calInfo: (() => {
        const lc = localCalRef.current;
        if (!lc) return null;
        return {
          bpmFit:   fitLS(lc.bpm,   MIN_PAIRS_BPM),
          rmssdFit: fitLS(lc.rmssd, MIN_PAIRS_RMSSD),
          sdnnFit:  fitLS(lc.sdnn,  MIN_PAIRS_SDNN),
          agree:    lc.agree || null,
        };
      })(),
      count: ss.bpm.length,
      polar: ss.polarRaw.length > 0,
      ppgWaveform: recordWaveBuf.current.length > 0
        ? [...recordWaveBuf.current]   // 全程錄製
        : [...ppgHistoryRef.current],  // fallback
      sqi: sqi,
      motionFree: motionFree,
    };

    // 保存到歷史
    try {
      const stored = await AsyncStorage.getItem(HISTORY_KEY);
      const history = stored ? JSON.parse(stored) : [];
      history.push(reportData);
      await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch (e) {
      console.error('Failed to save to history:', e);
    }

    setReportData(reportData);
    // 相機原始值留給 Polar 頁的手動對照用
    setLastCam(reportData.camRaw);
    saveLastCam(reportData.camRaw);
    refreshCalCount();
    setScreen('report');
  };

  // ── Polar RR → RMSSD/SDNN
  const calcPolarHRV = (buf) => {
    const valid = buf.filter(r => r>=333 && r<=2000);
    if (valid.length < 4) return;
    const mean  = valid.reduce((a,b)=>a+b,0)/valid.length;
    const sdnn_ = Math.round(Math.sqrt(valid.reduce((s,r)=>s+(r-mean)**2,0)/valid.length));
    const diffs = valid.slice(1).map((r,i)=>r-valid[i]);
    const rmssd_= Math.round(Math.sqrt(diffs.reduce((s,d)=>s+d*d,0)/diffs.length));
    setPolarRmssd(rmssd_); setPolarSdnn(sdnn_);
    if (timerActiveRef.current) {
      sessionStats.current.polarRmssd.push(rmssd_);
      sessionStats.current.polarSdnn.push(sdnn_);
    }
  };

  // ── Polar BLE
  const connectPolar = async () => {
    try {
      const dev  = await navigator.bluetooth.requestDevice({ filters:[{ services:['heart_rate'] }] });
      const srv  = await dev.gatt.connect();
      const svc  = await srv.getPrimaryService('heart_rate');
      const char = await svc.getCharacteristic('heart_rate_measurement');
      await char.startNotifications();
      char.addEventListener('characteristicvaluechanged', (e) => {
        const v = e.target.value;
        const flags = v.getUint8(0);
        const hrFmt = flags & 0x1;
        const p = hrFmt ? v.getUint16(1,true) : v.getUint8(1);
        polarBpmRef.current = p; setPolarBpm(p); setPolarOk(true);
        if (flags & 0x10) {
          let off = hrFmt ? 3 : 2;
          if (flags & 0x08) off += 2;
          const rrs = [];
          while (off+1 < v.byteLength) { rrs.push(Math.round(v.getUint16(off,true)*1000/1024)); off+=2; }
          if (rrs.length) {
            polarRRBuf.current.push(...rrs);
            if (polarRRBuf.current.length > 60) polarRRBuf.current = polarRRBuf.current.slice(-60);
            calcPolarHRV(polarRRBuf.current);
          }
        }
      });
    } catch (e) {
      if (e.name !== 'NotFoundError') alert('藍牙連線失敗');
    }
  };

  // ── Swipe
  const panResponder = React.useRef(PanResponder.create({
    onMoveShouldSetPanResponder: (_,g) => Math.abs(g.dx)>10 && Math.abs(g.dx)>Math.abs(g.dy),
    onPanResponderRelease: (_,g) => {
      if (!polarOk) return;
      if (g.vx<-0.3||g.dx<-50) { setPanel(1); Animated.spring(swipeAnim,{toValue:-1,useNativeDriver:true,tension:80,friction:12}).start(); }
      if (g.vx>0.3||g.dx>50)   { setPanel(0); Animated.spring(swipeAnim,{toValue:0, useNativeDriver:true,tension:80,friction:12}).start(); }
    },
  })).current;

  // ── 後鏡頭容器
  const rearContainerRef = React.useCallback((node) => {
    if (!node) return;
    node.style.position = 'relative';

    const rearVid = document.createElement('video');
    rearVid.autoplay = true; rearVid.playsInline = true; rearVid.muted = true;
    rearVid.setAttribute('playsinline','');
    rearVid.setAttribute('webkit-playsinline','');
    rearVid.style.cssText = 'width:100%;height:100%;object-fit:cover;display:block;border-radius:50%;';

    const rearHcv = document.createElement('canvas');
    rearHcv.style.display = 'none';

    const ocv = document.createElement('canvas');
    ocv.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;border-radius:50%;';

    node.appendChild(rearVid);
    node.appendChild(rearHcv);
    node.appendChild(ocv);

    rearVideoRef.current     = rearVid;
    rearHiddenCvRef.current  = rearHcv;
    overlayCvRef.current     = ocv;

    // 並行初始化：stream 可能在 DOM 掛載前已取得，補接上去並啟動 rAF
    if (rearStreamRef.current) {
      rearVid.srcObject = rearStreamRef.current;
      rearVid.play().catch(() => {});
      if (!processActiveRef.current) {
        processActiveRef.current = true;
        rafRef.current = requestAnimationFrame(processFrame);
      }
    }
  }, []);

  // ── PPG 波形 Canvas 回調
  const ppgCanvasRef = React.useCallback((node) => {
    ppgWaveformCvRef.current = node;
    if (node) {
      node.style.width = '100%';
      node.style.height = '120px';
      drawPpgWaveform(ppgWaveformCvRef, ppgWaveform, totalSamplesRef.current);
    }
  }, [ppgWaveform, theme]);

  // ── Sub-components
  const Stat = ({ label, val, color }) => (
    <View style={{ flex:1, backgroundColor:theme.card, borderRadius:12, padding:14, margin:4, alignItems:'center' }}>
      <Text style={{ color:theme.textSub, fontSize:11, marginBottom:4 }}>{label}</Text>
      <Text style={{ color:color||theme.primary, fontSize:22, fontWeight:'700' }}>{val}</Text>
    </View>
  );

  const NormTable = ({ rows }) => (
    <View style={{ borderRadius:8, overflow:'hidden', borderWidth:1, borderColor:theme.card }}>
      {rows.map(([range,label,color]) => (
        <View key={range} style={{ flexDirection:'row', padding:8, borderBottomWidth:1, borderBottomColor:theme.card }}>
          <Text style={{ color, flex:1, fontSize:12 }}>{range}</Text>
          <Text style={{ color, fontSize:12 }}>{label}</Text>
        </View>
      ))}
    </View>
  );

  // ── 衛教卡片資料
  const EDU_CARDS = [
    {
      tag: '心率變異性',
      title: '什麼是 HRV？',
      body: 'HRV 是相鄰心跳間隔的變化程度。變化越大，代表自律神經越有彈性，身體的恢復與適應能力也越強。',
    },
    {
      tag: 'RMSSD',
      title: '副交感神經活性指標',
      body: 'RMSSD 為相鄰 RR 間期差值的均方根。數值越高代表副交感神經活性越強，通常反映良好的恢復狀態與壓力耐受力。健康範圍約 20–50 ms。',
    },
    {
      tag: 'SDNN',
      title: '整體自律神經調節力',
      body: 'SDNN 是所有 RR 間期的標準差，反映自律神經整體調節能力。數值越高代表心血管適應能力越佳，運動員通常高於一般人。健康範圍約 30–50 ms。',
    },
    {
      tag: '量測原理',
      title: '手指接觸式光電容積脈搏波',
      body: '後鏡頭讓食指覆蓋鏡頭以接觸式脈搏（PPG）同時計算心率與 HRV，單鏡頭設計簡單穩定。',
    },
  ];

  // ── Loading
  if (loadStep !== 'ready' && screen === 'dashboard') {
    const card = EDU_CARDS[eduIdx];
    return (
      <View style={{ flex:1, backgroundColor:theme.bg }}>
        <View style={{ flexDirection:'row', alignItems:'center',
          paddingHorizontal:16, paddingTop:16, paddingBottom:8 }}>
          <TouchableOpacity onPress={()=>{ cleanup(); onClose(); }} style={{ padding:5, minWidth:64 }}>
            <Text style={{ color:theme.primary, fontSize:14, fontWeight:'bold' }}>◀ 返回</Text>
          </TouchableOpacity>
        </View>
        <View style={{ flex:1, justifyContent:'center', alignItems:'center', paddingHorizontal:36 }}>
          <Text style={{ fontSize:64, marginBottom:20, textAlign:'center', color: theme.primary + 'aa' }}>♥</Text>
          <Text style={{ fontSize:10, letterSpacing:2, color:theme.textSub, textTransform:'uppercase', marginBottom:10 }}>
            {card.tag}
          </Text>
          <Text style={{ fontSize:18, fontWeight:'600', color:theme.textMain, textAlign:'center', marginBottom:14, lineHeight:26 }}>
            {card.title}
          </Text>
          <Text style={{ fontSize:13, color:theme.textSub, textAlign:'center', lineHeight:22 }}>
            {card.body}
          </Text>
          <View style={{ flexDirection:'row', marginTop:28 }}>
            {EDU_CARDS.map((_, i) => (
              <View key={i} style={{
                width: i === eduIdx ? 16 : 5, height: 5, borderRadius: 3, marginHorizontal: 3,
                backgroundColor: i === eduIdx ? theme.primary : theme.card,
              }} />
            ))}
          </View>
        </View>
        <View style={{ paddingHorizontal:32, paddingBottom:52 }}>
          <View style={{ flexDirection:'row', justifyContent:'space-between', marginBottom:8 }}>
            <Text style={{ fontSize:12, color: loadStep === 'error' ? '#ff4d4d' : theme.textSub }}>
              {loadMsg}
            </Text>
            {loadStep === 'loading' && <Text style={{ fontSize:12, color:theme.textSub }}>{loadPct}%</Text>}
          </View>
          <View style={{ height:3, backgroundColor:theme.card, borderRadius:2, overflow:'hidden' }}>
            <View style={{
              height: '100%', width: `${loadStep === 'error' ? 100 : loadPct}%`,
              backgroundColor: loadStep === 'error' ? '#ff4d4d' : theme.primary, borderRadius: 2,
            }} />
          </View>
          {loadStep === 'loading' && (
            <View style={{ marginTop:14, backgroundColor: theme.primary + '14',
              borderRadius:10, paddingVertical:10, paddingHorizontal:13,
              borderLeftWidth:3, borderLeftColor: theme.primary }}>
              <Text style={{ fontSize:12, color: theme.textMain, fontWeight:'600',
                lineHeight:18 }}>
                接下來會跳出相機權限，請選「允許」
              </Text>
              <Text style={{ fontSize:11, color: theme.textSub, marginTop:3, lineHeight:16 }}>
                量測是用後鏡頭讀手指的脈搏，畫面不會被錄下來，也不會上傳。
              </Text>
            </View>
          )}
          {loadStep === 'loading' && (
            <Text style={{ fontSize:10, color:theme.textSub, marginTop:10, textAlign:'center', opacity:0.5 }}>
              首次載入約需 20–30 秒
            </Text>
          )}
        </View>
      </View>
    );
  }

  // ── Report
  if (screen === 'report' && reportData) {
    const rState = reportData.rmssd<20 ? ['偏低','#ff4d4d'] : reportData.rmssd<=50 ? ['正常',theme.textMain] : ['良好',theme.primary];
    const sState = reportData.sdnn<30  ? ['偏低','#ff4d4d'] : reportData.sdnn<=50  ? ['正常',theme.textMain] : ['良好',theme.primary];
    return (
      <View style={{ flex:1, backgroundColor:theme.bg }}>
        <View style={{ flexDirection:'row', alignItems:'center', padding:14, borderBottomWidth:1, borderBottomColor:theme.card }}>
          <TouchableOpacity onPress={resetToMeasure} style={{ padding:5, minWidth:64 }}>
            <Text style={{ color:theme.primary, fontSize:14, fontWeight:'bold' }}>◀ 返回</Text>
          </TouchableOpacity>
          <Text style={{ flex:1, textAlign:'center', color:theme.textMain, fontSize:15, fontWeight:'700' }}>📊 測量報告</Text>
          <View style={{ minWidth:64 }} />
        </View>
        <ScrollView contentContainerStyle={{ padding:16 }}>

          {/* 未達可用時長的警語。數值照樣顯示 —— 但要讓人知道它不能拿來比較。 */}
          {reportData.valid === false && (
            <View style={{ backgroundColor:'#ff950022', borderRadius:12, padding:12, marginBottom:12,
              borderLeftWidth:3, borderLeftColor:'#ff9500' }}>
              <Text style={{ color:'#ff9500', fontSize:12, lineHeight:19 }}>
                這次只量了 {fmt(reportData.durationSec || 0)}，未滿 1 分鐘。
                RMSSD 在這麼短的窗口下不穩定，數值僅供參考，不建議跟其他次比較。
              </Text>
            </View>
          )}

          <View style={{ flexDirection:'row', marginBottom:10 }}>
            <Stat label="平均心率（手指）" val={`${reportData.bpm} BPM`} />
            <Stat label="有效樣本" val={reportData.count} color={theme.textMain} />
          </View>
          <View style={{ flexDirection:'row', marginBottom:20 }}>
            <Stat label={`RMSSD ✦${reportData.hrvSource==='Polar' ? 'Polar' : '手指'}`} val={reportData.rmssd||'--'} color="#ff9500" />
            <Stat label={`SDNN ✦${reportData.hrvSource==='Polar' ? 'Polar' : '手指'}`}  val={reportData.sdnn||'--'}   color="#0077ff" />
          </View>

          {/* 訊號品質指標 */}
          <View style={{ flexDirection:'row', gap:10, marginBottom:20 }}>
            <View style={{ flex:1, backgroundColor:theme.card, borderRadius:12, padding:12, alignItems:'center' }}>
              <Text style={{ color:theme.textSub, fontSize:10 }}>訊號品質 (SQI)</Text>
              <Text style={{ color: reportData.sqi >= 0.50 ? theme.primary : '#ff9500', fontSize:20, fontWeight:'700', marginTop:4 }}>
                {(reportData.sqi * 100).toFixed(0)}%
              </Text>
            </View>
            <View style={{ flex:1, backgroundColor:theme.card, borderRadius:12, padding:12, alignItems:'center' }}>
              <Text style={{ color:theme.textSub, fontSize:10 }}>量測時長</Text>
              <Text style={{ color: reportData.valid === false ? '#ff9500' : theme.primary,
                fontSize:20, fontWeight:'700', marginTop:4 }}>
                {fmt(reportData.durationSec || 0)}
              </Text>
            </View>
          </View>


          {reportData.polar && (
            <View style={{ backgroundColor:theme.primary+'22', borderRadius:12, padding:14, marginBottom:16, flexDirection:'row', alignItems:'center' }}>
              <Text style={{ color:theme.primary, fontSize:18, marginRight:10 }}>✅</Text>
              <Text style={{ color:theme.primary, fontSize:13, flex:1, lineHeight:18 }}>
                已上傳 Polar 校正資料，將幫助提升所有用戶的量測準確度
              </Text>
            </View>
          )}
          <View style={{ backgroundColor:theme.card, borderRadius:14, padding:16, marginBottom:14, borderLeftWidth:4, borderLeftColor:'#ff9500' }}>
            <View style={{ flexDirection:'row', justifyContent:'space-between', marginBottom:10 }}>
              <Text style={{ color:theme.textMain, fontWeight:'700' }}>RMSSD（副交感神經）</Text>
              <Text style={{ color:rState[1], fontWeight:'600' }}>{rState[0]}</Text>
            </View>
            <NormTable rows={[
              ['< 20 ms','偏低（疲勞）','#ff4d4d'],
              ['20–50 ms','正常範圍',theme.textMain],
              ['> 50 ms','良好（恢復佳）',theme.primary],
            ]} />
          </View>
          <View style={{ backgroundColor:theme.card, borderRadius:14, padding:16, marginBottom:32, borderLeftWidth:4, borderLeftColor:'#0077ff' }}>
            <View style={{ flexDirection:'row', justifyContent:'space-between', marginBottom:10 }}>
              <Text style={{ color:theme.textMain, fontWeight:'700' }}>SDNN（整體自律神經）</Text>
              <Text style={{ color:sState[1], fontWeight:'600' }}>{sState[0]}</Text>
            </View>
            <NormTable rows={[
              ['< 30 ms','偏低（調節力弱）','#ff4d4d'],
              ['30–50 ms','正常範圍',theme.textMain],
              ['> 50 ms','良好（適應力佳）',theme.primary],
            ]} />
          </View>


          <Text style={{ color:theme.textSub, fontSize:10, textAlign:'center', marginBottom:20 }}>
            ⚠️ 以上數值僅供參考，請勿做為醫療診斷依據
          </Text>
        </ScrollView>
      </View>
    );
  }

  // ── Dashboard
  return (
    <View style={{ flex:1, backgroundColor:theme.bg }} {...panResponder.panHandlers}>
      <View style={{ flexDirection:'row', alignItems:'center', padding:12, borderBottomWidth:1, borderBottomColor:theme.card }}>
        <TouchableOpacity onPress={()=>{ cleanup(); onClose(); }} style={{ padding:5, minWidth:64 }}>
          <Text style={{ color:theme.primary, fontSize:14, fontWeight:'bold' }}>◀ 返回</Text>
        </TouchableOpacity>
        <Text numberOfLines={1} style={{ flex:1, textAlign:'center', color:theme.textMain, fontSize:15, fontWeight:'700' }}>❤️ HRV 量測</Text>
        <View style={{ width:60, alignItems:'flex-end', flexDirection:'row', justifyContent:'flex-end', gap:6 }}>
          <TouchableOpacity onPress={() => { cleanup(); onNavigate?.('HRVHistory'); }}
            style={{ paddingHorizontal:8, paddingVertical:4, borderRadius:8, backgroundColor: theme.card }}>
            <Text style={{ color:theme.textSub, fontSize:11, fontWeight:'600' }}>📋</Text>
          </TouchableOpacity>
          {/* 進 Polar 頁而不是直接連線：iPhone 連不上胸帶，
              直接觸發連線只會跳一個失敗訊息，而使用者真正需要的是
              那一頁裡的手動輸入。
              走 App.js 的路由（跟旁邊的 📋 一樣）而不是在畫面內切換 ——
              內部切換的話 HRVScreen 不會卸載，相機和手電筒會一直開著。
              cleanup() 才是關掉手電筒的那一步。 */}
          <TouchableOpacity onPress={() => { cleanup(); onNavigate?.('POLAR'); }}
            style={{ paddingHorizontal:10, paddingVertical:4, borderRadius:8, backgroundColor: polarOk ? theme.primary+'33' : theme.card }}>
            <Text style={{ color: polarOk ? theme.primary : theme.textSub, fontSize:11, fontWeight:'600' }}>
              {polarOk ? `❤️${polarBpm}` : '🔗'}
            </Text>
          </TouchableOpacity>
        </View>
      </View>

      {polarOk && (
        <View style={{ flexDirection:'row', justifyContent:'center', paddingVertical:8, gap:8 }}>
          {[0,1].map(i => (
            <TouchableOpacity key={i} onPress={()=>{
              setPanel(i);
              Animated.spring(swipeAnim,{toValue:-i,useNativeDriver:true,tension:80,friction:12}).start();
            }}>
              <View style={{ width:8, height:8, borderRadius:4, backgroundColor: panel===i ? theme.primary : theme.card }} />
            </TouchableOpacity>
          ))}
          <Text style={{ color:theme.textSub, fontSize:11, marginLeft:4 }}>
            {panel===0 ? '← 滑動看 Polar' : '滑動回相機 →'}
          </Text>
        </View>
      )}

      <ScrollView contentContainerStyle={{ flexGrow:1 }}>
        {panel===0 ? (
          <>
            <View style={{ marginHorizontal:14, marginTop:14, marginBottom:10 }}>

              {/* ── 量測環 ──────────────────────────────────────
                  倒數改正數，SQI 併進環的顏色，切換鏡頭與手電筒
                  疊在環上而不是各佔一列 —— 原本那兩列在短螢幕會被擠掉。 */}
              {(() => {
                const R = 130;                          // 圓直徑的一半
                const target = timerSecs || 180;        // 不限時就拿 3 分當視覺參考
                const prog = timerMode === 'running'
                  ? Math.min(1, timerElapsed / target) : 0;
                const C = 2 * Math.PI * (R - 6);
                const valid = timerElapsed >= MIN_VALID_SEC;
                const ringColor = fingerStatus.err ? '#ff4d4d'
                  : valid ? '#2E9E6B' : theme.primary;

                return (
                  <View style={{ width:'100%', alignItems:'center', paddingVertical:14 }}>
                    <View style={{ width:R*2, height:R*2, position:'relative' }}>

                      {/* 進度環。不限時畫成虛線 —— 一眼看出沒有終點。 */}
                      <svg width={R*2} height={R*2}
                        style={{ position:'absolute', top:0, left:0 }}>
                        <circle cx={R} cy={R} r={R-6} fill="none"
                          stroke={theme.card} strokeWidth="6"
                          strokeDasharray={timerSecs === UNLIMITED ? '6 7' : undefined} />
                        {timerMode === 'running' && timerSecs > 0 && (
                          <circle cx={R} cy={R} r={R-6} fill="none"
                            stroke={ringColor} strokeWidth="6" strokeLinecap="round"
                            strokeDasharray={C} strokeDashoffset={C * (1 - prog)}
                            transform={`rotate(-90 ${R} ${R})`} />
                        )}
                        {/* 60 秒的可用門檻刻度 */}
                        {timerSecs > 0 && (
                          <line x1={R} y1="0" x2={R} y2="11"
                            stroke={theme.textSub} strokeWidth="2"
                            transform={`rotate(${(MIN_VALID_SEC/target)*360} ${R} ${R})`} />
                        )}
                      </svg>

                      {/* 相機圓 */}
                      <View style={{
                        position:'absolute', top:14, left:14,
                        width:(R-14)*2, height:(R-14)*2, borderRadius:R,
                        overflow:'hidden', backgroundColor:'#111' }}>
                        <View ref={rearContainerRef} style={{ position:'absolute', top:0, left:0, right:0, bottom:0 }} />

                        {timerMode === 'idle' && (
                          <View style={{ position:'absolute', top:0, left:0, right:0, bottom:0,
                            justifyContent:'center', alignItems:'center', paddingHorizontal:26 }}>
                            <Text style={{ fontSize:34, marginBottom:8 }}>👆</Text>
                            {/* 壓太用力會壓迫微血管，訊號反而更差 ——
                                這是 PPG 最常見的失敗原因，而多數 App 不會講。 */}
                            <Text style={{ color:'#fff', fontSize:12.5, opacity:0.85, textAlign:'center', lineHeight:19 }}>
                              手指輕蓋鏡頭{'\n'}不要用力壓
                            </Text>
                          </View>
                        )}

                        {timerMode !== 'idle' && (
                          <View style={{ position:'absolute', top:0, left:0, right:0, bottom:0,
                            justifyContent:'center', alignItems:'center' }}>
                            <Text style={{ color:'#fff', fontSize:40, fontWeight:'700',
                              fontVariant:['tabular-nums'], letterSpacing:-1 }}>
                              {fmt(timerElapsed)}
                            </Text>
                            <Text style={{ fontSize:10, marginTop:4,
                              color: valid ? '#8FE0BC' : 'rgba(255,255,255,0.6)' }}>
                              {valid ? '✓ 已達可用時長'
                                     : `還需 ${Math.max(0, MIN_VALID_SEC - timerElapsed)} 秒才可用`}
                            </Text>
                          </View>
                        )}

                        {!motionFree && (
                          <View style={{ position:'absolute', top:10, left:0, right:0, alignItems:'center' }}>
                            <View style={{ backgroundColor:'#ff4d4dcc', borderRadius:6, paddingHorizontal:10, paddingVertical:4 }}>
                              <Text style={{ color:'#fff', fontSize:11, fontWeight:'700' }}>⚠️ 請保持靜止</Text>
                            </View>
                          </View>
                        )}
                        {timerMode !== 'idle' && (
                          <View style={{ position:'absolute', bottom:10, left:0, right:0, alignItems:'center' }}>
                            <View style={{ backgroundColor:'#00000088', borderRadius:6, paddingHorizontal:8, paddingVertical:3 }}>
                              <Text style={{ color: fingerStatus.err ? '#ff4d4d' : '#8FE0BC', fontSize:10.5 }}>
                                {fingerStatus.msg}
                              </Text>
                            </View>
                          </View>
                        )}
                      </View>

                      {/* 環上左下：切換鏡頭。hitSlop 補到 44 —— 視覺小、手指好按。
                          按鈕上標示「第幾顆／共幾顆」：有使用者回報看不到這顆，
                          原因可能是瀏覽器只枚舉到一個邏輯裝置（cameras.length === 1），
                          跟版面沒關係。標出數字才查得到。 */}
                      {cameras.length > 1 && (
                        <TouchableOpacity onPress={cycleLens}
                          hitSlop={{ top:10, bottom:10, left:10, right:10 }}
                          style={{ position:'absolute', left:-2, bottom:18,
                            width:38, height:38, borderRadius:19, backgroundColor:theme.card,
                            borderWidth:1, borderColor:theme.primary+'55',
                            alignItems:'center', justifyContent:'center' }}>
                          <IconLensSwap size={19} color={theme.primary} />
                          <Text style={{ position:'absolute', bottom:1, right:3,
                            fontSize:7.5, color:theme.textSub }}>{lensIdx+1}/{cameras.length}</Text>
                        </TouchableOpacity>
                      )}

                      {/* 環上右下：手電筒。不支援就整顆不渲染 ——
                          iOS Safari 沒有 torch 約束，按不動的按鈕比沒有更糟。 */}
                      {torchAvail && (
                        <TouchableOpacity onPress={toggleTorch}
                          hitSlop={{ top:10, bottom:10, left:10, right:10 }}
                          style={{ position:'absolute', right:-2, bottom:18,
                            width:38, height:38, borderRadius:19,
                            backgroundColor: torchOn ? '#FFF3D6' : theme.card,
                            borderWidth:1, borderColor: torchOn ? '#EFC96A' : theme.primary+'55',
                            alignItems:'center', justifyContent:'center' }}>
                          <IconTorch size={19} on={torchOn}
                            color={torchOn ? '#B57F12' : theme.textSub} />
                        </TouchableOpacity>
                      )}
                    </View>

                    {/* 訊號品質：顏色比數字快。0% 配橘字要讀兩次才懂。 */}
                    <View style={{ flexDirection:'row', alignItems:'center', gap:6, marginTop:12 }}>
                      <View style={{ width:7, height:7, borderRadius:4,
                        backgroundColor: sqi >= 0.6 ? '#2E9E6B' : sqi >= 0.50 ? '#ff9500' : '#ff4d4d' }} />
                      <Text style={{ color:theme.textSub, fontSize:11 }}>
                        訊號{sqi >= 0.6 ? '良好' : sqi >= 0.50 ? '普通' : '不足'}　{(sqi*100).toFixed(0)}%
                      </Text>
                      {torchAvail && !torchOn && (
                        <Text style={{ color:'#ff9500', fontSize:10 }}>・建議開燈</Text>
                      )}
                    </View>

                    {/* 鏡頭自動挑選的進度／結果。
                        平常是空的；只有在 Android 上「手指貼著卻一直量不到」、
                        程式開始逐顆試鏡頭時才出現。
                        遠端測試的人可以直接截這一行，回報最後選到哪一顆。 */}
                    {!!lensNote && (
                      <Text style={{ color:theme.textSub, fontSize:10.5, lineHeight:16,
                        marginTop:6, marginHorizontal:12, textAlign:'center' }}>
                        {lensNote}
                      </Text>
                    )}

                    {/* 取樣率過低／分析錯誤的說明。
                        原本疊在圓圈裡，會被相機畫面和計時數字蓋掉 ——
                        那是整個畫面最不適合放長句子的位置。
                        移到訊號品質下方，橫向有整個寬度可用。 */}
                    {!!slowHint && (
                      <View style={{ marginTop:10, marginHorizontal:8,
                        backgroundColor:'#ff950018', borderRadius:10,
                        borderLeftWidth:3, borderLeftColor:'#ff9500',
                        paddingVertical:8, paddingHorizontal:11 }}>
                        <Text style={{ color:'#B57F12', fontSize:11, lineHeight:17 }}>
                          {slowHint}
                        </Text>
                      </View>
                    )}

                    {/* 相機打不開的原因。用紅色跟上面的黃色警示分開 ——
                        黃色是「量測品質不好」，紅色是「根本沒開始」。 */}
                    {!!camErr && (
                      <View style={{ marginTop:10, marginHorizontal:8,
                        backgroundColor:'#ff3b3018', borderRadius:10,
                        borderLeftWidth:3, borderLeftColor:'#ff3b30',
                        paddingVertical:9, paddingHorizontal:11 }}>
                        <Text style={{ color:'#C0392B', fontSize:11.5, lineHeight:18, fontWeight:'600' }}>
                          {camErr}
                        </Text>

                        {/* 手動再試一次。
                            這顆按鈕不只是「重試」，它是這個畫面裡唯一一個
                            **由使用者手勢觸發**的相機請求。
                            自動啟動那條路是在 useEffect 裡、等 Pyodide 載完才跑的，
                            那時候開啟 App 帶來的 user activation 早就過期了 ——
                            Chrome 對沒有使用者手勢的權限請求會直接不跳提示，
                            而且連續幾次之後會把這個網站設成自動封鎖。
                            從按鈕進來的請求有完整的手勢，提示才會出現。 */}
                        <TouchableOpacity
                          onPress={() => startCameras()}
                          sfxKey="confirm"
                          activeOpacity={0.85}
                          style={{ marginTop:10, alignSelf:'flex-start',
                            backgroundColor:'#C0392B', borderRadius:99,
                            paddingHorizontal:18, paddingVertical:9 }}>
                          <Text style={{ color:'#fff', fontSize:12.5, fontWeight:'800' }}>
                            允許使用相機
                          </Text>
                        </TouchableOpacity>

                        {/* 技術細節。使用者看不懂沒關係，它的用途是被截圖 ——
                            「沒反應」這三個字沒辦法除錯，這一行可以。 */}
                        {!!camDiag && (
                          <Text style={{ color:'#C0392B99', fontSize:9.5, marginTop:8 }}>
                            {camDiag}
                          </Text>
                        )}
                      </View>
                    )}
                  </View>
                );
              })()}

              {/* ── 時長：加減鍵 ＋ 點數字手打 ────────────────── */}
              {timerMode === 'idle' && (
                <View style={{ alignItems:'center', marginTop:4 }}>
                  <Text style={{ color:theme.textSub, fontSize:10, marginBottom:6 }}>量測時長</Text>
                  <View style={{ flexDirection:'row', alignItems:'center', gap:14 }}>
                    <TouchableOpacity onPress={() => stepDur(-DUR_STEP)}
                      disabled={timerSecs === UNLIMITED}
                      hitSlop={{ top:8, bottom:8, left:8, right:8 }}
                      style={{ width:38, height:38, borderRadius:19, backgroundColor:theme.card,
                        alignItems:'center', justifyContent:'center',
                        opacity: timerSecs === UNLIMITED ? 0.35 : 1 }}>
                      <Text style={{ color:theme.textMain, fontSize:22, lineHeight:26 }}>−</Text>
                    </TouchableOpacity>

                    <View style={{ minWidth:96, alignItems:'center' }}>
                      {durEditing ? (
                        <TextInput
                          value={durText}
                          // 非數字直接濾掉，不要等到 commit 才處理 ——
                          // 讓使用者打得出來又不生效，比不讓他打更困惑。
                          onChangeText={(t) => setDurText(String(t).replace(/\D/g, '').slice(0, 4))}
                          autoFocus
                          selectTextOnFocus
                          keyboardType="number-pad"
                          returnKeyType="done"
                          onBlur={commitDur}
                          onSubmitEditing={commitDur}
                          maxLength={4}
                          style={{ color:theme.primary, fontSize:28, fontWeight:'700',
                            textAlign:'center', minWidth:92, padding:0,
                            borderBottomWidth:2, borderBottomColor:theme.primary }}
                        />
                      ) : (
                        <TouchableOpacity
                          onPress={() => {
                            if (timerSecs === UNLIMITED) return;
                            // 帶入純數字（不含冒號）：180 秒 → '300'（3 分 00 秒）
                            const m = Math.floor(timerSecs / 60), sc = timerSecs % 60;
                            setDurText(`${m}${String(sc).padStart(2, '0')}`);
                            setDurEditing(true);
                          }}
                          hitSlop={{ top:10, bottom:10, left:14, right:14 }}>
                          <Text style={{ color:theme.textMain, fontSize:28, fontWeight:'700',
                            fontVariant:['tabular-nums'] }}>
                            {timerSecs === UNLIMITED ? '不限時' : fmt(timerSecs)}
                          </Text>
                        </TouchableOpacity>
                      )}
                      {/* 打字時即時把解讀結果顯示出來 ——
                          「100 會變成什麼」不該等按了確認才知道。 */}
                      <Text style={{ color:theme.textSub, fontSize:9, marginTop:2, opacity:0.75 }}>
                        {durEditing
                          ? (parseDur(durText) !== null
                              ? `＝ ${fmt(Math.max(DUR_MIN, Math.min(DUR_MAX, parseDur(durText))))}`
                              : '輸入數字，例如 100 ＝ 1:00')
                          : timerSecs === UNLIMITED ? '按 ＋ 回到計時' : '點數字輸入'}
                      </Text>
                    </View>

                    <TouchableOpacity onPress={() => stepDur(DUR_STEP)}
                      hitSlop={{ top:8, bottom:8, left:8, right:8 }}
                      style={{ width:38, height:38, borderRadius:19, backgroundColor:theme.card,
                        alignItems:'center', justifyContent:'center' }}>
                      <Text style={{ color:theme.textMain, fontSize:22, lineHeight:26 }}>＋</Text>
                    </TouchableOpacity>
                  </View>
                </View>
              )}

              {/* ── 主要動作鍵 ──────────────────────────────── */}
              <View style={{ marginHorizontal:16, marginTop:14 }}>
                {timerMode === 'idle' && (
                  <TouchableOpacity sfxKey="start" onPress={() => startTimer(timerSecs)}
                    style={{ backgroundColor:theme.primary, borderRadius:14,
                      paddingVertical:14, alignItems:'center' }}>
                    <Text style={{ color:'#fff', fontWeight:'700', fontSize:15 }}>開始量測</Text>
                  </TouchableOpacity>
                )}
                {timerMode === 'running' && (
                  <>
                    <TouchableOpacity onPress={finishTimer}
                      style={{ backgroundColor: timerElapsed >= MIN_VALID_SEC ? '#2E9E6B' : theme.card,
                        borderRadius:14, paddingVertical:14, alignItems:'center',
                        borderWidth:1, borderColor: timerElapsed >= MIN_VALID_SEC ? '#2E9E6B' : theme.primary+'55' }}>
                      <Text style={{ fontWeight:'700', fontSize:15,
                        color: timerElapsed >= MIN_VALID_SEC ? '#fff' : theme.primary }}>
                        結束並儲存
                      </Text>
                    </TouchableOpacity>
                    <Text style={{ color:theme.textSub, fontSize:10, textAlign:'center', marginTop:8, opacity:0.75 }}>
                      撐不住可以隨時結束{timerElapsed < MIN_VALID_SEC ? '，但未滿 1 分鐘的數值僅供參考' : ''}
                    </Text>
                  </>
                )}
                {timerMode === 'done' && (
                  <View style={{ gap:10 }}>
                    <TouchableOpacity sfxKey="select" onPress={genReport}
                      style={{ backgroundColor:theme.primary, borderRadius:14, paddingVertical:14, alignItems:'center' }}>
                      <Text style={{ color:'#fff', fontWeight:'700', fontSize:15 }}>查看報告</Text>
                    </TouchableOpacity>
                    <TouchableOpacity sfxKey="start" onPress={resetToMeasure}
                      style={{ backgroundColor:theme.card, borderRadius:14, paddingVertical:12, alignItems:'center' }}>
                      <Text style={{ color:theme.primary, fontWeight:'600', fontSize:14 }}>再測一次</Text>
                    </TouchableOpacity>
                  </View>
                )}
              </View>

              {/* 呼吸提示（量測中）*/}
              {timerMode === 'running' && (
                <Text style={{ color:theme.textSub, fontSize:11, textAlign:'center', marginTop:10, opacity:0.7 }}>
                  自然呼吸，避免深呼吸或憋氣
                </Text>
              )}

              {/* PPG 波形區塊 */}
              <View style={{ width:'100%', backgroundColor:theme.card, borderRadius:12, padding:12, marginTop:16 }}>
                <Text style={{ color:theme.textMain, fontWeight:'700', marginBottom:8, fontSize:13 }}>PPG 波形</Text>
                <canvas ref={ppgCanvasRef} style={{ width:'100%', height:'100px' }} />
              </View>

            </View>
          </>
        ) : (
          <View style={{ margin:16, backgroundColor:theme.card, borderRadius:16, padding:32, alignItems:'center' }}>
            <Text style={{ color:theme.textSub, fontSize:13, marginBottom:8 }}>Polar 即時心率</Text>
            <Text style={{ color:'#ff4d4d', fontSize:56, fontWeight:'700', lineHeight:64 }}>{polarBpm}</Text>
            <Text style={{ color:theme.textSub, marginBottom:24 }}>BPM</Text>
            {(polarRmssd>0||polarSdnn>0) && (
              <View style={{ flexDirection:'row', width:'100%', gap:10 }}>
                <View style={{ flex:1, backgroundColor:theme.bg, borderRadius:12, padding:12, alignItems:'center', borderLeftWidth:3, borderLeftColor:'#ff9500' }}>
                  <Text style={{ color:theme.textSub, fontSize:11 }}>RMSSD</Text>
                  <Text style={{ color:'#ff9500', fontSize:22, fontWeight:'700' }}>{polarRmssd}</Text>
                  <Text style={{ color:theme.textSub, fontSize:10 }}>ms ✦真實值</Text>
                </View>
                <View style={{ flex:1, backgroundColor:theme.bg, borderRadius:12, padding:12, alignItems:'center', borderLeftWidth:3, borderLeftColor:'#0077ff' }}>
                  <Text style={{ color:theme.textSub, fontSize:11 }}>SDNN</Text>
                  <Text style={{ color:'#0077ff', fontSize:22, fontWeight:'700' }}>{polarSdnn}</Text>
                  <Text style={{ color:theme.textSub, fontSize:10 }}>ms ✦真實值</Text>
                </View>
              </View>
            )}
            {bpm > 0 && (
              <View style={{ backgroundColor:theme.bg, borderRadius:12, padding:14, alignItems:'center', width:'100%', marginTop:12 }}>
                <Text style={{ color:theme.textSub, fontSize:12, marginBottom:6 }}>手指 BPM vs Polar BPM 差異</Text>
                <Text style={{ fontWeight:'700', fontSize:22, color: Math.abs(bpm-polarBpm)<5 ? theme.primary : '#ff9500' }}>
                  {bpm>polarBpm?'+':''}{bpm-polarBpm} BPM
                </Text>
                <Text style={{ color:theme.textSub, fontSize:11, marginTop:4 }}>
                  {Math.abs(bpm-polarBpm)<5 ? '準確度良好' : '差距偏大，請保持靜止'}
                </Text>
              </View>
            )}
            {calN > 0 && <Text style={{ color:theme.textSub, fontSize:11, marginTop:16 }}>已累積 {calN} 筆校正資料</Text>}
          </View>
        )}

        {/* ── 統計數據：三分類 ─────────────────────────────────── */}
        <View style={{ marginHorizontal:12, marginTop:12, marginBottom:8 }}>

          {/* 心跳 */}
          <Text style={{ color:theme.textSub, fontSize:10, letterSpacing:1,
            textTransform:'uppercase', marginBottom:4, paddingLeft:4 }}>心跳</Text>
          <View style={{ flexDirection:'row', marginBottom:10 }}>
            <Stat label="心率 BPM" val={bpm > 0 ? String(bpm) : '--'} />
          </View>

          {/* 時域分析 */}
          <Text style={{ color:theme.textSub, fontSize:10, letterSpacing:1,
            textTransform:'uppercase', marginBottom:4, paddingLeft:4 }}>時域分析</Text>
          <View style={{ flexDirection:'row', marginBottom:10 }}>
            <Stat label="RMSSD (ms)" val={rmssd > 0 ? String(rmssd) : '--'} color="#ff9500" />
            <Stat label="SDNN (ms)"  val={sdnn  > 0 ? String(sdnn)  : '--'} color="#0077ff" />
          </View>
        </View>

        {/* 量測完成提示 */}
        {timerMode === 'done' && (
          <Text style={{ color:theme.textSub, fontSize:12, textAlign:'center', marginBottom:8 }}>
            量測完成 — 點上方查看報告
          </Text>
        )}

        <Text style={{ color:theme.textSub, fontSize:10, textAlign:'center', marginBottom:16, opacity:0.6 }}>
          FPS: {realFps} · ⚠️ 數值僅供參考，非醫療診斷
        </Text>
      </ScrollView>
    </View>
  );
};

export default HRVScreen;