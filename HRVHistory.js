/**
 * HRVHistory.js
 * HRV 測量歷史紀錄頁面
 * - 日曆視圖：有測量的日期顯示主題色圓點
 * - 點擊圓點：彈出小框顯示當日狀態
 * - 詳細頁：色碼數字 + 常模參考 + 大刪除按鈕
 */
import React, { useRef } from 'react';
import {
  View, Text, ScrollView, Alert,
  Platform, Modal, Animated, PanResponder,
} from 'react-native';
import { TouchableOpacity } from './SoundTouchable';
import AsyncStorage from '@react-native-async-storage/async-storage';

const HISTORY_KEY = '@hrv_history';

// ── 左滑顯示刪除（ProfilePage SwipeRow 同架構）────────────────────────────
const SWIPE_DEL_W = 72;
const SwipeDeleteRow = ({ children, onDelete }) => {
  const pan = useRef(new Animated.Value(0)).current;
  const [rowW, setRowW] = React.useState(null);
  const panResponder = useRef(PanResponder.create({
    onMoveShouldSetPanResponder: (_, g) =>
      Math.abs(g.dx) > 12 && Math.abs(g.dy) < 44,
    onPanResponderMove: (_, g) => {
      if (g.dx < 0) pan.setValue(Math.max(-SWIPE_DEL_W, g.dx));
    },
    onPanResponderRelease: (_, g) => {
      if (g.dx < -30)
        Animated.spring(pan, { toValue: -SWIPE_DEL_W, useNativeDriver: true }).start();
      else
        Animated.spring(pan, { toValue: 0, useNativeDriver: true }).start();
    },
  })).current;
  const reset = () => Animated.spring(pan, { toValue: 0, useNativeDriver: true }).start();
  return (
    <View
      style={{ overflow: 'hidden', borderRadius: 8, opacity: rowW ? 1 : 0 }}
      onLayout={e => setRowW(e.nativeEvent.layout.width)}
    >
      <Animated.View
        {...panResponder.panHandlers}
        style={{ flexDirection: 'row', transform: [{ translateX: pan }] }}>
        {/* 固定寬度讓刪除按鈕真正溢出容器，web 的 overflow:hidden 才能裁掉 */}
        <View style={rowW ? { width: rowW } : { flex: 1 }}>{children}</View>
        <TouchableOpacity sfxKey="warn"
          onPress={() => { reset(); onDelete(); }}
          style={{ width: SWIPE_DEL_W, backgroundColor: '#E05555', alignItems: 'center', justifyContent: 'center', borderRadius: 8 }}>
          <Text style={{ color: '#fff', fontSize: 13, fontWeight: '700' }}>刪除</Text>
        </TouchableOpacity>
      </Animated.View>
    </View>
  );
};

// ── HRV 正常範圍常模 ────────────────────────────────────────────────────────
const NORMS = {
  bpm:   { low: 50, high: 100 },   // <50 偏低, 50-100 正常, >100 偏高
  rmssd: { low: 20, good: 50 },    // <20 偏低, 20-50 正常, >50 良好
  sdnn:  { low: 30, good: 100 },   // <30 偏低, 30-100 正常, >100 良好
};

const getBpmStatus = (bpm) => {
  if (bpm < NORMS.bpm.low)  return { text: '偏低', color: '#ff9500' };
  if (bpm > NORMS.bpm.high) return { text: '偏高', color: '#ff4d4d' };
  return { text: '正常', color: '#0aaf60' };
};

const getRmssdStatus = (v) => {
  if (v < NORMS.rmssd.low)  return { text: '偏低', color: '#ff4d4d' };
  if (v >= NORMS.rmssd.good) return { text: '良好', color: '#0aaf60' };
  return { text: '正常', color: '#ff9500' };
};

const getSdnnStatus = (v) => {
  if (v < NORMS.sdnn.low)  return { text: '偏低', color: '#ff4d4d' };
  if (v >= NORMS.sdnn.good) return { text: '良好', color: '#0aaf60' };
  return { text: '正常', color: '#ff9500' };
};

// 整體狀態（以 RMSSD 為主）
const getOverallStatus = (r) => getRmssdStatus(r.rmssd);

// ── 格式化日期 ────────────────────────────────────────────────────────────────
const formatDate = (ts) => {
  const d = new Date(ts);
  return d.toLocaleString('zh-TW', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
};

// ── 主元件 ────────────────────────────────────────────────────────────────────
const HRVHistory = ({ theme, onClose, onNavigate }) => {
  const [records, setRecords]           = React.useState([]);
  const [loading, setLoading]           = React.useState(true);
  const [selectedRecord, setSelectedRecord] = React.useState(null);
  const [showDetail, setShowDetail]     = React.useState(false);
  // 日曆
  const [calYear, setCalYear]   = React.useState(new Date().getFullYear());
  const [calMonth, setCalMonth] = React.useState(new Date().getMonth()); // 0-indexed
  // 小彈窗
  const [popupDay, setPopupDay]   = React.useState(null);   // { dayKey, records[] }
  const [popupPos, setPopupPos]   = React.useState({ x: 0, y: 0 });
  // 刪除確認 Modal
  const [deleteConfirmTs, setDeleteConfirmTs] = React.useState(null);

  React.useEffect(() => { loadHistory(); }, []);

  const loadHistory = async () => {
    try {
      setLoading(true);
      const stored = await AsyncStorage.getItem(HISTORY_KEY);
      if (stored) {
        const data = JSON.parse(stored);
        data.sort((a, b) => b.timestamp - a.timestamp);
        setRecords(data);
      }
    } catch (e) {
      Alert.alert('錯誤', '無法讀取歷史紀錄');
    } finally {
      setLoading(false);
    }
  };

  const saveHistory = async (data) => {
    try {
      await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(data));
    } catch (e) {
      Alert.alert('錯誤', '無法保存歷史紀錄');
    }
  };

  const deleteRecord = async (ts) => {
    const next = records.filter(r => r.timestamp !== ts);
    setRecords(next);
    await saveHistory(next);
    setDeleteConfirmTs(null);
    setShowDetail(false);
    // 同步更新彈窗（日曆彈窗刪除後移除該筆，若全刪則關閉彈窗）
    if (popupDay) {
      const remaining = popupDay.records.filter(r => r.timestamp !== ts);
      if (remaining.length === 0) setPopupDay(null);
      else setPopupDay({ ...popupDay, records: remaining });
    }
  };

  const exportAsCSV = () => {
    if (records.length === 0) { Alert.alert('提示', '沒有任何紀錄可以匯出'); return; }
    const headers = ['日期時間', '心率 (BPM)', 'RMSSD (ms)', 'SDNN (ms)', '來源', '樣本數'];
    const rows = records.map(r => [
      new Date(r.timestamp).toLocaleString('zh-TW'),
      r.bpm, r.rmssd, r.sdnn, r.hrvSource || '手指', r.count || '-',
    ]);
    const csv = [headers.join(','), ...rows.map(r => r.map(c => `"${c}"`).join(','))].join('\n');
    if (Platform.OS === 'web') {
      const blob = new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8;' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `HRV_History_${new Date().toISOString().split('T')[0]}.csv`;
      a.style.visibility = 'hidden';
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      Alert.alert('成功', '已下載 CSV 檔案');
    } else {
      Alert.alert('提示', '請複製以下資料:\n\n' + csv.substring(0, 200) + '...');
    }
  };

  // ── 日曆邏輯 ────────────────────────────────────────────────────────────────
  // 建立 { 'YYYY-MM-DD': [records] } 映射
  const dayMap = React.useMemo(() => {
    const map = {};
    records.forEach(r => {
      const d = new Date(r.timestamp);
      const key = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
      if (!map[key]) map[key] = [];
      map[key].push(r);
    });
    return map;
  }, [records]);

  const prevMonth = () => {
    if (calMonth === 0) { setCalMonth(11); setCalYear(y => y - 1); }
    else setCalMonth(m => m - 1);
  };
  const nextMonth = () => {
    if (calMonth === 11) { setCalMonth(0); setCalYear(y => y + 1); }
    else setCalMonth(m => m + 1);
  };

  // 產生當月格子
  const calCells = React.useMemo(() => {
    const firstDay = new Date(calYear, calMonth, 1).getDay(); // 0=Sun
    const daysInMonth = new Date(calYear, calMonth + 1, 0).getDate();
    const cells = [];
    for (let i = 0; i < firstDay; i++) cells.push(null);
    for (let d = 1; d <= daysInMonth; d++) cells.push(d);
    return cells;
  }, [calYear, calMonth]);

  const WEEK = ['日', '一', '二', '三', '四', '五', '六'];

  // ── 詳細頁面 ─────────────────────────────────────────────────────────────────
  if (showDetail && selectedRecord) {
    const r = selectedRecord;
    const bpmS   = getBpmStatus(r.bpm);
    const rmssdS = getRmssdStatus(r.rmssd);
    const sdnnS  = getSdnnStatus(r.sdnn);

    return (
      <View style={{ flex: 1, backgroundColor: theme.bg }}>
        {/* Header */}
        <View style={{ flexDirection: 'row', alignItems: 'center', padding: 14, borderBottomWidth: 1, borderBottomColor: theme.card }}>
          <TouchableOpacity onPress={() => setShowDetail(false)} style={{ padding: 5, minWidth: 64 }}>
            <Text style={{ color: theme.primary, fontSize: 14, fontWeight: 'bold' }}>◀ 返回</Text>
          </TouchableOpacity>
          <Text numberOfLines={1} style={{ flex: 1, textAlign: 'center', color: theme.textMain, fontSize: 15, fontWeight: '700' }}>📋 測量詳情</Text>
          <View style={{ width: 60 }} />
        </View>

        <ScrollView contentContainerStyle={{ padding: 16, paddingBottom: 32 }}>
          {/* 三個色碼數字卡 */}
          <View style={{ flexDirection: 'row', gap: 10, marginBottom: 16 }}>
            {/* BPM */}
            <View style={{ flex: 1, backgroundColor: theme.card, borderRadius: 12, padding: 12, alignItems: 'center' }}>
              <Text style={{ color: theme.textSub, fontSize: 10, marginBottom: 4 }}>心率</Text>
              <Text style={{ color: bpmS.color, fontSize: 26, fontWeight: '700' }}>{r.bpm}</Text>
              <Text style={{ color: theme.textSub, fontSize: 9 }}>BPM</Text>
              <View style={{ backgroundColor: bpmS.color + '22', borderRadius: 5, paddingHorizontal: 6, paddingVertical: 2, marginTop: 5 }}>
                <Text style={{ color: bpmS.color, fontSize: 9, fontWeight: '700' }}>{bpmS.text}</Text>
              </View>
            </View>
            {/* RMSSD */}
            <View style={{ flex: 1, backgroundColor: theme.card, borderRadius: 12, padding: 12, alignItems: 'center' }}>
              <Text style={{ color: theme.textSub, fontSize: 10, marginBottom: 4 }}>RMSSD</Text>
              <Text style={{ color: rmssdS.color, fontSize: 26, fontWeight: '700' }}>{r.rmssd}</Text>
              <Text style={{ color: theme.textSub, fontSize: 9 }}>ms</Text>
              <View style={{ backgroundColor: rmssdS.color + '22', borderRadius: 5, paddingHorizontal: 6, paddingVertical: 2, marginTop: 5 }}>
                <Text style={{ color: rmssdS.color, fontSize: 9, fontWeight: '700' }}>{rmssdS.text}</Text>
              </View>
            </View>
            {/* SDNN */}
            <View style={{ flex: 1, backgroundColor: theme.card, borderRadius: 12, padding: 12, alignItems: 'center' }}>
              <Text style={{ color: theme.textSub, fontSize: 10, marginBottom: 4 }}>SDNN</Text>
              <Text style={{ color: sdnnS.color, fontSize: 26, fontWeight: '700' }}>{r.sdnn}</Text>
              <Text style={{ color: theme.textSub, fontSize: 9 }}>ms</Text>
              <View style={{ backgroundColor: sdnnS.color + '22', borderRadius: 5, paddingHorizontal: 6, paddingVertical: 2, marginTop: 5 }}>
                <Text style={{ color: sdnnS.color, fontSize: 9, fontWeight: '700' }}>{sdnnS.text}</Text>
              </View>
            </View>
          </View>

          {/* 測量時間 */}
          <View style={{ backgroundColor: theme.card, borderRadius: 12, padding: 12, marginBottom: 12 }}>
            <Text style={{ color: theme.textSub, fontSize: 10 }}>測量時間</Text>
            <Text style={{ color: theme.textMain, fontSize: 14, fontWeight: '600', marginTop: 5 }}>
              {formatDate(r.timestamp)}
            </Text>
          </View>

          {/* 詳細資訊 */}
          <View style={{ backgroundColor: theme.card, borderRadius: 12, padding: 12, marginBottom: 12 }}>
            <Text style={{ color: theme.textMain, fontWeight: '700', fontSize: 13, marginBottom: 10 }}>詳細資訊</Text>
            {[
              ['HRV 來源',  r.hrvSource || '手指 PPG'],
              ['有效樣本',  String(r.count || '-')],
              ['Polar 資料', r.polar ? '✓ 有' : '✗ 無'],
            ].map(([k, v]) => (
              <View key={k} style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6 }}>
                <Text style={{ color: theme.textSub, fontSize: 12 }}>{k}</Text>
                <Text style={{ color: theme.textMain, fontSize: 12, fontWeight: '600' }}>{v}</Text>
              </View>
            ))}
          </View>

          {/* 常模參考 */}
          <View style={{ backgroundColor: theme.card, borderRadius: 12, padding: 12, marginBottom: 24 }}>
            <Text style={{ color: theme.textMain, fontWeight: '700', fontSize: 13, marginBottom: 10 }}>📊 常模參考</Text>
            {/* BPM */}
            <View style={{ marginBottom: 10 }}>
              <Text style={{ color: theme.textSub, fontSize: 11, marginBottom: 4 }}>心率 (BPM)</Text>
              <View style={{ flexDirection: 'row', gap: 6 }}>
                {[
                  { label: '偏低', range: '< 50',    color: '#ff9500' },
                  { label: '正常', range: '60–100',  color: '#0aaf60' },
                  { label: '偏高', range: '> 100',   color: '#ff4d4d' },
                ].map(item => (
                  <View key={item.label} style={{ flex: 1, backgroundColor: item.color + '18', borderRadius: 8, padding: 6, alignItems: 'center' }}>
                    <Text style={{ color: item.color, fontSize: 9, fontWeight: '700' }}>{item.label}</Text>
                    <Text style={{ color: item.color, fontSize: 9, marginTop: 2 }}>{item.range}</Text>
                  </View>
                ))}
              </View>
            </View>
            {/* RMSSD */}
            <View style={{ marginBottom: 10 }}>
              <Text style={{ color: theme.textSub, fontSize: 11, marginBottom: 4 }}>RMSSD (ms)</Text>
              <View style={{ flexDirection: 'row', gap: 6 }}>
                {[
                  { label: '偏低', range: '< 20',  color: '#ff4d4d' },
                  { label: '正常', range: '20–50', color: '#ff9500' },
                  { label: '良好', range: '≥ 50',  color: '#0aaf60' },
                ].map(item => (
                  <View key={item.label} style={{ flex: 1, backgroundColor: item.color + '18', borderRadius: 8, padding: 6, alignItems: 'center' }}>
                    <Text style={{ color: item.color, fontSize: 9, fontWeight: '700' }}>{item.label}</Text>
                    <Text style={{ color: item.color, fontSize: 9, marginTop: 2 }}>{item.range}</Text>
                  </View>
                ))}
              </View>
            </View>
            {/* SDNN */}
            <View>
              <Text style={{ color: theme.textSub, fontSize: 11, marginBottom: 4 }}>SDNN (ms)</Text>
              <View style={{ flexDirection: 'row', gap: 6 }}>
                {[
                  { label: '偏低', range: '< 30',    color: '#ff4d4d' },
                  { label: '正常', range: '30–100',  color: '#ff9500' },
                  { label: '良好', range: '≥ 100',   color: '#0aaf60' },
                ].map(item => (
                  <View key={item.label} style={{ flex: 1, backgroundColor: item.color + '18', borderRadius: 8, padding: 6, alignItems: 'center' }}>
                    <Text style={{ color: item.color, fontSize: 9, fontWeight: '700' }}>{item.label}</Text>
                    <Text style={{ color: item.color, fontSize: 9, marginTop: 2 }}>{item.range}</Text>
                  </View>
                ))}
              </View>
            </View>
          </View>

          {/* 大刪除按鈕 */}
          <TouchableOpacity sfxKey="warn"
            onPress={() => setDeleteConfirmTs(r.timestamp)}
            style={{
              backgroundColor: '#ff4d4d',
              borderRadius: 14,
              paddingVertical: 14,
              alignItems: 'center',
            }}>
            <Text style={{ color: '#fff', fontSize: 16, fontWeight: '700' }}>刪除這筆紀錄</Text>
          </TouchableOpacity>
        </ScrollView>

        {/* 刪除確認 Modal */}
        <Modal visible={!!deleteConfirmTs} transparent animationType="fade" onRequestClose={() => setDeleteConfirmTs(null)}>
          <View style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.45)', alignItems: 'center', justifyContent: 'center' }}>
            <View style={{ backgroundColor: '#fff', borderRadius: 18, padding: 26, width: 280, alignItems: 'center' }}>
              <Text style={{ fontSize: 17, fontWeight: '700', color: '#1a1a1a', marginBottom: 8 }}>確認刪除</Text>
              <Text style={{ fontSize: 14, color: '#555', textAlign: 'center', marginBottom: 22, lineHeight: 20 }}>
                此動作無法復原，{'\n'}確定要刪除這筆紀錄嗎？
              </Text>
              <View style={{ flexDirection: 'row', gap: 12, width: '100%' }}>
                <TouchableOpacity sfxKey="back"
                  onPress={() => setDeleteConfirmTs(null)}
                  style={{ flex: 1, paddingVertical: 12, borderRadius: 10, alignItems: 'center', borderWidth: 1.5, borderColor: '#ddd' }}>
                  <Text style={{ color: '#555', fontWeight: '600', fontSize: 15 }}>取消</Text>
                </TouchableOpacity>
                <TouchableOpacity sfxKey="warn"
                  onPress={() => deleteRecord(deleteConfirmTs)}
                  style={{ flex: 1, paddingVertical: 12, borderRadius: 10, alignItems: 'center', backgroundColor: '#E05555' }}>
                  <Text style={{ color: '#fff', fontWeight: '700', fontSize: 15 }}>刪除</Text>
                </TouchableOpacity>
              </View>
            </View>
          </View>
        </Modal>
      </View>
    );
  }

  // ── 主日曆頁 ─────────────────────────────────────────────────────────────────
  if (loading) {
    return (
      <View style={{ flex: 1, backgroundColor: theme.bg, justifyContent: 'center', alignItems: 'center' }}>
        <Text style={{ color: theme.textMain }}>載入中...</Text>
      </View>
    );
  }

  return (
    <View style={{ flex: 1, backgroundColor: theme.bg }}>
      {/* Header */}
      <View style={{ flexDirection: 'row', alignItems: 'center', padding: 14, borderBottomWidth: 1, borderBottomColor: theme.card }}>
        <TouchableOpacity onPress={onClose} style={{ padding: 5, minWidth: 64 }}>
          <Text style={{ color: theme.primary, fontSize: 14, fontWeight: 'bold' }}>◀ 返回</Text>
        </TouchableOpacity>
        <Text numberOfLines={1} style={{ flex: 1, textAlign: 'center', color: theme.textMain, fontSize: 15, fontWeight: '700' }}>📋 測量歷史</Text>
        <View style={{ width: 60, alignItems: 'flex-end' }}>
          <TouchableOpacity onPress={exportAsCSV}
            style={{ paddingHorizontal: 8, paddingVertical: 4, borderRadius: 8, backgroundColor: theme.card }}>
            <Text style={{ color: theme.textSub, fontSize: 12 }}>⬇️</Text>
          </TouchableOpacity>
        </View>
      </View>

      <ScrollView contentContainerStyle={{ padding: 16, paddingBottom: 32 }}>
        {/* 月份導航 */}
        <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          <TouchableOpacity onPress={prevMonth}
            style={{ paddingHorizontal: 16, paddingVertical: 8, borderRadius: 10, backgroundColor: theme.card }}>
            <Text style={{ color: theme.textMain, fontSize: 16 }}>‹</Text>
          </TouchableOpacity>
          <Text style={{ color: theme.textMain, fontSize: 16, fontWeight: '700' }}>
            {calYear} 年 {calMonth + 1} 月
          </Text>
          <TouchableOpacity onPress={nextMonth}
            style={{ paddingHorizontal: 16, paddingVertical: 8, borderRadius: 10, backgroundColor: theme.card }}>
            <Text style={{ color: theme.textMain, fontSize: 16 }}>›</Text>
          </TouchableOpacity>
        </View>

        {/* 星期標頭 */}
        <View style={{ flexDirection: 'row', marginBottom: 8 }}>
          {WEEK.map(w => (
            <View key={w} style={{ flex: 1, alignItems: 'center' }}>
              <Text style={{ color: theme.textSub, fontSize: 11, fontWeight: '600' }}>{w}</Text>
            </View>
          ))}
        </View>

        {/* 日曆格子 */}
        <View style={{ flexDirection: 'row', flexWrap: 'wrap' }}>
          {calCells.map((day, idx) => {
            if (!day) return <View key={`empty-${idx}`} style={{ width: '14.28%', aspectRatio: 1 }} />;
            const key = `${calYear}-${String(calMonth+1).padStart(2,'0')}-${String(day).padStart(2,'0')}`;
            const dayRecs = dayMap[key];
            const today = new Date();
            const isToday = today.getFullYear() === calYear && today.getMonth() === calMonth && today.getDate() === day;
            const dotColor = dayRecs ? getOverallStatus(dayRecs[0]).color : null;

            return (
              <TouchableOpacity
                key={key}
                onPress={() => {
                  if (dayRecs) {
                    if (popupDay?.key === key) {
                      setPopupDay(null);
                    } else {
                      setPopupDay({ key, records: dayRecs });
                    }
                  }
                }}
                activeOpacity={dayRecs ? 0.7 : 1}
                style={{ width: '14.28%', aspectRatio: 1, alignItems: 'center', justifyContent: 'center' }}>
                <View style={{
                  width: 32, height: 32,
                  borderRadius: 16,
                  alignItems: 'center', justifyContent: 'center',
                  backgroundColor: isToday ? theme.primary + '22' : 'transparent',
                }}>
                  <Text style={{
                    color: isToday ? theme.primary : theme.textMain,
                    fontSize: 13,
                    fontWeight: isToday ? '700' : '400',
                  }}>{day}</Text>
                </View>
                {dotColor && (
                  <View style={{
                    width: 6, height: 6, borderRadius: 3,
                    backgroundColor: dotColor,
                    marginTop: 1,
                  }} />
                )}
              </TouchableOpacity>
            );
          })}
        </View>

        {/* 小彈窗 */}
        {popupDay && (
          <View style={{
            backgroundColor: theme.card,
            borderRadius: 14,
            padding: 14,
            marginTop: 16,
            borderLeftWidth: 3,
            borderLeftColor: getOverallStatus(popupDay.records[0]).color,
          }}>
            {/* 彈窗標題 */}
            <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
              <Text style={{ color: theme.textMain, fontSize: 13, fontWeight: '700' }}>
                {popupDay.key.replace(/-/g, '/')}
              </Text>
              <TouchableOpacity onPress={() => setPopupDay(null)}>
                <Text style={{ color: theme.textSub, fontSize: 18 }}>×</Text>
              </TouchableOpacity>
            </View>

            {/* 每筆紀錄（同一天可能有多筆，左滑顯示刪除） */}
            {popupDay.records.map((rec, i) => {
              const bS = getBpmStatus(rec.bpm);
              const rS = getRmssdStatus(rec.rmssd);
              const sS = getSdnnStatus(rec.sdnn);
              return (
                <View key={rec.timestamp} style={{ marginBottom: i < popupDay.records.length - 1 ? 12 : 0 }}>
                  <SwipeDeleteRow onDelete={() => setDeleteConfirmTs(rec.timestamp)}>
                    <View style={{ paddingRight: 4 }}>
                      <Text style={{ color: theme.textSub, fontSize: 10, marginBottom: 6 }}>
                        {new Date(rec.timestamp).toLocaleTimeString('zh-TW', { hour: '2-digit', minute: '2-digit' })}
                      </Text>
                      <View style={{ flexDirection: 'row', gap: 8 }}>
                        {[
                          { label: '心率',  val: `${rec.bpm}`,   unit: 'BPM', s: bS },
                          { label: 'RMSSD', val: `${rec.rmssd}`, unit: 'ms',  s: rS },
                          { label: 'SDNN',  val: `${rec.sdnn}`,  unit: 'ms',  s: sS },
                        ].map(({ label, val, unit, s }) => (
                          <View key={label} style={{ flex: 1, alignItems: 'center' }}>
                            <Text style={{ color: theme.textSub, fontSize: 9 }}>{label}</Text>
                            <Text style={{ color: s.color, fontSize: 16, fontWeight: '700' }}>{val}</Text>
                            <Text style={{ color: theme.textSub, fontSize: 9 }}>{unit}</Text>
                            <View style={{ backgroundColor: s.color + '22', borderRadius: 4, paddingHorizontal: 5, paddingVertical: 1, marginTop: 3 }}>
                              <Text style={{ color: s.color, fontSize: 8, fontWeight: '700' }}>{s.text}</Text>
                            </View>
                          </View>
                        ))}
                      </View>
                      {/* 查看詳情按鈕 */}
                      <TouchableOpacity
                        onPress={() => { setSelectedRecord(rec); setPopupDay(null); setShowDetail(true); }}
                        style={{ marginTop: 10, backgroundColor: theme.primary + '22', borderRadius: 8, paddingVertical: 7, alignItems: 'center' }}>
                        <Text style={{ color: theme.primary, fontSize: 11, fontWeight: '600' }}>查看詳情 →</Text>
                      </TouchableOpacity>
                    </View>
                  </SwipeDeleteRow>
                </View>
              );
            })}
          </View>
        )}

        {/* 空狀態提示 */}
        {records.length === 0 && (
          <View style={{ alignItems: 'center', marginTop: 40 }}>
            <Text style={{ fontSize: 40, marginBottom: 12 }}>📊</Text>
            <Text style={{ color: theme.textMain, fontSize: 16, fontWeight: '700', marginBottom: 6 }}>沒有測量紀錄</Text>
            <Text style={{ color: theme.textSub, fontSize: 13, textAlign: 'center', lineHeight: 20 }}>
              開始測量後，測量結果將在此顯示。
            </Text>
            <TouchableOpacity sfxKey="start"
              onPress={() => onNavigate?.('HRV')}
              style={{ marginTop: 20, paddingHorizontal: 24, paddingVertical: 10, borderRadius: 12, backgroundColor: theme.primary }}>
              <Text style={{ color: '#fff', fontWeight: '600', fontSize: 14 }}>開始測量</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* 圖例 */}
        {records.length > 0 && (
          <View style={{ flexDirection: 'row', justifyContent: 'center', gap: 16, marginTop: 20 }}>
            {[
              { color: '#0aaf60', label: '良好' },
              { color: '#ff9500', label: '正常' },
              { color: '#ff4d4d', label: '偏低' },
            ].map(({ color, label }) => (
              <View key={label} style={{ flexDirection: 'row', alignItems: 'center', gap: 5 }}>
                <View style={{ width: 8, height: 8, borderRadius: 4, backgroundColor: color }} />
                <Text style={{ color: theme.textSub, fontSize: 11 }}>{label}</Text>
              </View>
            ))}
          </View>
        )}
      </ScrollView>

      {/* 刪除確認 Modal（日曆頁左滑刪除也需要） */}
      <Modal visible={!!deleteConfirmTs} transparent animationType="fade" onRequestClose={() => setDeleteConfirmTs(null)}>
        <View style={{ flex: 1, backgroundColor: 'rgba(0,0,0,0.45)', alignItems: 'center', justifyContent: 'center' }}>
          <View style={{ backgroundColor: '#fff', borderRadius: 18, padding: 26, width: 280, alignItems: 'center' }}>
            <Text style={{ fontSize: 17, fontWeight: '700', color: '#1a1a1a', marginBottom: 8 }}>確認刪除</Text>
            <Text style={{ fontSize: 14, color: '#555', textAlign: 'center', marginBottom: 22, lineHeight: 20 }}>
              此動作無法復原，{'\n'}確定要刪除這筆紀錄嗎？
            </Text>
            <View style={{ flexDirection: 'row', gap: 12, width: '100%' }}>
              <TouchableOpacity sfxKey="back"
                onPress={() => setDeleteConfirmTs(null)}
                style={{ flex: 1, paddingVertical: 12, borderRadius: 10, alignItems: 'center', borderWidth: 1.5, borderColor: '#ddd' }}>
                <Text style={{ color: '#555', fontWeight: '600', fontSize: 15 }}>取消</Text>
              </TouchableOpacity>
              <TouchableOpacity sfxKey="warn"
                onPress={() => deleteRecord(deleteConfirmTs)}
                style={{ flex: 1, paddingVertical: 12, borderRadius: 10, alignItems: 'center', backgroundColor: '#E05555' }}>
                <Text style={{ color: '#fff', fontWeight: '700', fontSize: 15 }}>刪除</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
};

export default HRVHistory;