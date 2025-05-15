# 標準庫
from sklearn.cluster import DBSCAN
from matplotlib.ticker import FuncFormatter
import os
from datetime import datetime
import warnings

# 第三方庫 - 數據處理
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import logging

# 第三方庫 - 科學計算
from scipy.signal import argrelextrema
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

# 第三方庫 - 機器學習
from sklearn.cluster import KMeans, DBSCAN

# 第三方庫 - 視覺化
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages

# FinLab相關
import finlab
from finlab import data
from finlab.dataframe import FinlabDataFrame

# 載入環境變數
load_dotenv()
# 使用環境變數
finlab.login(os.getenv('FINLAB_API_KEY'))

# 設置中文字體與全局圖表樣式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.facecolor'] = '#F5F5F5'
plt.rcParams['axes.facecolor'] = '#E6F0F5'
plt.rcParams['savefig.facecolor'] = '#F5F5F5'

# 設置數據儲存路徑
os.makedirs("E:\\pickle", exist_ok=True)
data.set_storage(data.FileStorage(path="E:\\pickle"))


class StockPlotter:
    def __init__(self, stock_selection_file, stock_topics_file):
        self.stock_selection_df = pd.read_excel(stock_selection_file)
        self.stock_topics_df = pd.read_excel(stock_topics_file)
        self.stock_counts = {}
        self.sorted_stocks = None
        self.result_df = None
        self.colors = None
        self.high_score_stocks = None
        warnings.filterwarnings("ignore")

        # 獲取大戶持股相關數據
        try:
            self.inv = data.get('inventory')
            self.close = data.get('price:收盤價')
            self.h1 = FinlabDataFrame(self.inv[self.inv.持股分級.astype(int) <= 4]
                                      .reset_index().groupby(['date', 'stock_id'], observed=True)
                                      .agg({'持有股數': 'sum'})
                                      .reset_index()
                                      .pivot(index='date', columns='stock_id', values='持有股數'))
            self.h2 = FinlabDataFrame(self.inv[(self.inv.持股分級.astype(int) >= 11) &
                                               (self.inv.持股分級.astype(int) <= 14)]
                                      .reset_index().groupby(['date', 'stock_id'], observed=True)
                                      .agg({'持有股數': 'sum'})
                                      .reset_index()
                                      .pivot(index='date', columns='stock_id', values='持有股數'))
            self.ratio = (self.h2 / (self.h1 + self.h2))
            self.first_diff = self.ratio.diff(6)
            self.second_diff = self.first_diff.diff(6)
        except Exception as e:
            print(f"初始化大戶持股數據時出錯：{str(e)}")
            self.ratio = self.first_diff = self.second_diff = None
            
            
            self.monthly_revenue = data.get('monthly_revenue:當月營收')
        except Exception as e:
            print(f"初始化月營收數據時出錯：{str(e)}")
            self.monthly_revenue = None

    def process_data(self):
        try:
            strategies = self.stock_selection_df.columns
            for strategy in strategies:
                stocks = self.stock_selection_df[strategy].dropna()
                for stock in stocks:
                    stock = int(float(stock))
                    if stock in self.stock_counts:
                        self.stock_counts[stock]['count'] += 1
                        self.stock_counts[stock]['strategies'].append(strategy)
                    else:
                        self.stock_counts[stock] = {
                            'count': 1, 'strategies': [strategy]}
            self.sorted_stocks = sorted(
                self.stock_counts.items(), key=lambda x: x[1]['count'], reverse=True)

            result_data = []
            for stock, data in self.sorted_stocks:
                stock_info = self.stock_topics_df.loc[self.stock_topics_df['stock_no'] == stock]
                stock_name = stock_info['stock_name'].values[0] if len(
                    stock_info) > 0 else "未知"
                stock_topic = stock_info['topic'].values[0] if len(
                    stock_info) > 0 else "無"
                result_data.append({
                    '股票代碼': stock,
                    '股票名稱': stock_name,
                    '主題': stock_topic,
                    '被選次數': data['count'],
                    '對應策略': data['strategies']
                })
            self.result_df = pd.DataFrame(result_data)

            stock_set = set()
            for index, row in self.stock_selection_df.iterrows():
                for stock_id in row:
                    if pd.notna(stock_id):
                        stock_set.add(str(int(stock_id)))
            self.high_score_stocks = sorted(stock_set)
        except Exception as e:
            print(f"處理數據時出錯：{str(e)}")
            self.result_df = pd.DataFrame()
            self.high_score_stocks = []

    def setup_colors(self):
        try:
            all_strategies = list(set(
                [strategy for strategies in self.result_df['對應策略'] for strategy in strategies]))
            color_map = plt.cm.get_cmap('Set1')
            self.colors = {strategy: color_map(
                i/len(all_strategies)) for i, strategy in enumerate(all_strategies)}
        except Exception as e:
            print(f"設置顏色時出錯：{str(e)}")
            self.colors = {}
            
            


    def plot_stock(self, ax, ax_volume, s, days=480):
        # 內部輔助函數：避免標籤重疊（僅用於支撐/壓力和其他標籤）
        def get_safe_label_y(date, y, fontsize, used_positions, buffer_base=0.05, date_buffer=5):
            buffer = buffer_base * (fontsize / 10)
            safe_y = y
            for used_date, used_y, used_buffer in used_positions:
                if abs((date - used_date).days) < date_buffer and abs(safe_y - used_y) < used_buffer:
                    safe_y += buffer if safe_y < 0.5 else -buffer
                    safe_y = max(0.02, min(0.98, safe_y))
            used_positions.append((date, safe_y, buffer))
            return safe_y

        # 內部輔助函數：設置日期軸
        def set_date_axis(ax, dates, days):
            interval = 2 if days <= 120 else 4 if days <= 240 else 8 if days <= 480 else 12
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(
                byweekday=mdates.MO, interval=interval))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.tick_params(axis='x', which='major', labelsize=18, colors='black')
            plt.setp(ax.xaxis.get_majorticklabels(),
                    rotation=45, ha='right', color='black')

        # 內部輔助函數：計算 Volume Profile
        def compute_vp(close, open_, high, low, volume, window, bins):
            if not all(isinstance(x, pd.Series) for x in [close, open_, high, low, volume]) or len(close) < window:
                return None, None, None, None
            close, open_, high, low, volume = [x.tail(window).copy() for x in [
                close, open_, high, low, volume]]
            if any(x.empty for x in [close, open_, high, low, volume]):
                return None, None, None, None
            prev_close = close.shift()
            tr = pd.concat([(high - low), (high - prev_close).abs(),
                        (low - prev_close).abs()], axis=1).max(axis=1)
            atr = tr.rolling(20).mean().iloc[-1] or (high.max() - low.min()) / 20
            price_bins = np.linspace(low.min(), high.max(), bins + 1)
            bin_centers = (price_bins[:-1] + price_bins[1:]) / 2
            up_vp, down_vp = np.zeros(len(bin_centers)), np.zeros(len(bin_centers))
            λ = 0.01
            time_weights = np.exp(-λ * np.arange(len(close))[::-1])
            for i in range(len(close)):
                price, vol = close.iloc[i], volume.iloc[i]
                idx = np.searchsorted(price_bins, price, side="right") - 1
                if 0 <= idx < len(bin_centers):
                    change_ratio = (close.iloc[i] - open_.iloc[i]) / open_.iloc[i]
                    body_size = abs(close.iloc[i] - open_.iloc[i])
                    body_ratio = body_size / (high.iloc[i] - low[i] + 1e-5)
                    weight = time_weights[i] * abs(change_ratio) * body_ratio
                    (up_vp if close.iloc[i] >= open_.iloc[i]
                    else down_vp)[idx] += vol * weight
            return bin_centers, up_vp, down_vp, price_bins

        try:
            # 初始化 Lekas
            used_label_positions = []

            # 載入數據
            open_data = data.get("price:開盤價")[s].tail(days)
            high_data = data.get("price:最高價")[s].tail(days)
            low_data = data.get("price:最低價")[s].tail(days)
            close_data = data.get("price:收盤價")[s].tail(days)
            volume_data = data.get("price:成交股數")[s].tail(days) / 1000
            if any(x.empty for x in [open_data, high_data, low_data, close_data]):
                ax.text(0.5, 0.5, f'股票 {s} 價格數據缺失', transform=ax.transAxes,
                        ha='center', fontsize=16, color='red')
                logging.error(f"股票代號 {s} 的價格數據缺失")
                return

            # 計算技術指標
            ma5, ma20, ma60 = close_data.rolling(5).mean(), close_data.rolling(
                20).mean(), close_data.rolling(60).mean()
            volume_ma5, volume_std = volume_data.rolling(
                5).mean(), volume_data.rolling(5).std()
            price_change = close_data.pct_change() * 100
            today_volume, today_change = volume_data.iloc[-1], price_change.iloc[-1]
            volume_deviation = ((volume_data.iloc[-1] - volume_ma5.iloc[-1]) / volume_std.iloc[-1]
                                if volume_std.iloc[-1] != 0 else 0)
            dates = pd.date_range(
                end=close_data.index[-1], periods=len(close_data))

            # 成交量狀態
            vol_info = ("成交量異常放大" if volume_deviation > 2 else "成交量高於平均" if volume_deviation > 1 else
                        "成交量異常萎縮" if volume_deviation < -2 else "成交量低於平均" if volume_deviation < -1 else "成交量正常")

            # K線圖
            ax.set_facecolor('#E6F0F5')
            up = close_data > open_data
            color = np.where(up, 'red', 'green')
            width, width2 = 0.9, 0.2
            ax.bar(dates[up], close_data[up] - open_data[up], width,
                bottom=open_data[up], color='red', edgecolor='black', linewidth=0.5)
            ax.bar(dates[~up], open_data[~up] - close_data[~up], width,
                bottom=close_data[~up], color='green', edgecolor='black', linewidth=0.5)
            ax.bar(dates, high_data - low_data, width2,
                bottom=low_data, color=color, zorder=3)
            ax.plot(dates, ma5, label='5MA', color='orange', linewidth=2)
            ax.plot(dates, ma20, label='20MA', color='blue', linewidth=2)
            ax.plot(dates, ma60, label='60MA', color='purple', linewidth=2)

            # Volume Profile
            bins = 50
            vp_cache = {}
            for window, label in [(60, 'short'), (240, 'long')]:
                bin_centers, up_vp, down_vp, price_bins = compute_vp(
                    close_data, open_data, high_data, low_data, volume_data, window, bins)
                vp_cache[label] = (bin_centers, up_vp, down_vp, price_bins)
            if vp_cache['short'][0] is None:
                logging.error(f"股票代號 {s} 的 Volume Profile 計算失敗")
            else:
                ax2 = ax.twiny()
                for label, color, alpha in [('long', '#999999', 0.2), ('short', None, 0.6)]:
                    bin_centers, up_vp, down_vp, price_bins = vp_cache[label]
                    net_vp = up_vp - down_vp
                    for i in range(len(net_vp)):
                        intensity = min(
                            abs(net_vp[i]) / max(abs(net_vp).max(), 1), 1)
                        bar_color = (0.8, 0.4, 0.2, 0.3 + intensity * 0.5) if net_vp[i] >= 0 else (
                            0.2, 0.6, 0.8, 0.3 + intensity * 0.5) if label == 'short' else color
                        ax2.barh(bin_centers[i], abs(net_vp[i]), height=price_bins[1] - price_bins[0],
                                align='center', color=bar_color, alpha=alpha, zorder=2 if label == 'short' else 1)
                ax2.set_xlim(0, max(abs(vp_cache['short'][2] - vp_cache['short'][1]).max(), abs(
                    vp_cache['long'][2] - vp_cache['long'][1]).max(), 1) * 1.2)
                ax2.set_xticks([])
                ax2.set_xlabel('Volume Profile (灰:近240日, 彩:近60日)',
                            fontsize=16, labelpad=8)

                # 套牢區
                current_price = close_data.tail(3).mean()
                threshold = np.percentile(vp_cache['short'][2], 75)
                trapped_zones = []
                for i in range(len(vp_cache['short'][2])):
                    if vp_cache['short'][2][i] > threshold and vp_cache['short'][0][i] > current_price:
                        trapped_zones.append(
                            {'lower': vp_cache['short'][3][i], 'upper': vp_cache['short'][3][i + 1]})
                if trapped_zones:
                    merged = [trapped_zones[0]]
                    for zone in trapped_zones[1:]:
                        if zone['lower'] == merged[-1]['upper']:
                            merged[-1]['upper'] = zone['upper']
                        else:
                            merged.append(zone)
                    for zone in merged:
                        ax.axhspan(zone['lower'], zone['upper'],
                                facecolor='#00BCD4', alpha=0.25, zorder=0)
                        y_mid = (zone['lower'] + zone['upper']) / 2
                        # 直接使用 y_mid，不檢查重疊
                        ax.text(-0.03, y_mid, f'套牢區: {zone["lower"]:.2f}~{zone["upper"]:.2f}', color='#007A8A', fontsize=12,
                                ha='right', va='center', transform=ax.get_yaxis_transform(),
                                bbox=dict(facecolor='white', alpha=0.9, edgecolor='#00BCD4'))

                # 支撐/壓力
                n = min(max(5, days // 24), 30)
                high_idx, low_idx = argrelextrema(high_data.values, np.greater, order=n)[
                    0], argrelextrema(low_data.values, np.less, order=n)[0]
                high_peaks, low_peaks = high_data.iloc[high_idx], low_data.iloc[low_idx]

                def score_peaks(peaks):
                    if peaks.empty:
                        return {}
                    peak_vals, peak_dates = peaks.values, peaks.index
                    score_dict = {}
                    λ = 0.01
                    for val in np.unique(np.round(peak_vals, 2)):
                        matches = peak_vals[np.round(peak_vals, 2) == val]
                        last_seen = peak_dates[np.round(peak_vals, 2) == val][-1]
                        days_since = (peak_dates[-1] - last_seen).days
                        score = len(matches) * np.exp(-λ * max(days_since, 0))
                        score_dict[val] = score
                    return score_dict

                support_dict, resistance_dict = score_peaks(
                    low_peaks), score_peaks(high_peaks)
                all_levels = [(lvl, '壓力', 'green', sc) for lvl, sc in resistance_dict.items(
                )] + [(lvl, '支撐', 'red', sc) for lvl, sc in support_dict.items()]
                for lvl, label, color, score in sorted(all_levels, key=lambda x: -x[3])[:5]:
                    ax.axhline(y=lvl, color=color, linestyle='--',
                            alpha=0.6, linewidth=1 + score * 2)
                    y_relative = (lvl - ax.get_ylim()
                                [0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
                    safe_y = get_safe_label_y(
                        dates[-1], y_relative, 10 + score * 2, used_label_positions)
                    ax.text(1.01, safe_y, f'{label}: {lvl:.2f} (★{score:.1f})', color=color, fontsize=10 + score * 2,
                            ha='left', va='center', transform=ax.transAxes,
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor=color))

            # 成交量圖
            ax_volume.set_facecolor('#E6F0F5')
            ax_volume.bar(dates[up], volume_data[up],
                        width, color='#FF6666', alpha=0.9)
            ax_volume.bar(dates[~up], volume_data[~up],
                        width, color='#66CC75', alpha=0.9)

            # 券商淨買超
            try:
                buy_vol = data.get('etl:broker_transactions:top15_buy').fillna(0)
                sell_vol = data.get('etl:broker_transactions:top15_sell').fillna(0)
                if s in buy_vol.columns and s in sell_vol.columns:
                    stock_net_vol = (buy_vol[s].tail(
                        days) - sell_vol[s].tail(days))
                    if len(stock_net_vol) == len(dates):
                        max_abs_val = max(stock_net_vol.abs().max() * 1.2, 1)
                        for i in range(len(dates) - 1):
                            if not pd.isna(stock_net_vol.iloc[i]):
                                intensity = max(
                                    min(abs(stock_net_vol.iloc[i]) / max_abs_val, 1), 0.2)
                                color = (1, 0.3, 0.3, intensity * 0.5) if stock_net_vol.iloc[i] > 0.01 else (
                                    0.3, 0.8, 0.3, intensity * 0.5) if stock_net_vol.iloc[i] < -0.01 else None
                                if color:
                                    ax_volume.axvspan(
                                        dates[i], dates[i + 1], color=color, zorder=-1)
                        recent_value, recent_ma5 = stock_net_vol.iloc[-1], stock_net_vol.iloc[-5:].mean(
                        )
                        recent_ma60 = stock_net_vol.iloc[-60:].mean() if len(
                            stock_net_vol) >= 60 else 0
                        status_text = f"券商淨{'買入' if recent_value > 0 else '賣出'}\n今日: {abs(recent_value):.2f}張\n5日均: {abs(recent_ma5):.2f}張\n60日均: {abs(recent_ma60):.2f}張"
                        if recent_ma5 * recent_ma60 > 0 and abs(recent_ma5) > 0.02 and abs(recent_ma60) > 0.02:
                            status_text += f"\n【{'上漲' if recent_ma5 > 0 else '下跌'}機率增】"
                        safe_y = get_safe_label_y(
                            dates[-1], 0.82, 12, used_label_positions)
                        ax_volume.text(1.03, safe_y, status_text, ha='left', va='top',
                                    color='#D32F2F' if recent_value > 0 else '#2E7D32',
                                    fontsize=12, fontweight='bold', transform=ax_volume.transAxes,
                                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            except Exception as e:
                logging.error(f"股票代號 {s} 券商淨買超視覺化錯誤: {str(e)}")

            # 融資/融券
            try:
                fin_use, short_use = data.get('margin_transactions:融資使用率')[s].tail(
                    days), data.get('margin_transactions:融券使用率')[s].tail(days)
                if len(fin_use) == len(dates) and len(short_use) == len(dates):
                    ax_margin = ax_volume.twinx()
                    ax_margin.plot(dates, fin_use, label='融資使用率',
                                color='orange', linewidth=2, alpha=0.7)
                    ax_margin.plot(dates, short_use, label='融券使用率',
                                color='blue', linewidth=2, alpha=0.7)
                    ax_margin.set_ylabel(
                        '融\n資\n融\n券\n使\n用\n率', fontsize=16, rotation=0, color='gray', labelpad=10)
                    ax_margin.tick_params(axis='y', labelsize=14, colors='gray')
                    ax_margin.set_ylim(
                        0, max(fin_use.max(), short_use.max(), 10) * 1.2)
                    fin_threshold = np.percentile(fin_use, 75)
                    high_fin_zones = []
                    start_idx = None
                    for i in range(len(fin_use)):
                        if fin_use.iloc[i] > fin_threshold:
                            if start_idx is None:
                                start_idx = i
                        elif start_idx is not None:
                            high_fin_zones.append((start_idx, i))
                            start_idx = None
                    if start_idx is not None:
                        high_fin_zones.append((start_idx, len(fin_use)))
                    # 儲存已使用的 Y 位置以避免高融資區文字重疊
                    used_margin_label_positions = []
                    for start, end in high_fin_zones:
                        ax_margin.axvspan(
                            dates[start], dates[end - 1], facecolor='#D1C4E9', alpha=0.5, zorder=0)
                        zone_fin_range = f"{fin_use.iloc[start:end].min():.1f}%~{fin_use.iloc[start:end].max():.1f}%"
                        initial_y = fin_threshold * 1.1 / ax_margin.get_ylim()[1]
                        safe_y = get_safe_label_y(
                            dates[start], initial_y, 10, used_margin_label_positions, buffer_base=0.1)
                        ax_margin.text(dates[start], safe_y * ax_margin.get_ylim()[1], f'高融資區: {zone_fin_range}',
                                    color='gray', fontsize=10, ha='left', va='bottom',
                                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
            except Exception as e:
                logging.error(f"股票代號 {s} 融資/融券視覺化錯誤: {str(e)}")

            # 設置軸屬性
            price_range = high_data.max() - low_data.min()
            price_mid = (high_data.max() + low_data.min()) / 2
            ax.set_ylim(price_mid - price_range * 0.6,
                        price_mid + price_range * 0.6)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
            ax.tick_params(axis='y', labelsize=18, colors='black')
            ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
            set_date_axis(ax, dates, days)
            ax_volume.set_ylim(0, volume_data.max() * 1.2)
            ax_volume.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f'{int(x)}\n張'))
            ax_volume.set_ylabel('成\n交\n量', fontsize=20, rotation=0,
                                fontweight='bold', color='black', labelpad=15)
            ax_volume.tick_params(axis='y', labelsize=18, colors='black')
            set_date_axis(ax_volume, dates, days)
            ax_volume.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
            ax.set_zorder(ax_volume.get_zorder() + 1)
            ax.patch.set_visible(False)

            # 圖例統一放置在圖表左下角
            lines, labels = [], []
            for ax_obj in [ax, ax_volume, locals().get('ax_margin', None)]:
                if ax_obj:
                    l, lab = ax_obj.get_legend_handles_labels()
                    lines.extend(l)
                    labels.extend(lab)
            ax.legend(lines, labels, fontsize=10, loc='lower left', bbox_to_anchor=(0.01, 0.01),
                    frameon=True, facecolor='white', edgecolor='black', framealpha=1.0)

            # 標題整合股票代號、公司名稱、成交量資訊
            stock_info = self.stock_topics_df[self.stock_topics_df['stock_no'] == int(
                s)]
            company_name = stock_info["stock_name"].values[0] if not stock_info.empty else s
            topic = stock_info["topic"].values[0] if not stock_info.empty else ""
            today = datetime.today().strftime('%Y-%m-%d')
            title = (f'{company_name} ({s}) - {topic} ({days} Days)\n'
                    f'今日成交: {today_volume:.0f}張 | 漲跌幅: {today_change:.2f}% | 量比標準差: {volume_deviation:.2f}σ | {vol_info}')
            plt.suptitle(title, fontsize=20, fontweight='bold',
                        color='black', y=0.98)

            # 調整圖表間距，增加底部空間以顯示 X 軸標籤
            plt.tight_layout()
            plt.subplots_adjust(top=0.90, bottom=0.15)

        except Exception as e:
            ax.text(0.5, 0.5, f'繪製股票 {s} 圖表失敗', transform=ax.transAxes,
                    ha='center', fontsize=16, color='red')
            logging.error(f"繪製股票代號 {s} 的圖表時出錯: {str(e)}")





       
            
            




    def get_concentration_score(self, stock_id, date):
        try:
            date = pd.to_datetime(date).normalize()
            self.inv['date'] = pd.to_datetime(self.inv['date']).dt.normalize()
            stock_id = str(stock_id)
            df = self.inv[(self.inv['stock_id'] == stock_id)
                          & (self.inv['date'] == date)]
            if df.empty:
                return None
            group_sum = df.groupby('持股分級')['持有股數'].sum()
            if 17 in group_sum.index:
                group_sum = group_sum.drop(17)
            retail_shares = group_sum[group_sum.index.astype(int) <= 4].sum()
            mid_shares = group_sum[(group_sum.index.astype(int) >= 11) & (
                group_sum.index.astype(int) <= 14)].sum()
            large_shares = group_sum[(group_sum.index.astype(int) >= 15) & (
                group_sum.index.astype(int) <= 16)].sum()
            total = group_sum.sum()
            if total == 0:
                return None
            max_ratio = 100
            mid_to_retail_ratio = min(
                mid_shares / retail_shares, max_ratio) if retail_shares > 0 else None
            large_to_retail_ratio = min(
                large_shares / retail_shares, max_ratio) if retail_shares > 0 else None
            return {
                'mid_to_retail_ratio': mid_to_retail_ratio,
                'large_to_retail_ratio': large_to_retail_ratio
            }
        except Exception as e:
            print(f"計算 {stock_id} 在 {date} 的集中度時出錯: {e}")
            return None

    def plot_investor_ratio_with_revenue(self, subplot_spec, stock_id):
        """
        繪製投資者比率和月營收圖表，第一個子圖替換為月營收柱狀圖，其餘保持不變
        
        Parameters:
        -----------
        subplot_spec : GridSpec
            用於放置此圖表的GridSpec物件
        stock_id : str
            股票代碼
        """
        # 確保stock_id為字符串
        stock_id = str(stock_id)

        # 檢查大戶持股數據是否可用
        if self.ratio is None or self.first_diff is None or self.second_diff is None:
            print(f"股票代號 {stock_id} 大戶持股數據不可用，跳過該股票。")
            return

        if stock_id not in self.ratio.columns or stock_id not in self.first_diff.columns or stock_id not in self.second_diff.columns:
            print(f"股票代號 {stock_id} 在大戶持股資料中不存在，跳過該股票。")
            return

        # 尋找共同日期
        ratio_dates = self.ratio[stock_id].dropna().index
        first_diff_dates = self.first_diff[stock_id].dropna().index
        second_diff_dates = self.second_diff[stock_id].dropna().index
        close_dates = self.close[stock_id].dropna().index

        common_dates = ratio_dates.intersection(first_diff_dates).intersection(
            second_diff_dates).intersection(close_dates)
        recent_dates = common_dates[-8:] if len(common_dates) > 8 else common_dates

        if len(recent_dates) == 0:
            print(f"股票代號 {stock_id} 無共同日期數據，跳過該股票。")
            return

        # 建立子圖
        gs_right = gridspec.GridSpecFromSubplotSpec(
            4, 1, subplot_spec=subplot_spec, height_ratios=[1, 1, 1, 1], hspace=0.3)

        # ===== 修改第一個子圖：月營收柱狀圖 =====
        ax1 = plt.subplot(gs_right[0, 0])
        ax1.set_facecolor('#E6F0F5')

        try:
            # 1. 獲取月營收數據 - 讀取全部後僅繪製最近24個月
            rev = data.get("monthly_revenue:當月營收")[stock_id]
            if rev is None or rev.empty:
                raise ValueError(f"無法獲取股票代碼 {stock_id} 的月營收數據")

            # 取最近24個月的數據
            rev = rev.tail(24)
            rev_ma = rev.rolling(window=2, min_periods=1).mean()

            # 2. 計算營收創新高 (9個月內的最大值)
            high_revenue = (rev_ma == rev_ma.rolling(9, min_periods=6).max())

            # 3. 獲取年增率數據
            yoy = data.get('monthly_revenue:去年同月增減(%)')[stock_id]
            # 只取與rev相同時間段的年增率數據
            yoy = yoy.reindex(rev.index)

            # 4. 決定柱狀圖顏色 (考慮創新高因素)
            def get_color(yoy_val, is_high):
                if is_high:
                    return '#FFD700'  # 金色，表示創新高
                if pd.isna(yoy_val):
                    return '#BBBBBB'  # 灰色，無法獲取年增率
                elif yoy_val > 0:
                    return '#FF0000' if yoy_val > 20 else ('#FF3333' if yoy_val > 10 else
                                                        '#FF6666' if yoy_val > 5 else '#FF9999')
                else:
                    return '#006400' if yoy_val < -20 else ('#008000' if yoy_val < -10 else
                                                            '#228B22' if yoy_val < -5 else '#90EE90')

            # 創建柱狀圖的顏色列表
            colors = [get_color(y, h) for y, h in zip(yoy.values, high_revenue.values)]

            # 繪製月營收柱狀圖
            bars = ax1.bar(rev.index, rev.values, color=colors, alpha=0.7,
                        width=20, zorder=1, edgecolor='black', linewidth=0.8)

            # 計算3個月移動平均線
            ma3 = rev.rolling(window=3, min_periods=1).mean()
            # 計算12個月移動平均線
            ma12 = rev.rolling(window=12, min_periods=1).mean()

            # 繪製移動平均線
            ax1.plot(ma3.index, ma3.values, color='#1E90FF',
                    label='3個月均線', linewidth=2, zorder=3)
          
            ax1.plot(ma12.index, ma12.values, color='#FF8C00',  # 改為深橙色，更易辨識
                     label='12個月均線', linewidth=2.5, zorder=3, linestyle='--')


            # ===== 新增部分：獲取並繪製月均價 =====
            try:
                # 獲取日K線數據
                daily_price = self.close[stock_id].copy()
                
                if not isinstance(daily_price.index, pd.DatetimeIndex):
                    # 如果索引不是日期格式，嘗試轉換
                    daily_price.index = pd.to_datetime(daily_price.index)
                    
                # 計算月平均價格
                monthly_price = daily_price.resample('M').mean()
                
                # 確保月均價和營收數據有相同的時間長度範圍
                start_date = rev.index.min()
                end_date = rev.index.max()
                
                # 篩選相應時間範圍的月均價
                filtered_monthly_price = monthly_price[(monthly_price.index >= start_date) & 
                                                    (monthly_price.index <= end_date)]
                
                # 檢查是否有數據
                if not filtered_monthly_price.empty:
                    # 創建雙Y軸
                    ax_price = ax1.twinx()
                    
                    # 繪製月均價曲線
                    ax_price.plot(filtered_monthly_price.index, filtered_monthly_price.values,
                            color='black',  # 深粉色
                            linewidth=2.5,
                            marker='s',  # 方形標記
                            markersize=6,
                            label='月均價',
                            zorder=4)
                    
                    # 設置Y軸格式
                    ax_price.set_ylabel('股\n價', color='black',
                                    fontsize=14, rotation=0, labelpad=15, fontweight='bold')
                    ax_price.tick_params(
                        axis='y', labelcolor='black', labelsize=12)
                    ax_price.grid(False)
                    
                    # 設置整數格式
                    ax_price.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
                    
                    # 調整Y軸範圍使曲線完全可見
                    y_min = filtered_monthly_price.min() * 0.95
                    y_max = filtered_monthly_price.max() * 1.05
                    ax_price.set_ylim(y_min, y_max)
                    
                    has_monthly_price = True
                    print(f"已成功添加 {stock_id} 的月均價數據，共 {len(filtered_monthly_price)} 個數據點")
                else:
                    has_monthly_price = False
                    print(f"警告: {stock_id} 月均價數據為空")
                    
            except Exception as e:
                has_monthly_price = False
                print(f"繪製 {stock_id} 月均價時出錯: {str(e)}")
                import traceback
                traceback.print_exc()

            # 計算最新營收和年增率 (考慮台股每月10日公布上月數據)
            latest_date = rev.index[-1]
            latest_rev = rev.iloc[-1]
            latest_yoy = yoy.iloc[-1] if not pd.isna(yoy.iloc[-1]) else 0

            # 格式化營收數字 (轉為百萬元單位)
            formatted_rev = format(int(latest_rev/1000), ',')

            # 設置標題 (顯示月份為數據月份，不是發布月份)
            month_num = latest_date.month
            ax1.set_title(f'{stock_id} 月營收走勢圖 ({month_num}月: {formatted_rev}百萬元, 年增率: {latest_yoy:.1f}%)',
                        fontsize=20)

            # 設置格式
            ax1.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
            ax1.set_ylabel('月\n營\n收', fontsize=16, rotation=0, labelpad=15)
            ax1.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f'{int(x/1000):,}'))

            # 使用每6個月放置一個刻度
            xtick_positions = rev.index[::6]
            ax1.set_xticks(xtick_positions)
            ax1.set_xticklabels([d.strftime('%Y/%m')
                                for d in xtick_positions], rotation=45, fontsize=10)

            # 合併兩個軸的圖例並置於最上層
            if has_monthly_price:
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax_price.get_legend_handles_labels()

                # 創建一個最上層的圖例
                legend = ax1.legend(lines1 + lines2, labels1 + labels2,
                                    fontsize=12,
                                    loc='upper left',  # 固定在左上角
                                    bbox_to_anchor=(0.01, 0.99),  # 微調位置
                                    frameon=True,
                                    facecolor='white',
                                    edgecolor='black',
                                    framealpha=1.0)

                # 設置高zorder使圖例顯示在最上層
                legend.set_zorder(999)

                # 增加邊框寬度使圖例更加突出
                legend.get_frame().set_linewidth(2.0)

                # 安全地設置圖例句柄的zorder (兼容不同版本的matplotlib)
                if hasattr(legend, 'legendHandles'):
                    for handle in legend.legendHandles:
                        handle.set_zorder(1000)
            else:
                legend = ax1.legend(fontsize=12,
                                    loc='upper left',
                                    bbox_to_anchor=(0.01, 0.99),
                                    frameon=True,
                                    facecolor='white',
                                    edgecolor='black',
                                    framealpha=1.0)

                legend.set_zorder(999)
                legend.get_frame().set_linewidth(2.0)
                
                # 同樣安全地設置圖例句柄的zorder
                if hasattr(legend, 'legendHandles'):
                    for handle in legend.legendHandles:
                        handle.set_zorder(1000)



            # 為最近6個月添加年增率標籤
            for i, bar in enumerate(bars[-6:]):
                idx = i - 6
                if not pd.isna(yoy.iloc[idx]):
                    height = bar.get_height()
                    text_color = '#FF4444' if yoy.iloc[idx] > 0 else '#00CC00'
                    ax1.text(bar.get_x() + bar.get_width()/2.,
                            height * 1.02,
                            f'{yoy.iloc[idx]:.1f}%',
                            ha='center',
                            va='bottom',
                            fontsize=11,
                            color=text_color,
                            fontweight='bold',
                            rotation=45,
                            zorder=5)

                    # 使用小黑點標記最新一期月營收
                    if i == len(bars) - 1:
                        ax1.plot(bar.get_x() + bar.get_width()/2., height,
                            'ko', markersize=5, zorder=6)

        except Exception as e:
            print(f"繪製股票代號 {stock_id} 的月營收圖表時出錯：{str(e)}")
            ax1.text(0.5, 0.5, f'無法繪製月營收圖表\n{str(e)}',
                    ha='center', va='center', fontsize=20)

        


  
        # ===== 第三個子圖：籌碼集中度 + 券商淨買賣量 =====
      
        ax3 = plt.subplot(gs_right[1, 0])
        ax3.set_facecolor('#E6F0F5')

        # 籌碼集中度計算
        ratios = [self.get_concentration_score(stock_id, date) for date in recent_dates]
        mid_ratios = [r['mid_to_retail_ratio'] if r is not None else None for r in ratios]
        large_ratios = [r['large_to_retail_ratio'] if r is not None else None for r in ratios]

        # 有效資料過濾
        valid_indices = [i for i, (mr, lr) in enumerate(zip(mid_ratios, large_ratios)) if mr is not None and lr is not None]
        valid_dates = [recent_dates[i] for i in valid_indices]
        valid_mid_ratios = [mid_ratios[i] for i in valid_indices]
        valid_large_ratios = [large_ratios[i] for i in valid_indices]

        # 繪製籌碼集中度折線（標準化）
        if valid_dates:
            all_ratios = valid_mid_ratios + valid_large_ratios
            ratio_min = min([x for x in all_ratios if x is not None])
            ratio_max = max([x for x in all_ratios if x is not None])

            if ratio_max > ratio_min:
                normalized_mid = [(x - ratio_min) / (ratio_max - ratio_min) for x in valid_mid_ratios]
                normalized_large = [(x - ratio_min) / (ratio_max - ratio_min) for x in valid_large_ratios]
            else:
                normalized_mid = valid_mid_ratios
                normalized_large = valid_large_ratios

            ax3.plot(valid_dates, normalized_mid, color='purple', marker='o', markersize=8, label='中實戶/散戶 (標準化)')
            ax3.plot(valid_dates, normalized_large, color='orange', marker='o', markersize=8, label='大戶/散戶 (標準化)')

        # 建立右側 Y 軸，繪製券商淨買賣超柱狀圖
        try:
            buy_vol = data.get('etl:broker_transactions:top15_buy').fillna(0)
            sell_vol = data.get('etl:broker_transactions:top15_sell').fillna(0)
            close = self.close

            net_volume = (buy_vol - sell_vol) * close
            net_volume_week = net_volume.rolling(5).sum()
            net_volume_filtered = net_volume_week[stock_id].loc[recent_dates]

            # twin Y-axis
            ax3_right = ax3.twinx()
            bar_colors = ['red' if val > 0 else 'green' for val in net_volume_filtered]

            ax3_right.bar(net_volume_filtered.index, net_volume_filtered.values,
                        color=bar_colors, alpha=0.4, width=3,
                        label='券商週買賣超(紅買綠賣)', zorder=0)

            ax3_right.set_ylabel('券\n商\n買\n賣\n超', fontsize=14, rotation=0, labelpad=20, color='gray')
            ax3_right.tick_params(axis='y', labelcolor='gray', labelsize=12)
            
            def format_units(x, _):
                if abs(x) >= 1e8:
                    return f'{x/1e8:.1f}億'
                elif abs(x) >= 1e6:
                    return f'{x/1e6:.1f}M'
                elif abs(x) >= 1e3:
                    return f'{x/1e3:.1f}K'
                else:
                    return str(int(x))


            ax3_right.yaxis.set_major_formatter(FuncFormatter(format_units))


        except Exception as e:
            print(f"券商買賣超資料處理失敗：{e}")

        # 標題與主 Y 軸格式
        ax3.set_title(f'{stock_id} 近期籌碼集中度（標準化比較）', fontsize=20)
        ax3.grid(True)
        ax3.set_ylabel('標\n準\n化\n比\n率', fontsize=16, labelpad=15, rotation=0)
        ax3.tick_params(axis='both', which='major', labelsize=14)
        ax3.set_xticks([])
        ax3.set_xticklabels([])
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2f}'))

        # 合併圖例（來自 ax3 和 ax3_right）
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_right.get_legend_handles_labels()
        legend = ax3.legend(lines1 + lines2, labels1 + labels2,
                            fontsize=12,
                            loc='center left',
                            bbox_to_anchor=(1.11, 0.9),  # 圖表右側中央
                            borderaxespad=0.,
                            frameon=True,
                            facecolor='white',
                            edgecolor='black',
                            framealpha=1.0)

        legend.set_zorder(999)
        legend.get_frame().set_linewidth(1.5)
        
        # ===== 第二個子圖：保持原樣（大戶持股比例差分） =====
        ax2 = plt.subplot(gs_right[2, 0])
        ax2.set_facecolor('#E6F0F5')
        self.first_diff[stock_id].loc[recent_dates].plot(
            ax=ax2, color='green', marker='o', markersize=8, label='一階差分')
        self.second_diff[stock_id].loc[recent_dates].plot(
            ax=ax2, color='red', marker='o', markersize=8, label='二階差分')
        ax2.set_title(f'{stock_id} 近期大戶持股比例一階與二階差分', fontsize=20)
        ax2.grid(True)
        ax2.set_ylabel('差\n分', fontsize=16, labelpad=15, rotation=0)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.set_xticks([])
        ax2.set_xticklabels([])
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.2%}'))
        ax2.legend(fontsize=12, loc='best', frameon=True,
                   facecolor='white', edgecolor='black', framealpha=1.0)


        # ===== 第四個子圖：保持原樣（價格與大戶持股比例對比） =====
        ax4 = plt.subplot(gs_right[3, 0])
        ax4.set_facecolor('#E6F0F5')
        ax4.plot(self.close[stock_id].loc[recent_dates],
                'b-o', label='收盤價', markersize=8)
        ax4.set_ylabel('收\n盤\n價', color='b', fontsize=16,
                    labelpad=15, rotation=0)
        ax4.tick_params(axis='y', labelcolor='b', labelsize=14)
        ax4.set_title(f'{stock_id} 近期收盤價與大戶持股比例對比', fontsize=20)
        ax4.grid(True)
        ax5 = ax4.twinx()
        ax5.plot(self.ratio[stock_id].loc[recent_dates],
                'r-o', label='大戶持股比例', markersize=8)
        ax5.set_ylabel('大\n戶\n持\n股\n比\n例', color='r',
                    fontsize=16, labelpad=15, rotation=0)
        ax5.tick_params(axis='y', labelcolor='r', labelsize=14)
        ax5.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
        latest_date = recent_dates[-1]
        ratios = self.get_concentration_score(stock_id, latest_date)
        large_to_retail_ratio = ratios['large_to_retail_ratio'] if ratios is not None else None
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax5.get_legend_handles_labels()
        if large_to_retail_ratio is not None:
            labels1 = [f'{lbl}（大戶/散戶: {large_to_retail_ratio:.2f}）' if lbl ==
                    '收盤價' else lbl for lbl in labels1]
        legend = ax4.legend(lines1 + lines2, labels1 + labels2,
                            loc='best', fontsize=12,
                            frameon=True, facecolor='white', edgecolor='black',
                            framealpha=1.0)
        legend.get_frame().set_linewidth(1.5)
        legend.get_frame().set_zorder(10)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax4.tick_params(axis='x', which='major', labelsize=14)



    def calculate_stock_correlations(self):
        stock_prices = {}
        for stock in self.high_score_stocks:
            try:
                close_data = data.get("price:收盤價")[stock].tail(240)
                stock_prices[stock] = close_data
            except KeyError:
                continue
        price_df = pd.DataFrame(stock_prices)
        correlations = price_df.corr()
        distances = 1 - correlations
        condensed_dist = squareform(distances)
        Z = linkage(condensed_dist, method='ward')
        clusters = fcluster(Z, t=3, criterion='maxclust')
        clustered_stocks = []
        for i in range(1, max(clusters) + 1):
            cluster_stocks = [stock for j, stock in enumerate(
                correlations.index) if clusters[j] == i]
            clustered_stocks.extend(cluster_stocks)
        return clustered_stocks


    def create_pdf(self):
        try:
            today = datetime.today()
            year = today.strftime('%Y')
            month = today.strftime('%m')
            output_dir = os.path.join('output', year, month)
            os.makedirs(output_dir, exist_ok=True)
            pdf_filename = os.path.join(
                output_dir, f'tomstrategy_kbar_investor_{today.strftime("%Y%m%d")}.pdf')
            sorted_stocks = self.calculate_stock_correlations()

            with PdfPages(pdf_filename) as pdf:
                # 策略選股統計圖
                total_stocks = len(self.result_df)
                stocks_per_plot = 10
                num_plots = (total_stocks + stocks_per_plot -
                            1) // stocks_per_plot
                num_plots = min(4, num_plots)
                for plot_group in range((num_plots + 1) // 2):
                    fig = plt.figure(figsize=(48, 24))
                    fig.patch.set_facecolor('#F5F5F5')
                    gs = gridspec.GridSpec(2, 1, hspace=0.3, wspace=0.2)
                    for i in range(2):
                        plot_index = plot_group * 2 + i
                        if plot_index < num_plots:
                            ax = fig.add_subplot(gs[i])
                            ax.set_facecolor('#E6F0F5')
                            start_idx = plot_index * stocks_per_plot
                            end_idx = min((plot_index + 1) *
                                        stocks_per_plot, total_stocks)
                            subset = self.result_df.iloc[start_idx:end_idx]
                            n_stocks = len(subset)
                            bottoms = np.zeros(n_stocks)
                            for strategy in self.colors.keys():
                                heights = [strats.count(strategy)
                                        for strats in subset['對應策略']]
                                bars = ax.bar(subset['股票名稱'] + '\n(' + subset['股票代碼'].astype(str) + ')\n' + subset['主題'],
                                            heights, bottom=bottoms, color=self.colors[strategy], edgecolor='black')
                                bottoms += heights
                                for bar, height, strat in zip(bars, heights, subset['對應策略']):
                                    if height > 0:
                                        ax.annotate(f'{strategy}',
                                                    xy=(bar.get_x() + bar.get_width() / 2,
                                                        bar.get_y() + bar.get_height() / 2),
                                                    xytext=(0, 0), textcoords="offset points",
                                                    ha='center', va='center', fontsize=14, color='black')
                            ax.set_title(
                                f'被多策略選到的標的(優先觀察) ({start_idx+1}-{end_idx})', fontsize=30)
                            ax.set_ylabel('被\n選\n次\n數', fontsize=20,
                                        rotation=0, labelpad=20)
                            ax.set_xticks(range(len(subset)))
                            ax.set_xticklabels(subset['股票名稱'] + '\n(' + subset['股票代碼'].astype(str) + ')\n' + subset['主題'],
                                            rotation=45, ha='right', fontsize=22)
                            max_count = subset['被選次數'].max()
                            ax.set_ylim(0, max_count * 1.2)
                            for j, count in enumerate(subset['被選次數']):
                                ax.text(j, count, str(count),
                                        ha='center', va='bottom', fontsize=16)
                            ax.tick_params(axis='y', labelsize=18)
                            ax.yaxis.set_major_locator(
                                MaxNLocator(integer=True))
                            ax.grid(True, linestyle='--',
                                    alpha=0.7, linewidth=0.5)
                    plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=2.0)
                    pdf.savefig(fig)
                    plt.close(fig)

                # 主題統計圖
                fig = plt.figure(figsize=(48, 24))
                fig.patch.set_facecolor('#F5F5F5')
                ax = fig.add_subplot(111)
                ax.set_facecolor('#E6F0F5')
                topic_counts = self.result_df['主題'].value_counts()
                bar_colors = plt.cm.viridis(
                    np.linspace(0, 1, len(topic_counts)))
                topic_counts.plot(kind='bar', color=bar_colors, ax=ax)
                ax.set_title('熱門族群觀察', fontsize=30)
                ax.set_ylabel('出\n現\n次\n數', fontsize=20,
                            rotation=0, labelpad=20)
                new_labels = [f"{topic}\n({', '.join(map(str, self.result_df[self.result_df['主題'] == topic]['股票代碼'].tolist()))})"
                            for topic in topic_counts.index]
                ax.set_xticklabels(new_labels, rotation=45,
                                ha='right', fontsize=22)
                for index, topic in enumerate(topic_counts.index):
                    ax.text(index, topic_counts[topic], f'{topic_counts[topic]}',
                            ha='center', va='bottom', fontsize=20)
                ax.tick_params(axis='both', labelsize=18)
                ax.grid(True, linestyle='--', alpha=0.7, linewidth=0.5)
                plt.tight_layout(pad=3.0, w_pad=2.0, h_pad=2.0)
                pdf.savefig(fig)
                plt.close(fig)

                # K線圖與大戶持股圖表
                for stock in tqdm(sorted_stocks, desc="Generating stock charts"):
                    fig = plt.figure(figsize=(48, 40))
                    fig.patch.set_facecolor('#F5F5F5')
                    # 主GridSpec：2行（240天和480天），2列（左邊K線+成交量，右邊大戶持股）
                    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[
                                        1, 1], wspace=0.2, hspace=0.3)

                    # 為240天和480天分別創建子GridSpec，每個包含K線和成交量
                    gs_240 = gridspec.GridSpecFromSubplotSpec(
                        2, 1, subplot_spec=gs[0, 0], height_ratios=[2, 1], hspace=0.05)
                    gs_480 = gridspec.GridSpecFromSubplotSpec(
                        2, 1, subplot_spec=gs[1, 0], height_ratios=[2, 1], hspace=0.05)

                    # 設置策略標題（單行顯示）
                    stock_row = self.result_df[self.result_df['股票代碼'] == int(
                        stock)]
                    if not stock_row.empty:
                        strategies = stock_row['對應策略'].iloc[0]
                        strategy_title = f"選股策略: {', '.join(strategies)}"
                    else:
                        strategy_title = "選股策略: 無"
                    fig.suptitle(strategy_title, fontsize=28,
                                fontweight='bold', y=0.98)

                    # 240天K線圖和成交量
                    ax_kbar_240 = plt.subplot(gs_240[0, 0])
                    ax_volume_240 = plt.subplot(
                        gs_240[1, 0], sharex=ax_kbar_240)
                    self.plot_stock(ax_kbar_240, ax_volume_240,
                                    stock, days=240)

                    # 480天K線圖和成交量
                    ax_kbar_480 = plt.subplot(gs_480[0, 0])
                    ax_volume_480 = plt.subplot(
                        gs_480[1, 0], sharex=ax_kbar_480)
                    self.plot_stock(ax_kbar_480, ax_volume_480,
                                    stock, days=480)

            
                    # 右側大戶持股圖表（第一個子圖為月營收）
                    self.plot_investor_ratio_with_revenue(gs[:, 1], stock)  


                    plt.tight_layout(pad=3.0, w_pad=2.0,
                                    h_pad=2.0, rect=[0, 0, 1, 0.95])
                    pdf.savefig(fig)
                    plt.close(fig)
        except Exception as e:
            print(f"生成PDF時出錯：{str(e)}")


# 主程序
if __name__ == "__main__":
    try:
        stock_plotter = StockPlotter(
            'select_stock.xlsx', 'tw_stock_topics.xlsx')
        print("正在處理數據...")
        stock_plotter.process_data()
        print("正在設置圖表顏色...")
        stock_plotter.setup_colors()
        print("正在生成PDF報告...")
        stock_plotter.create_pdf()
        print("報告生成完成！檔案已保存於 output/YYYY/MM 資料夾")
    except Exception as e:
        print(f"程式執行出錯：{str(e)}")




# python onlykbarinv.py

