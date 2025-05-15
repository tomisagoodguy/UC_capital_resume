# 標準庫
import logging
import math
import os
# print(os.getenv('PYTHONPATH'))  # 應該顯示你的工作目錄路徑

import warnings
from datetime import datetime, timedelta
from pathlib import Path
import time
from datetime import datetime
# 數據處理和科學計算
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from typing import Dict, List

# Matplotlib 相關
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import (
    AutoMinorLocator,
    FuncFormatter,
    MaxNLocator,
    MultipleLocator
)
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools
# Finlab 相關
import finlab
from finlab import data
from finlab.dataframe import FinlabDataFrame
from pandas.core.indexing import convert_from_missing_indexer_tuple

# 其他工具
from dotenv import load_dotenv
from tqdm import tqdm
import yaml

# 系統設置
os.system('cls')

# 環境變數設置
load_dotenv()
finlab.login(os.getenv('FINLAB_API_KEY'))

# 檔案存儲設置
os.makedirs("E:\\pickle", exist_ok=True)
data.set_storage(data.FileStorage(path="E:\\pickle"))




# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



# 配置文件類

class Config:
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        if not Path(self.config_path).exists():
            self.create_default_config()

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def create_default_config(self):
        default_config = {
            'data_path': {
                'stock_selection': 'select_stock.xlsx',
                'stock_topics': 'tw_stock_topics.xlsx'
            },
            'plot_settings': {
                'figure_size': (48, 32),
                'dpi': 100,
                'font_size': {
                    'title': 30,
                    'label': 16,
                    'tick': 14
                }
            },
            'analysis_settings': {
                'correlation_threshold': 0.7,
                'lookback_periods': 240
            }
        }

        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, allow_unicode=True)


class StockPlotter:

    def __init__(self, stock_selection_file, stock_topics_file):
        """
        初始化股票繪圖器
        參數:
        stock_selection_file (str): 選股Excel檔案路徑
        stock_topics_file (str): 股票主題Excel檔案路徑
        """
        # 讀取數據文件
        self.stock_selection_df = pd.read_excel(stock_selection_file)
        self.stock_topics_df = pd.read_excel(stock_topics_file)

        # 初始化變數
        self.stock_counts = {}
        self.sorted_stocks = None
        self.result_df = None
        self.colors = None
        self.high_score_stocks = None
        self.data = data
        

        # 設置全局字體樣式
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['legend.fontsize'] = 14

        # 忽略所有警告
        warnings.filterwarnings("ignore")

    def process_data(self):
        """處理股票數據"""
        try:
            logger.info("開始處理股票數據...")
            self.stock_counts = {}  # 確保初始化

            # 處理策略選股數據
            strategies = self.stock_selection_df.columns

            for strategy in tqdm(strategies, desc="處理策略"):
                stocks = self.stock_selection_df[strategy].dropna()
                for stock in stocks:
                    try:
                        stock = int(float(stock))
                        if stock in self.stock_counts:
                            self.stock_counts[stock]['count'] += 1
                            self.stock_counts[stock]['strategies'].append(
                                strategy)
                        else:
                            self.stock_counts[stock] = {
                                'count': 1,
                                'strategies': [strategy]
                            }
                    except ValueError as e:
                        logger.warning(f"處理股票 {stock} 時出現數值轉換錯誤: {str(e)}")
                        continue

            # 根據被選次數排序
            logger.info("開始排序股票...")
            self.sorted_stocks = sorted(
                self.stock_counts.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )

            # 創建結果DataFrame
            logger.info("創建結果DataFrame...")
            result_data = []
            for stock, data in tqdm(self.sorted_stocks, desc="處理股票資訊"):
                try:
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
                except Exception as e:
                    logger.warning(f"處理股票 {stock} 資訊時出現錯誤: {str(e)}")
                    continue

            # 創建完整的結果DataFrame，不限制筆數
            self.result_df = pd.DataFrame(result_data)

            # 處理K線圖所需的股票數據
            logger.info("處理K線圖數據...")
            stock_set = set()
            for index, row in self.stock_selection_df.iterrows():
                for stock_id in row:
                    try:
                        if pd.notna(stock_id):
                            stock_set.add(str(int(stock_id)))
                    except Exception as e:
                        logger.warning(f"處理股票ID {stock_id} 時出現錯誤: {str(e)}")
                        continue

            # 保存所有股票代碼
            self.high_score_stocks = sorted(stock_set)

            logger.info(f"數據處理完成，共處理 {len(self.high_score_stocks)} 支股票")
            return self.result_df

        except Exception as e:
            logger.error(f"數據處理過程中發生錯誤: {str(e)}")
            raise

    def create_series(self, drawing_data):
        """根據提供的數據創建序列"""
        series = (pd.DataFrame(drawing_data)
                  .sort_values('x').groupby('x').mean()['y']
                  .pipe(lambda s: (s - s.min()) / (s.max() - s.min()))
                  )
        return series.reset_index(drop=True)

    def calculate_correlation(self):
        """計算股票之間的相關性"""
        # 獲取股票收盤價資料
        close_prices = pd.DataFrame()
        for stock_id in self.high_score_stocks:
            try:
                close_data = data.get("price:收盤價")[stock_id].tail(240)
                close_prices[stock_id] = close_data
            except KeyError:
                print(f"股票代號 {stock_id} 不存在於資料中，跳過該股票。")

        # 計算相關性矩陣
        correlation_matrix = close_prices.corr()

        # 記錄所有相關係數對
        all_correlation_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                stock1 = correlation_matrix.columns[i]
                stock2 = correlation_matrix.columns[j]
                correlation_value = correlation_matrix.iloc[i, j]
                all_correlation_pairs.append(
                    (stock1, stock2, correlation_value))

        # 整合繪圖數據
        drawing_data = {"x": [34, 33, 37, 42, 49, 58, 65, 72, 78, 82, 84, 85, 86, 87, 88, 88, 90, 92, 94, 98, 103, 109, 114, 120, 126, 132, 137, 143, 147, 151, 154, 156, 158, 160, 162, 163, 165, 166, 168, 170, 171, 172, 173, 174, 176, 179, 185, 194, 205, 214, 222, 228, 233, 235, 237, 238, 238, 239, 240, 242, 243, 244, 245, 246, 248, 251, 254, 256, 259, 261, 263, 265, 267, 269, 273, 278, 284, 290, 298, 308, 319, 330, 340, 347, 351, 352, 353, 353, 353, 353],
                        "y": [171, 168, 159, 148, 137, 125, 115, 108, 100, 95, 93, 92, 92, 91, 91, 91, 90, 90, 90, 89, 89, 89, 89, 89, 89, 91, 94, 98, 103, 107, 111, 115, 118, 121, 123, 125, 127, 129, 130, 131, 132, 133, 134, 134, 134, 132, 126, 117, 105, 95, 86, 78, 73, 71, 70, 69, 68, 68, 68, 67, 67, 67, 67, 68, 71, 76, 83, 90, 96, 100, 102, 104, 106, 107, 109, 111, 112, 113, 114, 111, 104, 95, 85, 78, 75, 73, 72, 72, 72, 71]}

        # 檢查 x 和 y 的長度
        if len(drawing_data['x']) != len(drawing_data['y']):
            raise ValueError("x 和 y 的長度不一致！")

        # 創建目標序列
        target_series = self.create_series(drawing_data)

        # 計算與目標模式的相關性
        pattern_correlations = self.calc_corr_each_stock(
            target_series, close_prices)

        return all_correlation_pairs, pattern_correlations

    def calc_corr_each_stock(self, series_, close_prices):
        """計算每檔股票與給定序列的相關性"""
        corr = {}
        target_len = len(series_)

        for stock_name in close_prices.columns:
            if not close_prices[stock_name].isnull().all():
                # 取最後 target_len 個數據點
                stock_data = close_prices[stock_name].iloc[-target_len:]

                # 正規化股票數據
                normalized_stock = (stock_data - stock_data.min()) / (
                    stock_data.max() - stock_data.min())

                # 確保兩個序列長度相同
                if len(normalized_stock) == len(series_):
                    corr[stock_name] = np.corrcoef(
                        series_.values.flatten(),
                        normalized_stock.values.flatten()
                    )[0][1]
                else:
                    print(f"警告：股票 {stock_name} 的數據長度不符合要求，已跳過")

        return pd.Series(corr).sort_values(ascending=False).dropna()

    def setup_colors(self):
        """設置策略顏色"""

        all_strategies = list(
            set([strategy for strategies in self.result_df['對應策略']
                for strategy in strategies])
        )

        color_map = plt.cm.get_cmap('Set1')
        self.colors = {
            strategy: color_map(i/len(all_strategies))
            for i, strategy in enumerate(all_strategies)
        }

    def _find_body_peaks(self, open_data, close_data, window=20):
        """
        尋找K線實體的高點和低點
        
        參數:
        open_data: Series, 開盤價數據
        close_data: Series, 收盤價數據
        window: int, 分析窗口大小
        
        返回:
        tuple: (high_peaks, low_peaks) 高點和低點列表
        """
        try:
            # 1. 數據有效性檢查
            if open_data is None or close_data is None:
                print("錯誤: 輸入數據為空")
                return [], []
                
            # 2. 檢查數據類型
            if not isinstance(open_data, pd.Series) or not isinstance(close_data, pd.Series):
                print("錯誤: 輸入數據必須是 Pandas Series 類型")
                return [], []
                
            # 3. 檢查數據長度
            if len(open_data) < window or len(close_data) < window:
                print(f"錯誤: 數據長度不足 {window} 個交易日")
                return [], []
                
            # 4. 檢查是否包含無效值
            if open_data.isnull().any() or close_data.isnull().any():
                print("警告: 輸入數據包含空值")
                # 移除空值
                open_data = open_data.dropna()
                close_data = close_data.dropna()
                
            # 5. 檢查是否包含無限值
            if np.isinf(open_data).any() or np.isinf(close_data).any():
                print("警告: 輸入數據包含無限值")
                # 移除無限值
                open_data = open_data.replace([np.inf, -np.inf], np.nan).dropna()
                close_data = close_data.replace([np.inf, -np.inf], np.nan).dropna()

            # 6. 取最近的數據
            recent_open = open_data.tail(window)
            recent_close = close_data.tail(window)

            # 7. 計算實體的最高和最低點
            body_high = pd.DataFrame({
                'open': recent_open,
                'close': recent_close
            }).max(axis=1)

            body_low = pd.DataFrame({
                'open': recent_open,
                'close': recent_close
            }).min(axis=1)

            # 8. 尋找高點
            high_peaks = []
            window_size = 3  # 考慮前後各1天，共3天的窗口
            for i in range(window_size, len(body_high)-window_size):
                window_values = body_high.iloc[i-window_size:i+window_size+1]
                if body_high.iloc[i] == max(window_values):
                    peak_value = float(body_high.iloc[i])
                    if np.isfinite(peak_value):  # 確保值是有限的
                        high_peaks.append((i, peak_value))

            # 9. 尋找低點
            low_peaks = []
            for i in range(window_size, len(body_low)-window_size):
                window_values = body_low.iloc[i-window_size:i+window_size+1]
                if body_low.iloc[i] == min(window_values):
                    low_value = float(body_low.iloc[i])
                    if np.isfinite(low_value):  # 確保值是有限的
                        low_peaks.append((i, low_value))

            # 10. 限制高低點數量
            if len(high_peaks) > 3:
                # 按價格排序，保留最高的3個點
                high_peaks.sort(key=lambda x: x[1], reverse=True)
                high_peaks = high_peaks[:3]
                # 按時間順序重新排序
                high_peaks.sort(key=lambda x: x[0])

            if len(low_peaks) > 3:
                # 按價格排序，保留最低的3個點
                low_peaks.sort(key=lambda x: x[1])
                low_peaks = low_peaks[:3]
                # 按時間順序重新排序
                low_peaks.sort(key=lambda x: x[0])

            # 11. 輸出診斷信息
            print(f"資料點數: {len(body_high)}")
            print(f"找到的高點數量: {len(high_peaks)}")
            print(f"找到的低點數量: {len(low_peaks)}")
            
            # 12. 檢查最終結果的有效性
            if not high_peaks and not low_peaks:
                print("警告: 未找到有效的高低點")
                
            return high_peaks, low_peaks

        except Exception as e:
            print(f"_find_body_peaks 發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()  # 印出詳細的錯誤堆疊
            return [], []


    def _calculate_trend_lines(self, high_peaks, low_peaks):
        """
        根據高低點計算趨勢線，保持高可信度要求
        
        參數:
        high_peaks: list of tuples, 高點資料 [(index, value), ...]
        low_peaks: list of tuples, 低點資料 [(index, value), ...]
        
        返回:
        tuple: (high_line, low_line) 高點和低點趨勢線
        """
        try:
            # === 修改區塊 1：新增輸入檢查 ===
            if not high_peaks and not low_peaks:
                print("警告: 沒有高低點資料")
                return None, None
                
            def fit_line(peaks):
                # === 修改區塊 2：基礎檢查 ===
                if not peaks:
                    print("警告: 輸入的峰值列表為空")
                    return None
                    
                if len(peaks) < 2:
                    print(f"點數不足，需要至少2個點，當前只有{len(peaks)}個點")
                    return None

                # === 修改區塊 3：數值有效性檢查 ===
                try:
                    x = np.array([float(p[0]) for p in peaks])
                    y = np.array([float(p[1]) for p in peaks])
                except (ValueError, TypeError) as e:
                    print(f"座標轉換錯誤: {str(e)}")
                    return None

                # === 修改區塊 4：檢查是否有非有限值 ===
                if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                    print("警告: 座標包含非有限值")
                    return None

                # === 修改區塊 5：點位檢查 ===
                if np.any(x < 0) or np.any(y < 0):
                    print("警告: 座標包含負值")
                    return None

                # 如果點數大於2，嘗試找出最佳的點組合
                if len(peaks) > 2:
                    best_r_squared = 0
                    best_line = None

                    # === 修改區塊 6：迴圈內加入檢查 ===
                    for i in range(len(peaks)-1):
                        for j in range(i+1, len(peaks)):
                            temp_x = np.array([float(peaks[i][0]), float(peaks[j][0])])
                            temp_y = np.array([float(peaks[i][1]), float(peaks[j][1])])
                            
                            # 檢查計算用的數值
                            if not (np.all(np.isfinite(temp_x)) and np.all(np.isfinite(temp_y))):
                                continue

                            try:
                                slope, intercept = np.polyfit(temp_x, temp_y, 1)
                                
                                # === 修改區塊 7：檢查斜率和截距 ===
                                if not (np.isfinite(slope) and np.isfinite(intercept)):
                                    continue
                                    
                                y_pred = slope * temp_x + intercept
                                
                                # 計算 R 平方值
                                if np.all(temp_y == np.mean(temp_y)):  # 避免除以零
                                    r_squared = 0
                                else:
                                    r_squared = 1 - np.sum((temp_y - y_pred) ** 2) / \
                                            np.sum((temp_y - np.mean(temp_y)) ** 2)

                                if r_squared > best_r_squared:
                                    # === 修改區塊 8：生成趨勢線時的檢查 ===
                                    line_x = np.array([0, 19])
                                    line_y = slope * line_x + intercept
                                    
                                    if np.all(np.isfinite(line_y)):
                                        best_r_squared = r_squared
                                        best_line = (line_x, line_y)

                            except Exception as e:
                                print(f"擬合計算錯誤: {str(e)}")
                                continue

                    # === 修改區塊 9：R平方值檢查 ===
                    if best_r_squared >= 0.4:
                        print(f"R平方值: {best_r_squared:.4f}")
                        return best_line
                        
                else:
                    # === 修改區塊 10：兩點情況的處理 ===
                    try:
                        slope, intercept = np.polyfit(x, y, 1)
                        if not (np.isfinite(slope) and np.isfinite(intercept)):
                            print("警告: 計算出的斜率或截距非有限值")
                            return None
                            
                        y_pred = slope * x + intercept
                        r_squared = 1 - np.sum((y - y_pred) ** 2) / \
                                np.sum((y - np.mean(y)) ** 2)

                        print(f"R平方值: {r_squared:.4f}")

                        if r_squared >= 0.4:
                            line_x = np.array([0, 19])
                            line_y = slope * line_x + intercept
                            if np.all(np.isfinite(line_y)):
                                return (line_x, line_y)
                                
                    except Exception as e:
                        print(f"兩點擬合計算錯誤: {str(e)}")
                        return None

                print("R平方值太低或計算結果無效，趨勢線不夠可靠")
                return None

            # === 修改區塊 11：計算高低點趨勢線 ===
            high_line = fit_line(high_peaks)
            low_line = fit_line(low_peaks)

            return high_line, low_line
            
        except Exception as e:
            print(f"_calculate_trend_lines 發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

                   
    def plot_stock(self, ax, s):
            """
            繪製單個股票的K線圖

            參數:
            ax: matplotlib軸對象
            s: 股票代碼
            """
            try:
                fig = ax.figure  # 從 ax 獲取 fig
                current_date = pd.Timestamp.now()

                # 獲取法說會資訊並進行預處理
                investor_conference = data.get(
                    'investors_conference').reset_index()
                investor_conference = investor_conference.pivot(
                    index='date',
                    columns='stock_id',
                    values="公司名稱"
                ).notna()

                # 獲取庫藏股資訊並進行預處理
                treasury_stock = data.get('treasury_stock:預定買回期間-起')
                treasury_stock = pd.DataFrame({
                    "stock_id": [stock for stock in treasury_stock.columns for _ in treasury_stock[stock].dropna()],
                    "dates": [date for stock in treasury_stock.columns for date in treasury_stock[stock].dropna()]
                })
                if not treasury_stock.empty:
                    treasury_stock["value"] = 1
                    treasury_stock = treasury_stock.pivot(
                        index="dates",
                        columns="stock_id",
                        values="value"
                    ).notna()
                    treasury_stock = treasury_stock.reindex(
                        investor_conference.index,
                        columns=investor_conference.columns
                    ).fillna(False)

                # 獲取基本數據
                close = data.get("price:收盤價").loc[:, s]
                price_book_ratio = data.get("price_earning_ratio:股價淨值比").loc[:, s]
                net_value = close / price_book_ratio

                # 獲取現金增資資訊
                dividend_info = data.get('dividend_announcement')
                dividend_info = dividend_info[dividend_info['stock_id'] == s]
                dividend_info = dividend_info.drop_duplicates(['除權交易日'])
                capital_increment = dividend_info.set_index('除權交易日')['現金增資總股數(股)']

                # 獲取公司基本資訊
                company_info = data.get('company_basic_info')
                stock_amount = company_info.loc[company_info['stock_id']
                                                == s, '已發行普通股數或TDR原發行股數'].iloc[0]

                # 計算增資比例
                increment_ratio = capital_increment / stock_amount

                # 獲取近360天K線數據
                open_data = data.get("price:開盤價").loc[:, s].tail(360)
                high_data = data.get("price:最高價").loc[:, s].tail(360)
                low_data = data.get("price:最低價").loc[:, s].tail(360)
                close_data = data.get("price:收盤價").loc[:, s].tail(360)
                volume_data = data.get("price:成交股數").loc[:, s].tail(360) / 1000

                # 計算技術指標 (使用360天數據)
                sma5 = close_data.rolling(window=5).mean()
                sma20 = close_data.rolling(window=20).mean()
                sma60 = close_data.rolling(window=60).mean()

                # 只顯示最近120天的數據
                dates = close_data.index[-120:]
                open_data = open_data.tail(120)
                high_data = high_data.tail(120)
                low_data = low_data.tail(120)
                close_data = close_data.tail(120)
                volume_data = volume_data.tail(120)
                sma5 = sma5.tail(120)
                sma20 = sma20.tail(120)
                sma60 = sma60.tail(120)

                # 計算最近20天的實體高低點和趨勢線
                last_20_dates = dates[-20:]
                last_20_open = open_data.tail(20)
                last_20_close = close_data.tail(20)

                # 計算價格意圖因子
                days = 120
                # volume = data.get("price:成交股數").loc[:, s]
                close = data.get("price:收盤價").loc[:, s]

                # 計算主力意圖指標
                v = close.pct_change().abs().rolling(days).sum()  # 變動率
                s_return = close / close.shift(days) - 1  # 報酬率
                price_intention = s_return / v  # 價格意圖因子

                # 獲取最新值
                latest_intention = price_intention.iloc[-1]
                latest_return = s_return.iloc[-1]

                # 計算近期趨勢
                recent_days = 5
                recent_return = (close_data.iloc[-1] /
                                close_data.iloc[-recent_days] - 1) * 100

                # 根據不同情況設置警示文字和顏色
                if latest_return > 0:  # 上漲情況
                    if latest_intention > 0.15:
                        warning_text = "強烈疑似主力護航"
                        title_color = '#FF0000'  # 深紅色
                    elif latest_intention > 0.1:
                        warning_text = "疑似有主力護航"
                        title_color = '#FF4500'  # 橙紅色
                    elif latest_intention > 0.05:
                        warning_text = "可能有主力介入"
                        title_color = '#FFA500'  # 橙色
                    elif latest_intention > 0.03:
                        warning_text = "輕微主力跡象"
                        title_color = '#FFD700'  # 金色
                    else:
                        warning_text = "自然上漲走勢"
                        title_color = '#008000'  # 綠色
                elif latest_return < 0:  # 下跌情況
                    if latest_intention < -0.15:
                        warning_text = "強烈疑似主力打壓"
                        title_color = '#800000'  # 暗紅色
                    elif latest_intention < -0.1:
                        warning_text = "疑似有主力打壓"
                        title_color = '#A52A2A'  # 褐色
                    elif latest_intention < -0.05:
                        warning_text = "可能有主力放空"
                        title_color = '#B8860B'  # 暗金色
                    elif latest_intention < -0.03:
                        warning_text = "輕微放空跡象"
                        title_color = '#BDB76B'  # 暗卡其色
                    else:
                        warning_text = "自然下跌走勢"
                        title_color = '#CD5C5C'  # 印度紅
                else:  # 盤整情況
                    if abs(latest_intention) > 0.05:
                        warning_text = "盤整中有主力跡象"
                        title_color = '#4B0082'  # 靛青色
                    elif abs(latest_intention) > 0.03:
                        warning_text = "盤整中輕微波動"
                        title_color = '#483D8B'  # 暗灰藍色
                    else:
                        warning_text = "自然盤整走勢"
                        title_color = '#2F4F4F'  # 暗灰色

                # 添加強度顯示和近期趨勢
                if abs(latest_intention) > 0.03:
                    intention_str = f" (強度:{latest_intention:.2f})"
                else:
                    intention_str = ""

                trend_str = f" [近{recent_days}日漲跌:{recent_return:.1f}%]"

                # 找出高低點並計算趨勢線
                high_peaks, low_peaks = self._find_body_peaks(
                    last_20_open, last_20_close)
                high_line, low_line = self._calculate_trend_lines(
                    high_peaks, low_peaks)

                # 繪製趨勢線
                if high_line is not None:
                    line_x, line_y = high_line
                    try:
                        line_dates = [last_20_dates[0], last_20_dates[-1]]
                        ax.plot(line_dates, line_y, '--', color='#FF6B6B',
                                linewidth=1.5, label='上趨勢線', alpha=0.8)
                        for idx, value in high_peaks:
                            ax.plot(last_20_dates[idx], value,
                                    'o', color='#FF6B6B', markersize=4)
                    except Exception as e:
                        print(f"繪製上趨勢線時發生錯誤: {str(e)}")

                if low_line is not None:
                    line_x, line_y = low_line
                    try:
                        line_dates = [last_20_dates[0], last_20_dates[-1]]
                        ax.plot(line_dates, line_y, '--', color='#4ECDC4',
                                linewidth=1.5, label='下趨勢線', alpha=0.8)
                        for idx, value in low_peaks:
                            ax.plot(last_20_dates[idx], value,
                                    'o', color='#4ECDC4', markersize=4)
                    except Exception as e:
                        print(f"繪製下趨勢線時發生錯誤: {str(e)}")

                # 計算漲跌
                up = close_data > open_data
                down = close_data < open_data
                equal = close_data == open_data

                # 使用十六進制顏色代碼
                color = np.select([up, down, equal], ['#FF4444',
                                                    '#00CC00', '#FFD700'], default='#FFD700')

                # 設置K線圖的寬度
                width = 0.8
                width2 = 0.2

                # 繪製蠟燭實體
                ax.bar(dates[up], close_data[up] - open_data[up], width,
                    bottom=open_data[up], color='red', edgecolor='black', linewidth=0.5)
                ax.bar(dates[~up], open_data[~up] - close_data[~up], width,
                    bottom=close_data[~up], color='green', edgecolor='black', linewidth=0.5)

                # 繪製上下影線
                ax.bar(dates, high_data - low_data, width2,
                    bottom=low_data, color=color, zorder=3)

                # 繪製移動平均線
                ax.plot(dates, sma5, color='blue', linewidth=2.5, label='SMA5')
                ax.plot(dates, sma20, color='orange', linewidth=2.5, label='SMA20')
                ax.plot(dates, sma60, color='purple', linewidth=2.5, label='SMA60')

                # 繪製成交量
                ax2 = ax.twinx()
                ax2.bar(dates[up], volume_data[up],
                        width, color='#FFB6C1', alpha=0.9)
                ax2.bar(dates[down], volume_data[down],
                        width, color='#98FB98', alpha=0.9)
                ax2.bar(dates[equal], volume_data[equal],
                        width, color='#F0E68C', alpha=0.9)

                # 設置成交量Y軸
                max_volume = max(volume_data)
                ax2.set_ylim(0, max_volume * 2.5)
                ax2.tick_params(axis='y', labelsize=15, colors='black')
                ax2.yaxis.set_major_formatter(
                    FuncFormatter(lambda x, _: f'{int(x)}\n張'))

                # 設置價格Y軸範圍
                ax.set_ylim(min(low_data) * 0.9, max(high_data) * 1.1)

                # 準備圖例標籤
                legend_labels = ['SMA5', 'SMA20', 'SMA60']
                legend_handles = [
                    Line2D([0], [0], color='blue', linewidth=2.5),
                    Line2D([0], [0], color='orange', linewidth=2.5),
                    Line2D([0], [0], color='purple', linewidth=2.5)
                ]

                # 如果有趨勢線，添加到圖例中
                if 'high_line' in locals() and high_line is not None:
                    legend_handles.append(
                        Line2D([0], [0], color='#FF6B6B', linestyle='--', linewidth=1.5))
                    legend_labels.append('上趨勢線')

                if 'low_line' in locals() and low_line is not None:
                    legend_handles.append(
                        Line2D([0], [0], color='#4ECDC4', linestyle='--', linewidth=1.5))
                    legend_labels.append('下趨勢線')

                # 修改現金增資部分
                if not increment_ratio.empty:
                    date_range_increments = increment_ratio[
                        (increment_ratio.index >= dates[0]) &
                        (increment_ratio.index <= current_date)
                    ]
                    significant_increments = date_range_increments[date_range_increments > 0.1]

                    for date, ratio in significant_increments.items():
                        ax.axvline(x=date, color='red', linestyle='-.', alpha=0.5)
                        ax.text(date, ax.get_ylim()[1] * 1.17, f'現增\n{ratio:.1%}',
                                rotation=0, verticalalignment='bottom',
                                horizontalalignment='right', color='red', fontsize=16)

                    if len(significant_increments) > 0:
                        last_increment = significant_increments.index[-1].strftime(
                            '%Y-%m-%d')
                        last_ratio = significant_increments.iloc[-1]
                        legend_handles.append(
                            Line2D([0], [0], color='red', linestyle='-.', alpha=0.5))
                        legend_labels.append(
                            f'最近現增: {last_increment} ({last_ratio:.1%})')

                # 修改法說會部分
                if s in investor_conference.columns:
                    stock_conference = investor_conference.loc[dates[0]:dates[-1], s]
                    past_conferences = stock_conference[stock_conference].loc[:current_date].index

                    if len(past_conferences) > 0:
                        for conf_date in past_conferences:
                            ax.axvline(x=conf_date, color='purple',
                                    linestyle='--', alpha=0.5)
                        # 繼續法說會部分
                        for conf_date in past_conferences:
                            ax.axvline(x=conf_date, color='purple',
                                    linestyle='--', alpha=0.5)
                            ax.text(conf_date, ax.get_ylim()[1] * 1.17, '法\n說\n會',
                                    rotation=0, verticalalignment='bottom',
                                    horizontalalignment='right', color='purple', fontsize=16)

                        # 只在圖例中顯示最近一次法說會
                        last_conference = past_conferences[-1].strftime('%Y-%m-%d')
                        legend_handles.append(
                            Line2D([0], [0], color='purple', linestyle='--', alpha=0.5))
                        legend_labels.append(f'最近法說會: {last_conference}')

                # 修改庫藏股部分
                if s in treasury_stock.columns:
                    stock_treasury = treasury_stock.loc[dates[0]:dates[-1], s]
                    past_treasury = stock_treasury[stock_treasury].loc[:current_date].index

                    if len(past_treasury) > 0:
                        # 在K線圖上標記已執行的庫藏股買回
                        for treasury_date in past_treasury:
                            ax.axvline(x=treasury_date, color='green',
                                    linestyle='-.', alpha=0.5)
                            ax.text(treasury_date, ax.get_ylim()[1] * 1.17, '庫\n藏\n股',
                                    rotation=0, verticalalignment='bottom',
                                    horizontalalignment='right', color='green', fontsize=16)

                        # 只在圖例中顯示最近一次庫藏股買回
                        last_treasury = past_treasury[-1].strftime('%Y-%m-%d')
                        legend_handles.append(
                            Line2D([0], [0], color='green', linestyle='-.', alpha=0.5))
                        legend_labels.append(f'最近庫藏股: {last_treasury}')

                # 設置標題
                stock_info = self.stock_topics_df[self.stock_topics_df['stock_no'] == int(
                    s)]

                # 修改設置標題的部分，加入主力判斷結果
                if not stock_info.empty:
                    stock_name = stock_info['stock_name'].values[0]
                    # topic = stock_info['topic'].values[0]
                    title = f'{stock_name} ({s})\n{warning_text}{
                        intention_str}{trend_str}'
                    ax.set_title(title, fontsize=30, fontweight='bold',
                                color=title_color, pad=22)
                else:
                    title = f'股票代號 {s}\n{warning_text}{intention_str}{trend_str}'
                    ax.set_title(title, fontsize=30, fontweight='bold',
                                color=title_color, pad=22)

                # 優化圖例設計
                n_items = len(legend_labels)
                optimal_cols = min(4, max(3, (n_items + 2) // 3))  # 調整為更合適的列數

                # 設置圖例，移除標題，優化樣式
                legend = ax.legend(legend_handles, legend_labels,
                                loc='upper center',
                                bbox_to_anchor=(0.5, 1.8),  # 調整到標題上方
                                fontsize=16,                  # 稍微縮小字體
                                frameon=True,
                                facecolor='#F8F9FA',         # 使用淺灰色背景
                                edgecolor='#DEE2E6',         # 淺色邊框
                                shadow=False,                # 移除陰影效果
                                borderpad=0.5,
                                labelspacing=0.3,            # 減少行間距
                                handletextpad=0.4,           # 減少圖例符號和文字間距
                                columnspacing=1.5,           # 增加列間距
                                ncol=optimal_cols,
                                framealpha=0.95)

                # 設置x軸格式
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

                # 設置網格
                ax.grid(True, linestyle='--', alpha=0.3)

                # 返回圖表對象
                return ax

            except Exception as e:
                print(f"繪製股票 {s} 時發生錯誤: {str(e)}")
                return None


# ======================================rci==============================        
            
            

    def compute_rci(self, stock_id, interval=20):
        """
        計算RCI指標 (Rank Correlation Index)

        參數:
        stock_id: 股票代碼
        interval: RCI計算週期，預設14天
        """
        close = self.data.get("price:收盤價")[stock_id]
        rci_list = [None for _ in range(interval - 1)]

        nb_close = len(close)
        for idx in range(nb_close):
            if (idx + interval > nb_close):
                break

            y = close[idx:idx + interval]
            x_rank = np.arange(len(y))
            y_rank = rankdata(y, method='ordinal') - 1
            sum_diff = sum((x_rank - y_rank) ** 2)
            rci = (1 - ((6 * sum_diff) / (interval ** 3 - interval))) * 100
            rci_list.append(rci)

        return pd.Series(rci_list, index=close.index)


# ========================================================rci=================================================

    def plot_stock_with_rci(self, ax, stock_id, rci_threshold=-70, interval=20):
        try:
            # 調整子圖的位置和大小，為圖例留出空間
            # 參數分別是：左, 底, 寬, 高
            ax.set_position([0.1, 0.1, 0.75, 0.8])

            # 計算RCI
            rci_series = self.compute_rci(stock_id, interval)
            rci_series = rci_series.tail(120)
            
            # 獲取對應時間段的收盤價
            close_data = self.data.get("price:收盤價")[stock_id]
            close_data = close_data[rci_series.index[0]:rci_series.index[-1]]
            
            latest_rci = rci_series.iloc[-1]

            # 創建雙軸圖
            ax2 = ax.twinx()
            # 同樣調整右側Y軸的位置
            ax2.set_position([0.1, 0.1, 0.75, 0.8])

            # 整合判斷
            def get_rci_analysis(rci):
                if rci > 80:
                    return "嚴重超買", "#FF0000"
                elif rci > 50:
                    return "超買", "#FF4444"
                elif rci <= -70:
                    return "破底趨勢", "#8B0000"
                elif rci < -50:
                    return "超賣", "#228B22"
                else:
                    return "-", "#808080"

            status, color = get_rci_analysis(latest_rci)

            # 設定標題
            title = (
                f'{stock_id} RCI({interval}) - 低波動趨勢指標\n'
                f'RCI: {latest_rci:.1f} | 狀態: {status}\n'
                f'特性: 適合低波動標的,擅長判斷高原區下跌\n'
                f'注意: 暴跌可能無法及時反應'
            )
            ax.set_title(title, fontsize=30, color=color, pad=15)

            # 繪製RCI曲線
            rci_line = ax.plot(rci_series.index, rci_series,
                            color='blue', linewidth=2, label='RCI值')

            # 繪製收盤價
            price_line = ax2.plot(close_data.index, close_data,
                                color='black', linestyle='--', 
                                linewidth=2, label='收盤價',
                                alpha=0.8)

            # 繪製各條重要水平線
            lines = []
            lines.append(ax.axhline(y=80, color='red', linestyle=':', alpha=0.5, label='超買線'))
            lines.append(ax.axhline(y=50, color='pink', linestyle=':', alpha=0.5, label='輕度超買'))
            lines.append(ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3))
            lines.append(ax.axhline(y=-50, color='lightgreen', linestyle=':', alpha=0.5, label='輕度超賣'))
            lines.append(ax.axhline(y=rci_threshold, color='red', linestyle='--',
                                linewidth=2, alpha=0.7, label='破底趨勢線'))

            # 填充區域
            fill_1 = ax.fill_between(rci_series.index, rci_series, y2=0,
                                    where=(rci_series >= 0), color='red', alpha=0.1,
                                    label='多方區')
            fill_2 = ax.fill_between(rci_series.index, rci_series, y2=0,
                                    where=(rci_series < 0), color='green', alpha=0.1,
                                    label='空方區')
            fill_3 = ax.fill_between(rci_series.index, -100, y2=rci_threshold,
                                    where=(rci_series <= rci_threshold),
                                    color='red', alpha=0.3,
                                    label='破底警示區')

            # 添加破底警示文字
            if latest_rci <= -70:
                ax.text(rci_series.index[-1], rci_threshold - 5,
                        '破底趨勢警示', color='red',
                        fontsize=15, ha='right')

            # 設定Y軸範圍和格線
            ax.set_ylim(-100, 100)
            ax.grid(True, linestyle='--', alpha=0.7)

            # 設定右側Y軸的範圍
            price_margin = (close_data.max() - close_data.min()) * 0.1
            ax2.set_ylim(close_data.min() - price_margin, 
                        close_data.max() + price_margin)

            # 設定X軸刻度
            dates = pd.date_range(start=rci_series.index[0],
                                end=rci_series.index[-1],
                                freq='2W')
            ax.set_xticks(dates)
            ax.set_xticklabels(dates.strftime('%m-%d'), rotation=45)

            # 設定軸標籤
            ax.set_ylabel('RCI值', fontsize=16)
            ax2.set_ylabel('股\n價', fontsize=16,labelpad=20,rotation=0)

            # 合併所有圖例元素
            legend_elements = (rci_line + price_line + lines + 
                            [fill_1, fill_2, fill_3])
            labels = [l.get_label() for l in legend_elements]
            
            # 將圖例放在子圖外面右側
            ax.legend(legend_elements, labels,
                    loc='center left',
                    bbox_to_anchor=(1.15, 0.5),
                    fontsize=12)

        except Exception as e:
            print(f"繪製股票 {stock_id} 的RCI時發生錯誤: {str(e)}")


    def compute_candle_volatility(self, timeperiod=20):
        """計算波動率"""
        try:
            close = self.data.get("price:收盤價")
            high = self.data.get("price:最高價")
            low = self.data.get("price:最低價")
            open_ = self.data.get("price:開盤價")

            # 計算波動率
            bullish_candle = close >= open_
            bullish_volatility = abs(
                close.shift() - open_) + abs(open_ - low) + abs(low - high) + abs(high - close)
            bearish_volatility = abs(
                close.shift() - open_) + abs(open_ - high) + abs(high-low) + abs(low - close)
            candle_volatility = pd.DataFrame(
                np.nan, index=close.index, columns=close.columns)
            candle_volatility[bullish_candle] = bullish_volatility
            candle_volatility[~bullish_candle] = bearish_volatility

            # 計算相對波動率（以百分比表示）
            relative_volatility = (candle_volatility / close) * 100
            volatility = relative_volatility.rolling(window=timeperiod).mean()

            return volatility

        except Exception as e:
            logger.error(f"計算波動率時發生錯誤: {str(e)}")
            return None
        
        

    def plot_market_strength(self, ax):
        """
        繪製市場強度分析圖
        用200日新高和新低的比率來展示市場強弱
        """
        try:
            # 獲取市場所有股票的收盤價
            close_prices = data.get('price:收盤價').tail(660)  # 修改為 660 天 (480 + 200 - 20)
            
            # 計算200日新高和新低指標
            highs = pd.Series(index=close_prices.index, dtype=float)
            lows = pd.Series(index=close_prices.index, dtype=float)
            
            for date in close_prices.index:
                # 計算200天內的窗口數據
                window = close_prices.loc[:date].tail(200)
                current_prices = window.iloc[-1]
                max_prices = window.max()
                min_prices = window.min()
                
                # 檢查是否有有效價格（排除NaN）
                valid_stocks = (~current_prices.isna()).sum()
                if valid_stocks == 0:
                    highs[date] = 0
                    lows[date] = 0
                    continue
                
                # 計算達到新高和新低的股票比例
                high_stocks = (current_prices == max_prices).sum()
                low_stocks = (current_prices == min_prices).sum()
                highs[date] = high_stocks / valid_stocks
                lows[date] = low_stocks / valid_stocks
            
            # 計算移動平均來平滑數據
            highs_ma = highs.rolling(20, min_periods=1).mean()
            lows_ma = lows.rolling(20, min_periods=1).mean()
            
            # 只顯示最近480天的數據
            last_480_days = slice(-480, None)
            dates = highs.index[last_480_days]
            highs_ma = highs_ma[last_480_days]
            lows_ma = lows_ma[last_480_days]
            
            # 設置顏色方案
            colors = {
                'high': '#FF3B30',        # 新高線紅色
                'low': '#4CAF50',         # 新低線綠色
                'grid': '#E5E5EA',        # 網格線灰色
                'text': '#1C1C1E',        # 文字黑色
                'background': 'white',     # 背景白色
            }
            
            # 繪製新高和新低線
            high_line = ax.plot(dates, highs_ma,
                                label='200日新高比率',
                                color=colors['high'],
                                linewidth=3.5,  # 增加線寬
                                zorder=4)
            
            low_line = ax.plot(dates, lows_ma,
                                label='200日新低比率',
                                color=colors['low'],
                                linewidth=3.5,  # 增加線寬
                                zorder=4)
            
            # 標記最新值
            latest_high = highs_ma.iloc[-1]
            latest_low = lows_ma.iloc[-1]
            
            # 新高值標註
            ax.plot(dates[-1], latest_high, 'ko', zorder=5)
            ax.annotate(f'新高: {latest_high:.1%}',
                        xy=(dates[-1], latest_high),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontsize=12,
                        fontweight='bold',
                        color=colors['high'],
                        zorder=5)
            
            # 新低值標註
            ax.plot(dates[-1], latest_low, 'ko', zorder=5)
            ax.annotate(f'新低: {latest_low:.1%}',
                        xy=(dates[-1], latest_low),
                        xytext=(10, -20),
                        textcoords='offset points',
                        fontsize=12,
                        fontweight='bold',
                        color=colors['low'],
                        zorder=5)

            # 標註轉折點 (簡化版)
            for i in range(1, len(highs_ma) - 1):
                if (highs_ma.iloc[i] > highs_ma.iloc[i-1] and highs_ma.iloc[i] > highs_ma.iloc[i+1]) or \
                (highs_ma.iloc[i] < highs_ma.iloc[i-1] and highs_ma.iloc[i] < highs_ma.iloc[i+1]):
                    ax.plot(dates[i], highs_ma.iloc[i], 'ro', markersize=8, alpha=0.7)  # 標註紅色圓點
                if (lows_ma.iloc[i] > lows_ma.iloc[i-1] and lows_ma.iloc[i] > lows_ma.iloc[i+1]) or \
                (lows_ma.iloc[i] < lows_ma.iloc[i-1] and lows_ma.iloc[i] < lows_ma.iloc[i+1]):
                    ax.plot(dates[i], lows_ma.iloc[i], 'go', markersize=8, alpha=0.7)  # 標註綠色圓點
            
            # 設置圖表樣式
            ax.set_title('市場強度分析 (200日新高/新低比率)', 
                        fontsize=35, pad=20,
                        weight='bold')
            ax.set_ylabel('比\n率', fontsize=22,
                        rotation=0, labelpad=15, va='center')
            
            # 設置圖例
            legend = ax.legend(loc='upper right', frameon=True,
                                fontsize=25,
                                bbox_to_anchor=(0.99, 0.99),
                                ncol=1)
            legend.get_frame().set_alpha(0.9)
            legend.get_frame().set_edgecolor(colors['grid'])
            
            # 設置X軸日期格式
            ax.xaxis.set_major_locator(
                mdates.WeekdayLocator(byweekday=mdates.MO))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            plt.setp(ax.xaxis.get_majorticklabels(),
                    rotation=45, ha='right', fontsize=22)
            
            # 設置Y軸刻度標籤大小和格式
            ax.tick_params(axis='y', labelsize=22)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            
            # 美化圖表
            ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
            ax.set_facecolor(colors['background'])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(colors['grid'])
            ax.spines['bottom'].set_color(colors['grid'])
            ax.tick_params(axis='both', colors=colors['text'])
            
            # 設置Y軸範圍（從0開始，因為是比率）
            ax.set_ylim(0, max(highs_ma.max(), lows_ma.max()) * 1.1)
            
        except Exception as e:
            logger.error(f"繪製市場強度圖表時發生錯誤: {str(e)}")
            ax.text(0.5, 0.5, f"數據繪製失敗\n{str(e)}",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=25)





    def calculate_value(self):
        """計算小型股相對於0050的價值"""
        try:
            price = self.data.get('etl:adj_close')
            if price is None or price.empty:
                return pd.Series(dtype=float)

            recent_price = price.tail(120)

            # 計算每日報酬率
            s1 = recent_price.pct_change().mean(axis=1)

            # 確保0050存在於數據中
            if '0050' not in recent_price.columns:
                return pd.Series(dtype=float)

            s2 = recent_price.pct_change()['2330']

            # 計算滾動年化收益率
            val = ((s1 - s2).rolling(240, min_periods=1).mean().add(1) ** 240) - 1

            return val.fillna(0)

        except Exception as e:
            logger.error(f"計算價值時發生錯誤: {str(e)}")
            return pd.Series(dtype=float)

    def plot_combined_chart(self, ax):
        """繪製市場強度分析圖和小型股相對於0050的價值圖"""
        try:
            # 繪製市場強度分析圖
            self.plot_market_strength(ax)

            # 獲取小型股相對於0050的價值
            val = self.calculate_value()

            # 檢查是否有有效數據
            if val.empty:
                logger.warning("無法獲取有效的價值數據")
                return

            # 限制到最近120天
            val = val.tail(120)

            # 檢查數據有效性
            if val.isna().all() or np.isinf(val).all():
                logger.warning("所有數據都是NaN或無窮大")
                return

            # 創建第二個Y軸
            ax2 = ax.twinx()

            # 繪製小型股相對於0050的價值
            val.plot(ax=ax2, color='orange', linewidth=2.5,
                     label='Small Cap Alpha to 2330')

            # 繪製當前值標記
            if not val.empty and pd.notna(val.iloc[-1]):
                current_value = val.iloc[-1]
                current_index = val.index[-1]

                if np.isfinite(current_value):
                    ax2.axhline(current_value, color='yellow', linestyle='--')
                    ax2.axhline(0, color='white', linestyle='--')
                    ax2.plot(current_index, current_value, 'o', color='yellow')
                    ax2.text(current_index, current_value - 0.07,
                             f'{current_value:.2f}', color='yellow',
                             fontsize=16, fontweight='bold')

            # 設置Y軸格式
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

            # 設置標籤
            ax.set_ylabel('市場強度', fontsize=22)
            ax2.set_ylabel('相對於2330的價值', fontsize=22)

            # 設置X軸日期格式
            ax.xaxis.set_major_locator(
                mdates.WeekdayLocator(byweekday=mdates.MO))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45,
                     ha='right', fontsize=22)

            # 合併圖例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2,
                       loc='upper right',
                       fontsize=25,
                       bbox_to_anchor=(0.99, 0.99))

            # 設置Y軸刻度標籤大小和顏色
            ax2.tick_params(axis='y', labelsize=22, colors='orange')
            ax2.spines['right'].set_color('orange')

        except Exception as e:
            logger.error(f"繪製組合圖表時發生錯誤: {str(e)}")
            ax.text(0.5, 0.5, f"數據繪製失敗\n{str(e)}",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=25)

    def plot_market_analysis(self, ax_top, ax_bottom):
        """
        在兩個子圖上分別繪製市場強度和相對價值分析
        ax_top: 上方子圖，用於繪製市場強度
        ax_bottom: 下方子圖，用於繪製相對價值
        """
        try:
            # 顏色方案
            colors = {
                'strength': '#666666',     # 市場強度線灰色
                'strong': '#FF3B30',       # 強勢區域紅色
                'weak': '#4CAF50',         # 弱勢區域綠色
                'alpha': '#FFA500',        # Alpha線橘色
                'grid': '#E5E5EA',         # 網格線灰色
                'text': '#1C1C1E',         # 文字黑色
                'background': 'white',      # 背景白色
                'zero_line': '#000000'     # 0軸線黑色
            }

            # ===== 上圖：市場強度 =====
            close_prices = self.data.get("price:收盤價").tail(360).fillna(0)
            market_highs = pd.Series(index=close_prices.index, dtype=float)
            market_lows = pd.Series(index=close_prices.index, dtype=float)  # 新增

            for date in close_prices.index:
                window = close_prices.loc[:date].tail(200)
                current_prices = window.iloc[-1]
                max_prices = window.max()
                min_prices = window.min()  # 新增

                total_stocks = (current_prices != 0).sum()
                if total_stocks == 0:
                    market_highs[date] = 0
                    market_lows[date] = 0  # 新增
                    continue

                high_stocks = (current_prices == max_prices).sum()
                low_stocks = (current_prices == min_prices).sum()  # 新增
                market_highs[date] = high_stocks / total_stocks
                market_lows[date] = low_stocks / total_stocks  # 新增

            market_strength_high = market_highs.rolling(20, min_periods=1).mean()  # 新增
            market_strength_low = market_lows.rolling(20, min_periods=1).mean()  # 新增

            # 只顯示最近120天
            last_120_days = slice(-240, None)
            dates = market_strength_high.index[last_120_days]  # 修改
            market_strength_high = market_strength_high[last_120_days]  # 新增
            market_strength_low = market_strength_low[last_120_days]  # 新增

            # 繪製市場強度 (繪製兩條折線圖)
            ax_top.plot(dates, market_strength_high,
                        color=colors['strong'],  # 修改顏色
                        linewidth=3.5,
                        label='200日新高數量')  # 修改標籤
            
            ax_top.plot(dates, market_strength_low,
                        color=colors['weak'],  # 修改顏色
                        linewidth=3.5,
                        label='200日新低數量')  # 修改標籤

            ax_top.axhline(y=0, color=colors['zero_line'],
                        linestyle='-', linewidth=1.5,
                        alpha=0.5)

            # ===== 下圖：相對價值 =====
            small_cap_value = self.calculate_value()

            # 限制到最近120天的數據
            small_cap_value_recent = small_cap_value.tail(120)

            # 繪製小型股相對於0050的價值，使用面積填充表示好壞
            ax_bottom.fill_between(small_cap_value_recent.index,
                                    small_cap_value_recent,
                                    where=(small_cap_value_recent >= 0),
                                    color='red', alpha=0.3, label='超額報酬')

            ax_bottom.fill_between(small_cap_value_recent.index,
                                    small_cap_value_recent,
                                    where=(small_cap_value_recent < 0),
                                    color='green', alpha=0.3, label='負超額報酬')

            ax_bottom.plot(small_cap_value_recent.index, small_cap_value_recent,
                        color=colors['alpha'], linewidth=3)

            ax_bottom.axhline(y=0, color=colors['zero_line'],
                            linestyle='-', linewidth=1.5,
                            alpha=0.5)

            # 設置圖表樣式
            for ax in [ax_top, ax_bottom]:
                ax.set_facecolor(colors['background'])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color(colors['grid'])
                ax.spines['bottom'].set_color(colors['grid'])
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.legend(loc='upper right', fontsize=20)

            # 設置標題
            ax_top.set_title('市場強度', fontsize=30, pad=10)
            ax_bottom.set_title('小型股對2330的超額報酬', fontsize=30, pad=10)

        except Exception as e:
            print(f"繪製市場分析圖時發生錯誤: {str(e)}")



# plot2==============================================================================




    def analyze_stock_metrics(self, s):
        """分析股票指標並返回分析結果和關鍵指標"""
        try:
            # 獲取基本數據
            pe = self.data.get('price_earning_ratio:本益比')[s].tail(480).ffill()
            amt = self.data.get('price:成交金額')[s].tail(480).ffill()
            close_data = self.data.get("price:收盤價")[s].tail(480).ffill()
            volume_data = self.data.get("price:成交股數")[s].tail(480).ffill()
            OPGR = self.data.get("fundamental_features:營業利益成長率")[s].tail(480).ffill()

            # 確保所有數據都是Series並且索引一致
            data_series = {
                'pe': pe,
                'amt': amt,
                'close': close_data,
                'volume': volume_data,
                'opgr': OPGR
            }
            
            # 將所有數據轉換為Series
            for key, value in data_series.items():
                if not isinstance(value, pd.Series):
                    data_series[key] = pd.Series(value)

            # 重新賦值
            pe, amt, close_data, volume_data, OPGR = (
                data_series['pe'],
                data_series['amt'],
                data_series['close'],
                data_series['volume'],
                data_series['opgr']
            )

            # 計算本益比的歷史統計數據
            pe_stats = {
                'current': pe.iloc[-1] if not pe.empty else None,
                'max_120': pe.tail(120).max() if len(pe) >= 120 else None,
                'min_120': pe.tail(120).min() if len(pe) >= 120 else None,
                'mean_120': pe.tail(120).mean() if len(pe) >= 120 else None,
                'std_120': pe.tail(120).std() if len(pe) >= 120 else None,
                'max_480': pe.max() if not pe.empty else None,
                'min_480': pe.min() if not pe.empty else None,
                'mean_480': pe.mean() if not pe.empty else None,
                'std_480': pe.std() if not pe.empty else None
            }

            # 計算PEG（確保索引對齊）
            pe, OPGR = pe.align(OPGR)
            peg = pd.Series(index=pe.index, dtype=float)
            mask = (OPGR > 0) & (~pe.isna()) & (~OPGR.isna())
            peg.loc[mask] = pe.loc[mask] / OPGR.loc[mask]
            peg.loc[~mask] = float('nan')

            # 處理產業分類數據
            themes = self.data.get('security_industry_themes').copy()
            themes['category'] = themes['category'].apply(lambda s: eval(s) if isinstance(s, str) else s)
            exploded_themes = themes.explode(['category'])
            
            # 獲取產業資訊
            stock_industry = self.get_stock_industry(exploded_themes, s)
            industry_pe_position = self.get_industry_position_by_pe(exploded_themes, pe, stock_industry)
            
            # 計算產業PE中位數
            industry_pe_median = None
            if stock_industry:
                stock_ids = exploded_themes[exploded_themes['category'] == stock_industry]['stock_id'].unique()
                all_pe_data = self.data.get('price_earning_ratio:本益比')
                if len(stock_ids) > 0 and all_pe_data is not None:
                    ind_pe = all_pe_data[stock_ids]
                    if not ind_pe.empty and ind_pe.iloc[-1].notna().any():
                        industry_pe_median = ind_pe.iloc[-1].median()

            # 基本數據計算（加入錯誤處理）
            metrics = {
                'latest_pe': pe_stats['current'],
                'pe_avg_120': pe_stats['mean_120'],
                'pe_max_120': pe_stats['max_120'],
                'pe_min_120': pe_stats['min_120'],
                'pe_std_120': pe_stats['std_120'],
                'pe_max_480': pe_stats['max_480'],
                'pe_min_480': pe_stats['min_480'],
                'pe_mean_480': pe_stats['mean_480'],
                'pe_std_480': pe_stats['std_480'],
                'six_month_price_change': (close_data.iloc[-1] / close_data.iloc[-120] - 1) * 100 if len(close_data) >= 120 else None,
                'recent_month_avg_amt': amt.tail(20).mean() / 100000000 if len(amt) >= 20 else None,
                'today_amt': amt.iloc[-1] / 100000000 if not amt.empty else None,
                'latest_peg': peg.iloc[-1] if not peg.empty else None,
                'peg_avg_120': peg.tail(120).mean() if len(peg) >= 120 else None,
                'opgr': OPGR.iloc[-1] if not OPGR.empty else None,
                'industry': stock_industry,
                'industry_pe_rank': industry_pe_position.iloc[-1] if not industry_pe_position.empty else None,
                'industry_pe_median': industry_pe_median
            }

            # 分析結果
            analysis = []
            risk_level = 0  # 風險評分(0-10)

            # 1. 本益比分析（加入歷史高低點比較）
            if metrics['latest_pe'] is not None:
                pe_text_parts = []
                current_pe = metrics['latest_pe']
                
                # 與120天高低點比較
                if metrics['pe_max_120'] is not None and metrics['pe_min_120'] is not None:
                    if current_pe >= metrics['pe_max_120'] * 0.9:
                        pe_text_parts.append(f"本益比{current_pe:.1f}接近半年高點{metrics['pe_max_120']:.1f}")
                        risk_level += 2
                    elif current_pe <= metrics['pe_min_120'] * 1.1:
                        pe_text_parts.append(f"本益比{current_pe:.1f}接近半年低點{metrics['pe_min_120']:.1f}")
                        risk_level -= 1
                    
                    # 與平均值比較
                    pe_deviation = (current_pe / metrics['pe_avg_120'] - 1) * 100
                    if abs(pe_deviation) > 20:
                        pe_text_parts.append(f"較半年均值偏離{pe_deviation:.1f}%")
                        if pe_deviation > 0:
                            risk_level += 1
                        else:
                            risk_level -= 1

                # 與產業比較
                if metrics['industry_pe_median'] is not None:
                    pe_position = "低於" if metrics['industry_pe_rank'] else "高於"
                    pe_text_parts.append(f"{pe_position}{metrics['industry']}產業中位數{metrics['industry_pe_median']:.1f}")
                    
                if pe_text_parts:
                    analysis.append(" | ".join(pe_text_parts))

            # 2. PEG分析
            if metrics['latest_peg'] is not None and not np.isnan(metrics['latest_peg']):
                if metrics['latest_peg'] < 0.7:
                    analysis.append(f"PEG={metrics['latest_peg']:.2f}低於0.7，具成長性價值")
                    risk_level -= 1
                elif metrics['latest_peg'] > 1.5:
                    analysis.append(f"PEG={metrics['latest_peg']:.2f}高於1.5，估值偏高")
                    risk_level += 1
            elif metrics['opgr'] is not None and metrics['opgr'] <= 0:
                analysis.append(f"營業利益衰退{abs(metrics['opgr']):.1f}%，無法計算PEG")
                risk_level += 1


            # 4. 成交量分析
            if metrics['today_amt'] is not None and metrics['recent_month_avg_amt'] is not None:
                vol_ratio = metrics['today_amt'] / metrics['recent_month_avg_amt']
                vol_5d = amt.tail(5) / 100000000
                vol_trend = vol_5d.pct_change().mean()

                if vol_ratio > 2:
                    if vol_trend > 0:
                        analysis.append(f"連續放量{metrics['today_amt']:.1f}億, 買盤積極")
                        risk_level -= 1
                    else:
                        analysis.append(f"單日爆量{metrics['today_amt']:.1f}億, 需留意賣壓")
                        risk_level += 1
                elif vol_ratio < 0.5:
                    if vol_trend < 0:
                        analysis.append(f"持續量縮至{metrics['today_amt']:.1f}億, 觀望氣氛濃")
                        risk_level += 1
                    else:
                        analysis.append(f"暫時量縮至{metrics['today_amt']:.1f}億, 等待突破")

            # 5. 近期走勢分析
            if len(close_data) >= 2:
                returns = close_data.pct_change()
                if not returns.empty and not returns.isna().all():
                    volatility = returns.std() * np.sqrt(240)
                    rolling_std = returns.rolling(20).std()
                    if not rolling_std.empty and not rolling_std.isna().all():
                        vol_percentile = stats.percentileofscore(rolling_std.dropna(), returns.std())

                        if metrics['six_month_price_change'] is not None:
                            if metrics['six_month_price_change'] > 30:
                                if vol_percentile > 80:
                                    analysis.append(f"半年漲幅{metrics['six_month_price_change']:.1f}%且波動率高, 高風險警示")
                                    risk_level += 2
                                else:
                                    analysis.append(f"半年漲幅{metrics['six_month_price_change']:.1f}%,波動率正常, 注意回檔")
                                    risk_level += 1
                            elif metrics['six_month_price_change'] < -30:
                                if vol_percentile > 80:
                                    analysis.append(f"半年跌幅{abs(metrics['six_month_price_change']):.1f}%且波動率高, 建議觀望")
                                    risk_level += 1
                                else:
                                    analysis.append(f"半年跌幅{abs(metrics['six_month_price_change']):.1f}%,波動率降低, 可能築底")
                                    risk_level -= 1

            # 6. 風險評級總結
            risk_text = "風險適中"  # 默認值
            if risk_level >= 3:
                risk_text = "風險偏高"
            elif risk_level >= 1:
                risk_text = "風險中等"
            elif risk_level <= -2:
                risk_text = "風險偏低"

            # 加入風險評級
            if analysis:  # 只有在有分析結果時才加入風險評級
                analysis.append(f"風險評級: {risk_text}")

            # 返回分析結果和指標
            return {
                'analysis': " | ".join(analysis) if analysis else "無足夠資料進行分析",
                'metrics': metrics,
                'additional_metrics': {
                    'volatility': volatility if 'volatility' in locals() else None,
                    'vol_percentile': vol_percentile if 'vol_percentile' in locals() else None,
                    'risk_level': risk_level
                }
            }

        except Exception as e:
            print(f"分析時發生錯誤: {str(e)}")  # 添加錯誤日誌
            return {
                'analysis': f"分析發生錯誤: {str(e)}",
                'metrics': {
                    'latest_pe': None,
                    'pe_avg_120': None,
                    'pe_max_120': None,
                    'pe_min_120': None,
                    'pe_std_120': None,
                    'pe_max_480': None,
                    'pe_min_480': None,
                    'pe_mean_480': None,
                    'pe_std_480': None,
                    'six_month_price_change': None,
                    'recent_month_avg_amt': None,
                    'today_amt': None,
                    'latest_peg': None,
                    'peg_avg_120': None,
                    'opgr': None,
                    'industry': None,
                    'industry_pe_rank': None,
                    'industry_pe_median': None
                },
                'additional_metrics': {
                    'volatility': None,
                    'vol_percentile': None,
                    'risk_level': 0
                }
            }



    def get_stock_industry(self, exploded_themes, stock_id):
        """
        獲取股票所屬產業
        
        Parameters:
            exploded_themes (pd.DataFrame): 展開後的產業主題資料
            stock_id (str): 股票代碼
        
        Returns:
            str or None: 股票所屬產業，如果找不到則返回None
        """
        try:
            # 確保輸入參數的有效性
            if exploded_themes is None or stock_id is None:
                return None
                
            # 檢查必要的列是否存在
            if 'stock_id' not in exploded_themes.columns or 'category' not in exploded_themes.columns:
                return None
                
            # 過濾出特定股票的產業資料
            stock_themes = exploded_themes[exploded_themes['stock_id'] == stock_id]
            
            # 檢查是否找到資料
            if not stock_themes.empty and not stock_themes['category'].isna().all():
                return stock_themes['category'].iloc[0]
                
            return None
            
        except Exception as e:
            print(f"獲取股票產業時發生錯誤: {str(e)}")
            return None

    def get_stock_industry(self, exploded_themes, stock_id):
        """
        獲取股票所屬產業
        
        Parameters:
            exploded_themes (pd.DataFrame): 展開後的產業主題資料
            stock_id (str): 股票代碼
        
        Returns:
            str or None: 股票所屬產業，如果找不到則返回None
        """
        try:
            # 確保輸入參數的有效性
            if exploded_themes is None or stock_id is None:
                return None
                
            # 檢查必要的列是否存在
            if 'stock_id' not in exploded_themes.columns or 'category' not in exploded_themes.columns:
                return None
                
            # 過濾出特定股票的產業資料
            stock_themes = exploded_themes[exploded_themes['stock_id'] == stock_id]
            
            # 檢查是否找到資料
            if not stock_themes.empty and not stock_themes['category'].isna().all():
                return stock_themes['category'].iloc[0]
                
            return None
            
        except Exception as e:
            print(f"獲取股票產業時發生錯誤: {str(e)}")
            return None

    def get_industry_position_by_pe(self, exploded_themes, pe_data, industry):
        """
        計算產業中所有股票的PE位置，返回所有股票PE是否低於產業中位數的布林值
        
        Parameters:
            exploded_themes (pd.DataFrame): 展開後的產業主題資料
            pe_data (pd.DataFrame): PE比率數據
            industry (str): 產業類別
            
        Returns:
            pd.DataFrame: 布林DataFrame，表示每個股票的PE是否低於產業中位數
        """
        try:
            # 基本參數檢查
            if exploded_themes is None or pe_data is None or not industry:
                print(f"參數缺失。")
                return pd.DataFrame(index=pe_data.index if hasattr(pe_data, 'index') else [])

            # 確保pe_data是DataFrame類型
            if not isinstance(pe_data, pd.DataFrame):
                print("不支援的PE數據類型，僅支援 DataFrame。")
                return pd.DataFrame(index=[])

            # 確保exploded_themes包含必要的列
            if 'category' not in exploded_themes.columns or 'stock_id' not in exploded_themes.columns:
                print("exploded_themes 缺少必要欄位。")
                return pd.DataFrame(index=pe_data.index)
                
            # 先從pe_data中移除'9915'（如果存在）
            pe_data_filtered = pe_data.copy()
            if '9915' in pe_data_filtered.columns:
                pe_data_filtered = pe_data_filtered.drop('9915', axis=1)

            # 獲取同產業的股票代碼，排除'9915'
            industry_mask = exploded_themes['category'] == industry
            all_industry_stocks = [stock for stock in exploded_themes[industry_mask]['stock_id'].unique() if stock != '9915']
            
            # 過濾出在pe_data_filtered中存在的股票代碼
            stock_ids = [stock for stock in all_industry_stocks if stock in pe_data_filtered.columns]

            # 如果沒有找到同產業的股票
            if len(stock_ids) == 0:
                print(f"產業 {industry} 內沒有找到股票，或股票不在PE數據中。")
                return pd.DataFrame(index=pe_data_filtered.index)

            # 取得產業內股票的PE數據
            ind_pe = pe_data_filtered[stock_ids].copy()
            
            # 如果產業PE數據為空
            if ind_pe.empty:
                print(f"產業 {industry} 的PE資料不足，跳過計算。")
                return pd.DataFrame(index=pe_data_filtered.index)
                
            # 計算產業PE中位數
            ind_pe_med = ind_pe.median(axis=1)
            
            # 創建結果DataFrame
            result = pd.DataFrame(index=pe_data_filtered.index)
            
            # 對產業內每支股票計算PE位置
            for stock_id in stock_ids:
                try:
                    stock_pe = ind_pe[stock_id]
                    
                    # 確保索引對齊
                    stock_pe, ind_pe_med_aligned = stock_pe.align(ind_pe_med)
                    
                    # 計算位置
                    position = pd.Series(False, index=stock_pe.index)
                    valid_mask = ~stock_pe.isna() & ~ind_pe_med_aligned.isna()
                    position[valid_mask] = stock_pe[valid_mask] < ind_pe_med_aligned[valid_mask]
                    
                    # 添加到結果中
                    result[stock_id] = position
                except KeyError as e:
                    print(f"處理股票 {stock_id} 時發生錯誤: {str(e)}")
                    continue
            
            return result
                
        except Exception as e:
            print(f"計算產業PE位置時發生錯誤: {str(e)}")
            return pd.DataFrame(index=pe_data.index if hasattr(pe_data, 'index') else [])



    def plot_stock2(self, ax, s):
        try:
            # 獲取近360天K線數據
            open_data = self.data.get("price:開盤價")[s].tail(480)
            high_data = self.data.get("price:最高價")[s].tail(480)
            low_data = self.data.get("price:最低價")[s].tail(480)
            close_data = self.data.get("price:收盤價")[s].tail(480)
            volume_data = self.data.get("price:成交股數")[s].tail(480) / 1000

            # 獲取分析結果和指標
            result = self.analyze_stock_metrics(s)
            analysis_result = result['analysis']
            metrics = result['metrics']
            
            # 獲取 PEG 和營業利益成長率相關指標
            latest_peg = metrics.get('latest_peg')
            peg_avg_120 = metrics.get('peg_avg_120')
            opgr = metrics.get('opgr')  # 營業利益成長率

            # 從metrics中獲取需要的值，使用更安全的方式處理本益比
            latest_pe = metrics.get('latest_pe')
            # 移除對 fundamental_features:本益比 的直接依賴
            
            six_month_price_change = metrics.get('six_month_price_change')
            recent_month_avg_amt = metrics.get('recent_month_avg_amt')
            today_amt = metrics.get('today_amt')

            # 新增均線距離計算
            ma5 = close_data.rolling(window=5).mean()
            ma20 = close_data.rolling(window=20).mean()
            ma_distance = ((ma5 - ma20) / ma20 * 100)

            # 計算波動率和賣出轉換線
            volatility = self.compute_candle_volatility(timeperiod=20)[s].tail(480)
            high_rolling = high_data.rolling(60).max()

            # 只在波動率大於6%時計算賣出轉換線
            sell_convert_price = pd.Series(index=high_rolling.index, dtype=float)
            mask = volatility >= 6.0
            sell_convert_price[mask] = high_rolling[mask] - 2 * volatility[mask] * close_data[mask] / 100
            sell_convert_price[~mask] = np.nan
            
            
            weeks_52_high = high_data.rolling(window=240).max()  # 約52週
            weeks_52_low = low_data.rolling(window=240).min()    # 約52週
            current_price = close_data.iloc[-1]

            # 計算價格相關因子
            price_vs_52w_low = (current_price - weeks_52_low.iloc[-1]) / weeks_52_low.iloc[-1] * 100
            price_vs_52w_high = (weeks_52_high.iloc[-1] - current_price) / weeks_52_high.iloc[-1] * 100

            # 計算成交量因子
            week_avg_volume = volume_data.rolling(window=5).mean()  # 過去一週平均成交量
            volume_increase = (volume_data.iloc[-1] / week_avg_volume.iloc[-1] - 1) * 100  # 相較一週平均增加比例
            
            

            # 只顯示最近120天的數據
            open_data = open_data.tail(240)
            high_data = high_data.tail(240)
            low_data = low_data.tail(240)
            close_data = close_data.tail(240)
            volume_data = volume_data.tail(240)
            sell_convert_price = sell_convert_price.tail(240)
            ma5 = ma5.tail(240)
            ma20 = ma20.tail(240)
            ma_distance = ma_distance.tail(240)

            # 計算漲跌
            returns = close_data.pct_change()
            alert_volatility = returns.rolling(window=20).std() * np.sqrt(240) * 100

            # 計算波動率指標
            alert_vol_ma = alert_volatility.rolling(window=20).mean()
            alert_vol_change = alert_volatility.pct_change()
            relative_alert_vol = alert_volatility / alert_volatility.rolling(window=60).mean()

            # 計算成交量相關指標
            vol_ma5 = volume_data.rolling(window=5).mean()
            vol_ma20 = volume_data.rolling(window=20).mean()
            volume_change = volume_data.pct_change()

            # 設置警戒標準
            alert_vol_threshold = alert_volatility.mean() + alert_volatility.std()
            volume_drop_threshold = -0.2

            # 檢查最新的警訊
            latest_alert_vol = alert_volatility.iloc[-1]
            latest_vol_change = volume_change.iloc[-1]
            current_vol = volume_data.iloc[-1]

            # 生成信號條件
            low_vol_condition = alert_volatility.iloc[-1] < alert_vol_ma.iloc[-1] * 0.8
            vol_increase_condition = alert_vol_change.iloc[-1] > 0.1
            vol_spike_condition = alert_vol_change.iloc[-1] > 0.2
            extremely_low_vol = relative_alert_vol.iloc[-1] < 0.5

            # 生成警訊文字
            warning_texts = []

            # 波動率信號判斷
            if vol_spike_condition:
                warning_texts.append(f"波動率突增警告，隨時落跑 ({alert_vol_change.iloc[-1]*100:.1f}%)")
            elif low_vol_condition and vol_increase_condition and not extremely_low_vol:
                warning_texts.append(f"波動率開始上升，密切關注 ({alert_vol_change.iloc[-1]*100:.1f}%)")
            elif extremely_low_vol:
                warning_texts.append("異常低波動率，大行情出現")

            # 加入波動率異常警訊
            if latest_alert_vol > alert_vol_threshold:
                warning_texts.append(f"波動率異常評估風險 ({latest_alert_vol:.1f}%)")

            # 成交量萎縮判斷條件
            volume_shrink_conditions = []

            # 檢查各種成交量萎縮條件
            vol_to_ma5_ratio = (current_vol / vol_ma5.iloc[-1] - 1)
            vol_to_ma20_ratio = (current_vol / vol_ma20.iloc[-1] - 1)

            if vol_to_ma5_ratio < -0.5:
                volume_shrink_conditions.append(f"量低於5日均量 ({vol_to_ma5_ratio*100:.1f}%)")

            if vol_to_ma20_ratio < -0.7:
                volume_shrink_conditions.append(f"量低於20日均量 ({vol_to_ma20_ratio*100:.1f}%)")

            # 新增均線距離警示
            latest_ma_distance = ma_distance.iloc[-1]

            # 過熱區間
            if latest_ma_distance > 8:
                warning_texts.append(f"短線超漲警報 ({latest_ma_distance:.1f}%)")
            elif latest_ma_distance > 5:
                warning_texts.append(f"漲多注意風險 ({latest_ma_distance:.1f}%)")
            # 超跌區間
            elif latest_ma_distance < -8:
                warning_texts.append(f"短線超跌反彈 ({latest_ma_distance:.1f}%)")
            elif latest_ma_distance < -5:
                warning_texts.append(f"跌深可留意 ({latest_ma_distance:.1f}%)")

            # 趨勢判斷
            if len(ma_distance) >= 3:
                trend = ma_distance.iloc[-1] - ma_distance.iloc[-3]
                if abs(trend) > 3:
                    if trend > 0:
                        warning_texts.append("多頭加速")
                    else:
                        warning_texts.append("空頭加速")

            # 檢查連續3天成交量遞減
            last_3_days_change = volume_data.iloc[-3:].pct_change().dropna()
            if len(last_3_days_change) == 2 and all(last_3_days_change < 0):
                volume_shrink_conditions.append("連續3天量縮")

            # 加入成交量萎縮警訊
            if volume_shrink_conditions:
                warning_texts.append(f"成交量萎縮 ({' | '.join(volume_shrink_conditions)})")
            elif latest_vol_change < volume_drop_threshold:
                warning_texts.append(f"買賣雙方交易減少 ({latest_vol_change*100:.1f}%)")

            # 如果沒有警訊則顯示安心持有
            warning_text = " | ".join(warning_texts) if warning_texts else "安心持有"
            
            
            # 價格相關因子判斷
            if price_vs_52w_low >= 30:
                warning_texts.append(f"股價較近年低點上漲{price_vs_52w_low:.1f}%")
            if price_vs_52w_high <= 30:
                warning_texts.append(f"股價接近年高點 差距{price_vs_52w_high:.1f}%")

            # 判斷是否突破底部整理
            if len(close_data) >= 20:
                recent_low = low_data.tail(20).min()
                recent_avg_price = close_data.tail(20).mean()
                if (current_price > recent_low * 1.05 and 
                    volume_data.iloc[-1] > week_avg_volume.iloc[-1] * 1.3):
                    breakthrough_text = "突破底部整理："
                    breakthrough_details = []
                    
                    # 添加具體的突破細節
                    breakthrough_details.append(f"突破近期低點{((current_price/recent_low-1)*100):.1f}%")
                    breakthrough_details.append(f"成交量放大{((volume_data.iloc[-1]/week_avg_volume.iloc[-1]-1)*100):.1f}%")
                    
                    # 如果近期價格波動較小，更可能是真實突破
                    price_volatility = (close_data.tail(20).max() - recent_low) / recent_low * 100
                    if price_volatility < 10:
                        breakthrough_details.append("近期波動小，突破信號較強")
                    
                    breakthrough_text += "（" + "，".join(breakthrough_details) + "）"
                    warning_texts.append(breakthrough_text)

            # 成交量因子判斷
            if volume_increase >= 50:
                volume_text = f"成交量較週均增{volume_increase:.1f}%"
                if volume_increase >= 100:
                    volume_text += "（成交爆量，交易活絡）"
                elif volume_increase >= 50:
                    volume_text += "（成交明顯放大）"
                warning_texts.append(volume_text)
                
            warning_text = " | ".join(warning_texts) if warning_texts else "安心持有"

            
            

            # 定義配色方案
            colors = {
                'up_candle': '#FF3B30',      # 紅色
                'down_candle': '#34C759',    # 綠色
                'up_volume': '#FF6666',
                'down_volume': '#66CC75',
                'sell_line': '#FF9500',
                'grid': '#E5E5EA',
                'text': '#1C1C1E',
                'bull_text': '#FF3B30',      # 多頭文字顏色（紅色）
                'bear_text': '#34C759',      # 空頭文字顏色（綠色）
                'neutral_text': '#1C1C1E'    # 中性文字顏色（黑色）
            }

            def get_title_color(warning_text):
                bull_signals = {
                    "多頭加速": 2,
                    "異常低波動率": 1,
                    "短線超跌反彈": 2,
                    "跌深可留意": 1,
                    "突破底部整理": 2,  # 修改為更明確的權重
                    "成交爆量": 2,      # 新增爆量信號
                    "成交明顯放大": 1   # 新增普通放量信號
                }
                
                bear_signals = {
                    "空頭加速": 2,
                    "波動率突增警告": 2,
                    "短線超漲警報": 2,
                    "漲多注意風險": 1,
                    "成交量萎縮": 1,
                    "買賣雙方交易減少": 1,
                    "股價接近52週高點": 1  # 新增
                }
                
                bull_score = 0
                bear_score = 0
                
                for signal, weight in bull_signals.items():
                    if signal in warning_text:
                        bull_score += weight
                        
                for signal, weight in bear_signals.items():
                    if signal in warning_text:
                        bear_score += weight
                
                if bull_score > bear_score and bull_score >= 2:
                    return colors['bull_text']
                elif bear_score > bull_score and bear_score >= 2:
                    return colors['bear_text']
                else:
                    return colors['neutral_text']

            def format_title_line(text, max_length=50):
                """
                將長文字按照指定長度分行，優先按照分隔符號分行
                """
                if len(text) <= max_length:
                    return text
                
                if ' | ' in text:
                    parts = text.split(' | ')
                    lines = []
                    current_line = parts[0]
                    
                    for part in parts[1:]:
                        if len(current_line) + len(part) + 3 <= max_length:
                            current_line += ' | ' + part
                        else:
                            lines.append(current_line)
                            current_line = part
                    
                    lines.append(current_line)
                    return '\n'.join(lines)
                
                words = text.split()
                lines = []
                current_line = words[0]
                
                for word in words[1:]:
                    if len(current_line) + len(word) + 1 <= max_length:
                        current_line += ' ' + word
                    else:
                        lines.append(current_line)
                        current_line = word
                
                lines.append(current_line)
                return '\n'.join(lines)

            # 獲取標題顏色
            title_color = get_title_color(warning_text)

            # 創建日期範圍
            dates = pd.date_range(end=close_data.index[-1], periods=len(close_data))

            # 計算漲跌
            up = close_data > open_data
            down = close_data < open_data

            # 計算漲跌幅
            price_change = close_data.iloc[-1] - close_data.iloc[-2]
            price_change_pct = (price_change / close_data.iloc[-2]) * 100

            # 獲取股票資訊
            stock_info = self.stock_topics_df[self.stock_topics_df['stock_no'] == int(s)]
            if not stock_info.empty:
                stock_name = stock_info['stock_name'].values[0]
                topic = stock_info['topic'].values[0]
                current_price = close_data.iloc[-1]
            else:
                stock_name = f'股票{s}'
                topic = ''
                current_price = close_data.iloc[-1]

            # 計算價格區間
            price_range = ((high_data.tail(60).max() - low_data.tail(60).min()) / 
                        low_data.tail(60).min()) * 100

            # 構建本益比相關文字
            pe_text_parts = []
            # 只保留PEG相關資訊
            if latest_peg is not None and not np.isnan(latest_peg):
                pe_text_parts.append(f"PEG: {latest_peg:.2f}")

            pe_text = " | ".join(pe_text_parts)

            # 構建漲幅和價差文字
            price_metrics = []

            # 52週高低點相關指標及建議
            if price_vs_52w_low is not None and not np.isnan(price_vs_52w_low):
                if price_vs_52w_low >= 30:
                    price_metrics.append(f"近年低點: +{price_vs_52w_low:.1f}% 符合低點上漲30%")
                else:
                    price_metrics.append(f"近年低點: +{price_vs_52w_low:.1f}% 未達低點上漲30%")

            if price_vs_52w_high is not None and not np.isnan(price_vs_52w_high):
                if price_vs_52w_high <= 30:
                    price_metrics.append(f"近年高點: -{price_vs_52w_high:.1f}% 在高點30%區間內")
                else:
                    price_metrics.append(f"近年高點: -{price_vs_52w_high:.1f}% 遠離高點")

            # 原有的6月漲幅和3月價差指標
            if six_month_price_change is not None and not np.isnan(six_month_price_change):
                price_metrics.append(f"6月漲幅: {six_month_price_change:.2f}%")
            if price_range is not None and not np.isnan(price_range):
                price_metrics.append(f"3月價差: {price_range:.2f}%")

            # 組合所有價格指標文字
            price_text = " | ".join(price_metrics) if price_metrics else "無價格變動資料"

            # 如果同時符合兩個條件，加入特別提示
            if (price_vs_52w_low >= 30 and price_vs_52w_high <= 30):
                price_metrics.append("符合低點上漲且在高點區間內")


            # 構建成交額文字
            volume_metrics = []
            if recent_month_avg_amt is not None and not np.isnan(recent_month_avg_amt):
                volume_metrics.append(f"月均額: {recent_month_avg_amt:.2f}億")
            if today_amt is not None and not np.isnan(today_amt):
                volume_metrics.append(f"今日額: {today_amt:.2f}億")
            volume_text = " | ".join(volume_metrics) if volume_metrics else "無成交額資料"

            
            # 構建 PEG 和營業利益成長率文字
            growth_metrics = []

            # PEG 分析文字
            if latest_peg is not None and not np.isnan(latest_peg):
                if latest_peg < 0.7:
                    growth_metrics.append(f"PEG={latest_peg:.2f}<0.7 具成長價值")
                elif latest_peg > 1.5:
                    growth_metrics.append(f"PEG={latest_peg:.2f}>1.5 估值偏高")
                else:
                    growth_metrics.append(f"PEG={latest_peg:.2f} 估值合理")

            # 營業利益成長率分析文字
            if opgr is not None and not np.isnan(opgr):
                if opgr > 20:
                    growth_metrics.append(f"營業利益年增{opgr:.1f}% 成長強勁")
                elif opgr < 0:
                    growth_metrics.append(f"營業利益年減{abs(opgr):.1f}% 獲利衰退")
                else:
                    growth_metrics.append(f"營業利益年增{opgr:.1f}%")

            growth_text = " | ".join(
                growth_metrics) if growth_metrics else "無成長指標資料"
            
            
            # 設置標題
            title_lines = [
                f'{stock_name}({s})${current_price:.2f}漲跌: {price_change:.2f} ({price_change_pct:.2f}%)',
                format_title_line(warning_text),
                format_title_line(growth_text),  
                format_title_line(price_text),
                format_title_line(volume_text),
                format_title_line(analysis_result)
               
            ]

            # 設置完整標題
            full_title = '\n'.join(title_lines)
            ax.set_title(full_title, fontsize=30, pad=20, y=1, color=title_color, linespacing=1.3)

            # 設置K線圖的寬度
            width = 0.8
            width2 = 0.2

            # 繪製蠟燭實體
            ax.bar(dates[up], close_data[up] - open_data[up], width,
                bottom=open_data[up], color=colors['up_candle'],
                edgecolor='black', linewidth=0.5, alpha=0.9)
            ax.bar(dates[down], open_data[down] - close_data[down], width,
                bottom=close_data[down], color=colors['down_candle'],
                edgecolor='black', linewidth=0.5, alpha=0.9)

            # 繪製上下影線
            ax.vlines(dates[up], high_data[up], low_data[up],
                    color=colors['up_candle'], linewidth=width2)
            ax.vlines(dates[down], high_data[down], low_data[down],
                    color=colors['down_candle'], linewidth=width2)

            # 繪製賣出轉換線（只在波動率>6%時顯示）
            valid_sell_line = sell_convert_price.dropna()
            if not valid_sell_line.empty:
                ax.plot(valid_sell_line.index, valid_sell_line.values,
                        color=colors['sell_line'], linestyle='--',
                        label='高檔跌破賣出', linewidth=2.5, alpha=0.8)
                
                #用來顯示圖例
                ax.legend(loc='upper left', fontsize=20, framealpha=0.8)

            # 繪製成交量
            ax2 = ax.twinx()
            ax2.bar(dates[up], volume_data[up], width,
                    color=colors['up_volume'], alpha=0.7)
            ax2.bar(dates[down], volume_data[down], width,
                    color=colors['down_volume'], alpha=0.7)

            # 設置成交量Y軸
            max_volume = max(volume_data)
            ax2.set_ylim(0, max_volume * 2.5)
            ax2.tick_params(axis='y', labelsize=16, colors='black')
            ax2.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f'{int(x)}\n張'))
            
            # 確保圖例不會被其他元素遮擋
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)

            return ax

        except Exception as e:
            print(f"繪圖發生錯誤: {str(e)}")
            return None


# =====================revenue================================================================================================

    def plot_revenue(self, ax, stock_id):
        """
        繪製股票營收圖表

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            繪圖用的axes物件
        stock_id : str
            股票代碼

        Returns:
        --------
        matplotlib.axes.Axes
            繪製完成的axes物件
        """
        try:
            # 1. 數據獲取和預處理
            if not isinstance(stock_id, str):
                stock_id = str(stock_id)
                
            print(f"=== 檢查點 1：開始處理股票 {stock_id} ===")

            rev = data.get("monthly_revenue:當月營收")[stock_id]
            
            print(f"營收資料筆數：{len(rev) if rev is not None else 0}")
            print(f"最新營收日期：{rev.index[-1] if rev is not None and len(rev) > 0 else 'N/A'}")
            print(f"是否有空值：{rev.isna().sum() if rev is not None else 'N/A'}")
            
            if rev is None or rev.empty:
                raise ValueError("無營收數據")

            rev = rev.tail(120)
            close = data.get("price:收盤價")[stock_id]
            if close is None or close.empty:
                raise ValueError("無股價數據")

            monthly_close = close.resample("M").mean().tail(120)

            # 2. 計算技術指標
            indicators = self._calculate_technical_indicators(rev)
            rev3, rev12, rev24, yoy, growth = (
                indicators['ma3'],
                indicators['ma12'],
                indicators['ma24'],
                indicators['yoy'],
                indicators['growth']
            )

            # 3. 設置圖表軸
            ax1 = ax
            ax2 = ax1.twinx()
            last_24_months = slice(-24, None)

            # 確保數據長度正確
            if len(rev.iloc[last_24_months]) == 0:
                raise ValueError("無最近24個月的數據")

            def get_color(yoy_val, growth_val, is_max):
                print(f"=== 顏色判斷詳細資訊 ===")
                print(f"年增率: {yoy_val:.2f}%")
                print(f"趨勢向上: {growth_val}")
                print(f"創新高: {is_max}")
                """
                決定柱狀圖顏色的函數
                
                Parameters:
                -----------
                yoy_val : float
                    年增率值
                growth_val : bool
                    True 表示12個月移動平均成長中，False 表示下降中
                is_max : bool
                    True 表示是9個月內的新高
                """
                # 如果是創新高，使用特殊顏色
                if is_max:
                    print("使用創新高顏色: #FFC125")
                    return '#FFC125'
    
                # 根據年增率和成長趨勢決定顏色
                if yoy_val > 0:
                    if growth_val:
                        color = (
                            '#FF0000' if yoy_val > 20 else
                            '#FF3333' if yoy_val > 10 else
                            '#FF6666' if yoy_val > 5 else
                            '#FF9999'
                        )
                        print(f"正成長且趨勢向上，使用顏色: {color}")
                        return color
                    else:
                        color = (
                            '#FF3333' if yoy_val > 20 else
                            '#FF6666' if yoy_val > 10 else
                            '#FF9999' if yoy_val > 5 else
                            '#FFCCCC'
                        )
                        print(f"正成長但趨勢向下，使用顏色: {color}")
                        return color
                else:
                    if not growth_val:
                        color = (
                            '#006400' if yoy_val < -20 else
                            '#008000' if yoy_val < -10 else
                            '#228B22' if yoy_val < -5 else
                            '#90EE90'
                        )
                        print(f"負成長且趨勢向下，使用顏色: {color}")
                        return color
                    else:
                        color = (
                            '#008000' if yoy_val < -20 else
                            '#228B22' if yoy_val < -10 else
                            '#90EE90' if yoy_val < -5 else
                            '#98FB98'
                        )
                        print(f"負成長但趨勢向上，使用顏色: {color}")
                        return color
            # 4. 繪製圖表

            # 在繪製柱狀圖時，修改參數傳遞方式
            colors = [get_color(y, g, m) 
                    for y, g, m in zip(yoy.iloc[last_24_months],
                                    growth.iloc[last_24_months],
                                    indicators['is_ma9_max'].iloc[last_24_months])]

            print("=== 繪圖階段 ===")
            print(f"最後一筆是否創新高: {indicators['is_ma9_max'].iloc[-1]}")
            print(f"顏色列表最後一個: {colors[-1]}")


            bars = ax2.bar(rev.index[last_24_months],
                        rev.values[last_24_months],
                        color=colors,
                        alpha=0.6,
                        width=20,
                        zorder=1,
                        edgecolor='black',
                        linewidth=1)
            # 為創新高的柱狀圖添加特殊效果
            for i, (is_max, bar) in enumerate(zip(
                indicators['is_ma9_max'].iloc[last_24_months],
                bars)):
                if is_max:
                    bar.set_alpha(0.8)
                    bar.set_linewidth(2)
                    bar.set_edgecolor('gold')
                    
                    # 添加星號標記
                    ax2.plot(bar.get_x() + bar.get_width()/2, 
                            bar.get_height(), 
                            'k*', 
                            markersize=15,
                            zorder=5,
                            label='創新高' if i == 0 else "")


                
            # 5. 繪製移動平均線
            lines = self._plot_moving_averages(
                ax2, rev3, rev12, rev24, last_24_months)
            line3ma, line12ma, line24ma = lines

            # 6. 繪製月均價
            line_monthly_close = self._plot_monthly_close(
                ax1, monthly_close, last_24_months)

            # 7. 清除原有標題
            ax1.set_title('')
            for text in ax1.texts:
                text.remove()

            # 8. 設置標題信息
            title_info = self._generate_title(stock_id, rev)

            # 調整圖表邊距，為標題留出適當空間
            plt.subplots_adjust(top=0.86)

            # 調整標題位置和間距
            y_position = 1.45
            line_spacing = 0.11
            
            # 第一行：股票名稱和代號（置中）
            ax1.text(0.5, y_position,
                    title_info['main_title'],
                    transform=ax1.transAxes,
                    ha='center', va='bottom',
                    fontsize=30, fontweight='bold')

            # 第二行：產業主題和整體評估
            y_position -= line_spacing
            ax1.text(0.5, y_position,
                    title_info['subtitle'],
                    transform=ax1.transAxes,
                    ha='center', va='bottom',
                    fontsize=30, fontweight='bold')

            # 第三行：當月營收和營收狀態
            y_position -= line_spacing
            ax1.text(0.5, y_position,
                    title_info['revenue_line'],
                    transform=ax1.transAxes,
                    ha='center', va='bottom',
                    fontsize=30, fontweight='bold')

            # 營收表現文字第四行 - 年增率、月增率、季增率
            y_position -= line_spacing
            ax1.text(0.15, y_position,
                    f"{title_info['yoy_text']}",
                    transform=ax1.transAxes,
                    ha='center', va='bottom',
                    fontsize=30, fontweight='bold',
                    color=title_info['yoy_color'])

            ax1.text(0.5, y_position,
                    f"{title_info['mom_text']}",
                    transform=ax1.transAxes,
                    ha='center', va='bottom',
                    fontsize=30, fontweight='bold',
                    color=title_info['mom_color'])

            ax1.text(0.85, y_position,
                    f"{title_info['qoq_text']}",
                    transform=ax1.transAxes,
                    ha='center', va='bottom',
                    fontsize=30, fontweight='bold',
                    color=title_info['qoq_color'])

            # 營收表現文字第五行 - MA比率和PR值
            y_position -= line_spacing
            ax1.text(0.35, y_position,
                    f"{title_info['ma_ratio_text']}",
                    transform=ax1.transAxes,
                    ha='center', va='bottom',
                    fontsize=30, fontweight='bold',
                    color=title_info['ma_ratio_color'])

            ax1.text(0.75, y_position,
                    f"{title_info['pr_text']}",
                    transform=ax1.transAxes,
                    ha='center', va='bottom',
                    fontsize=30, fontweight='bold',
                    color=title_info['pr_color'])

            # 10. 添加其他圖表元素
            self._add_yoy_labels(ax2, bars, yoy, rev)  # 標示年增率
            self._set_axes_format(ax1, ax2)  # 設置軸標籤和格式
            self._set_legend(ax1, line_monthly_close,
                            line3ma, line12ma, line24ma)  # 設置圖例
            ax1.grid(True, linestyle='--', alpha=0.7)  # 設置網格

        except Exception as e:
            logger.error(f"繪製營收圖表時發生錯誤: {str(e)}")
            logger.exception("詳細錯誤信息：")
            ax.text(0.5, 0.5, '無法取得營收資料\n請檢查資料來源',
                    ha='center', va='center', fontsize=20)

        return ax
    
    
        # 在使用官方數據前，先確保索引對齊
    def _align_data(self, official_data, calculated_data):
        """確保官方數據和計算數據的索引對齊"""
        if not isinstance(official_data, pd.Series):
            official_data = pd.Series(official_data)
        # 確保兩個序列有相同的索引
        common_index = official_data.index.intersection(calculated_data.index)
        # 如果沒有共同索引，返回計算數據
        if len(common_index) == 0:
            return calculated_data
        # 對齊數據
        official_aligned = official_data[common_index]
        calculated_aligned = calculated_data[common_index]
        # 合併數據，優先使用官方數據
        result = calculated_aligned.copy()
        result.loc[~official_aligned.isna()] = official_aligned.loc[~official_aligned.isna()]
        
        return result


    def _calculate_technical_indicators(self, revenue_data):
        """
        計算技術指標
        
        參數:
            revenue_data (pd.Series): 月營收數據時間序列
                
        返回:
            dict: 包含以下技術指標的字典:
                - ma3: 3個月移動平均
                - ma12: 12個月移動平均
                - ma24: 24個月移動平均
                - yoy: 年增率
                - mom: 月增率
                - qoq: 季增率
                - growth: 成長趨勢
                - ma3_ma12_ratio: 3月/12月均線比率
                - pr_value: PR值
                - is_ma9_max: 是否創9個月新高
                - rev_ma2: 2個月移動平均
                - rev_ma9_max: 9個月最高值
        """
        try:
            # 1. 數據預處理
            if revenue_data.empty:
                logger.warning("輸入的營收數據為空")
                return self._get_default_indicators()

            if revenue_data.isna().all():
                logger.warning("輸入的營收數據全為空值")
                return self._get_default_indicators()

            if not isinstance(revenue_data, pd.Series):
                logger.error("輸入數據必須是 pandas Series 類型")
                return self._get_default_indicators()

            # 2. 基本數據處理
            revenue_data = revenue_data.fillna(method='ffill')
            
            print("\n=== 檢查點 3：技術指標計算 ===")
            print(f"輸入資料筆數：{len(revenue_data)}")
            print(f"資料範圍：{revenue_data.index[0]} 到 {revenue_data.index[-1]}")
            print(f"空值比例：{revenue_data.isna().mean():.2%}")
            
            # 3. 計算移動平均
            ma3 = revenue_data.rolling(window=3, min_periods=1).mean()
            print("\n移動平均計算結果：")
            print(f"3MA 最新值：{ma3.iloc[-1]:,.0f}")
            ma12 = revenue_data.rolling(window=12, min_periods=1).mean()
            ma24 = revenue_data.rolling(window=24, min_periods=1).mean()
            
            # 4. 計算新高指標
            rev_ma2 = revenue_data.rolling(2, min_periods=1).mean()
            rev_ma9_max = rev_ma2.rolling(9, min_periods=1).max()
            is_ma9_max = (revenue_data >= rev_ma9_max)
            
            # 5. 計算年增率 (YoY)
            yoy = pd.Series(index=revenue_data.index)
            shifted_revenue_12 = revenue_data.shift(12)
            mask_yoy = shifted_revenue_12 != 0
            yoy[mask_yoy] = (revenue_data[mask_yoy] / shifted_revenue_12[mask_yoy] - 1) * 100
            yoy = yoy.fillna(0)

            # 6. 計算月增率 (MoM)
            mom = pd.Series(index=revenue_data.index)
            shifted_revenue_1 = revenue_data.shift(1)
            mask_mom = shifted_revenue_1 != 0
            mom[mask_mom] = (revenue_data[mask_mom] / shifted_revenue_1[mask_mom] - 1) * 100
            mom = mom.fillna(0)

            # 7. 計算季增率 (QoQ)
            qoq = pd.Series(index=revenue_data.index)
            shifted_revenue_3 = revenue_data.shift(3)
            mask_qoq = shifted_revenue_3 != 0
            qoq[mask_qoq] = (revenue_data[mask_qoq] / shifted_revenue_3[mask_qoq] - 1) * 100
            qoq = qoq.fillna(0)

            # 8. 使用官方數據（如果有的話）
            try:
                if hasattr(self, 'data'):
                    yoy_official = self.data.get('monthly_revenue:去年同月增減(%)', {}).get(revenue_data.name, pd.Series())
                    mom_official = self.data.get('monthly_revenue:上月比較增減(%)', {}).get(revenue_data.name, pd.Series())
                    qoq_official = self.data.get('monthly_revenue:前期比較增減(%)', {}).get(revenue_data.name, pd.Series())
                    
                    # 確保數據是 Series 類型並對齊索引
                    if not isinstance(yoy_official, pd.Series):
                        yoy_official = pd.Series(yoy_official)
                    if not isinstance(mom_official, pd.Series):
                        mom_official = pd.Series(mom_official)
                    if not isinstance(qoq_official, pd.Series):
                        qoq_official = pd.Series(qoq_official)
                    
                    yoy_official.index = revenue_data.index
                    mom_official.index = revenue_data.index
                    qoq_official.index = revenue_data.index
                     
                    yoy = self._align_data(yoy_official, yoy)
                    mom = self._align_data(mom_official, mom)
                    qoq = self._align_data(qoq_official, qoq)
                    
            except Exception as e:
                logger.warning(f"使用自行計算的增率數據: {str(e)}")

            # 9. 計算成長趨勢
            growth = (revenue_data.rolling(12).mean() > revenue_data.shift(12).rolling(12).mean())

            # 10. 計算3MA/12MA比率
            ma3_ma12_ratio = (ma3 / ma12 - 1) * 100

            # 11. 計算PR值
            try:
                recent_60m = revenue_data.tail(60)
                latest_rev = revenue_data.iloc[-1]
                
                if pd.isna(latest_rev) or recent_60m.isna().all():
                    pr_value = 50
                else:
                    valid_data = recent_60m.dropna()
                    if len(valid_data) > 0:
                        pr_value = stats.percentileofscore(valid_data, latest_rev)
                    else:
                        pr_value = 50
            except Exception as e:
                logger.error(f"計算PR值時發生錯誤: {str(e)}")
                pr_value = 50

            # 12. 診斷信息輸出
            print("\n=== 技術指標計算結果 ===")
            print(f"最新營收: {revenue_data.iloc[-1]:,.0f}")
            print(f"年增率: {yoy.iloc[-1]:.2f}%")
            print(f"月增率: {mom.iloc[-1]:.2f}%")
            print(f"季增率: {qoq.iloc[-1]:.2f}%")
            print(f"3MA/12MA比率: {ma3_ma12_ratio.iloc[-1]:.2f}%")
            print(f"PR值: {pr_value:.0f}")
            print(f"是否創新高: {is_ma9_max.iloc[-1]}")

            # 13. 返回結果
            return {
                'ma3': ma3,
                'ma12': ma12,
                'ma24': ma24,
                'yoy': yoy,
                'mom': mom,
                'qoq': qoq,
                'growth': growth,
                'ma3_ma12_ratio': ma3_ma12_ratio.fillna(0),
                'pr_value': pr_value,
                'is_ma9_max': is_ma9_max,
                'rev_ma2': rev_ma2,
                'rev_ma9_max': rev_ma9_max
            }

        except Exception as e:
            logger.error(f"計算技術指標時發生錯誤: {str(e)}")
            logger.exception("詳細錯誤信息：")
            return self._get_default_indicators()

    def _get_default_indicators(self):
        """返回預設的技術指標值"""
        return {
            'ma3': pd.Series(),
            'ma12': pd.Series(),
            'ma24': pd.Series(),
            'yoy': pd.Series(),
            'mom': pd.Series(),
            'qoq': pd.Series(),
            'growth': pd.Series(),
            'ma3_ma12_ratio': pd.Series(),
            'pr_value': 50,
            'is_ma9_max': pd.Series(),
            'rev_ma2': pd.Series(),
            'rev_ma9_max': pd.Series()
        }



    def _plot_moving_averages(self, ax, rev3, rev12, rev24, time_slice):
        """繪製移動平均線"""
        line3ma, = ax.plot(rev3.index[time_slice], rev3.values[time_slice],
                           color='#1E90FF', label='3MA', linewidth=2.5, zorder=3)

        line12ma, = ax.plot(rev12.index[time_slice], rev12.values[time_slice],
                            color='#9932CC', label='12MA', linewidth=2.5, zorder=3)

        line24ma, = ax.plot(rev24.index[time_slice], rev24.values[time_slice],
                            color='#FF1493', label='24MA', linewidth=2.5, zorder=3)

        return line3ma, line12ma, line24ma

    def _plot_monthly_close(self, ax, monthly_close, time_slice):
        """繪製月均價"""
        line, = ax.plot(monthly_close.index[time_slice],
                        monthly_close.values[time_slice],
                        color='black',
                        label='月均價',
                        linewidth=2.5,
                        linestyle='--',
                        zorder=4)
        return line

    def _generate_title(self, stock_id, rev):
        """
        生成圖表標題，包含營收表現分析與自動判讀
        
        Parameters:
        -----------
        stock_id : str
            股票代碼
        rev : pandas.Series
            營收數據
                
        Returns:
        --------
        dict
            包含標題相關資訊的字典
        """
        try:
            # 預處理數據
            if rev is None or len(rev) == 0:
                logger.error(f"股票 {stock_id} 無營收數據")
                return self._get_default_title_dict(stock_id)
            
            # 修改獲取最新有效營收月份的部分
            latest_date = rev.index[-1]
            latest_rev = rev.iloc[-1]

            # 修改點 1：當數據為 NaN 時的月份處理
            if pd.isna(latest_rev):
                valid_rev = rev.dropna()
                if not valid_rev.empty:
                    latest_date = valid_rev.index[-1]
                    latest_rev = valid_rev.iloc[-1]
                    # 這裡修改了
                    revenue_month = f"{12 if latest_date.month == 1 else latest_date.month - 1}月"
                    month_prefix = "最新營收為"
                else:
                    revenue_month = "無營收資料"
                    month_prefix = ""
            else:
                # 修改點 2：一般情況下的月份處理
                revenue_month = f"{12 if latest_date.month == 1 else latest_date.month - 1}月"
                month_prefix = ""
                   
            # 檢查數據品質
            if rev.isna().sum() / len(rev) > 0.5:  # 如果超過50%是空值
                logger.warning(f"股票 {stock_id} 的營收數據品質不佳")

            # 計算所有技術指標
            indicators = self._calculate_technical_indicators(rev)
            
            if indicators is None or not indicators:
                logger.error(f"股票 {stock_id} 技術指標計算失敗")
                return self._get_default_title_dict(stock_id)
            
            # 檢查是否創新高
            is_new_high = indicators.get('is_ma9_max', pd.Series()).iloc[-1] if 'is_ma9_max' in indicators else False
            new_high_text = "創新高 " if is_new_high else ""

            # 使用安全的資料獲取方法
            def safe_get_value(data_dict, key, idx=-1, default=0.0):
                try:
                    if key not in data_dict:
                        logger.warning(f"找不到指標 {key}")
                        return default
                        
                    value = data_dict[key]
                    
                    if isinstance(value, (pd.Series, np.ndarray)):
                        return value.iloc[idx] if isinstance(value, pd.Series) else value[idx]
                    
                    return value if not pd.isna(value) else default
                    
                except Exception as e:
                    logger.error(f"獲取 {key} 數據時發生錯誤: {str(e)}")
                    return default

            # 安全地獲取各項指標
            yoy_value = safe_get_value(indicators, 'yoy', -1)
            mom_value = safe_get_value(indicators, 'mom', -1)
            qoq_value = safe_get_value(indicators, 'qoq', -1)
            ma3_ma12_ratio = safe_get_value(indicators, 'ma3_ma12_ratio', -1)
            pr_value = safe_get_value(indicators, 'pr_value', default=50)

            # === 1. 營收強度判讀 ===
            revenue_status = (
                "營收強勢成長" if yoy_value >= 20 else
                "營收穩定成長" if yoy_value >= 10 else
                "營收微幅成長" if yoy_value >= 0 else
                "營收略為下滑" if yoy_value >= -10 else
                "營收明顯衰退"
            )

            # === 2. 趨勢判讀 ===
            trend_status = (
                "上升趨勢強勁" if ma3_ma12_ratio >= 5 else
                "呈現上升趨勢" if ma3_ma12_ratio >= 0 else
                "趨勢略為下滑" if ma3_ma12_ratio >= -5 else
                "下降趨勢明顯"
            )

            # === 3. PR值強度判讀 ===
            pr_status = (
                "極度強勢" if pr_value >= 90 else
                "表現強勢" if pr_value >= 70 else
                "表現平穩" if pr_value >= 30 else
                "表現偏弱" if pr_value >= 10 else
                "極度弱勢"
            )

            # === 4. 季增率判讀 ===
            qoq_status = (
                "季增強勁" if qoq_value >= 10 else
                "季增穩定" if qoq_value >= 0 else
                "季減輕微" if qoq_value >= -10 else
                "季減明顯"
            )

            # === 5. 綜合評估 ===
            overall = (
                "營運展望樂觀" if (
                    yoy_value > 0 and
                    ma3_ma12_ratio > 0 and
                    pr_value > 60 and
                    qoq_status in ["季增強勁", "季增穩定"] and
                    trend_status in ["上升趨勢強勁", "呈現上升趨勢"]
                ) else
                "營運展望保守" if (
                    yoy_value < 0 and
                    ma3_ma12_ratio < 0 and
                    pr_value < 40 and
                    qoq_status in ["季減輕微", "季減明顯"] and
                    trend_status in ["趨勢略為下滑", "下降趨勢明顯"]
                ) else
                "營運表現平穩"
            )

            # 決定顏色
            def get_color(value, threshold=0):
                return '#FF4444' if value >= threshold else '#00CC00'

            yoy_color = get_color(yoy_value)
            mom_color = get_color(mom_value)
            qoq_color = get_color(qoq_value)
            ma_ratio_color = get_color(ma3_ma12_ratio)

            # PR值顏色
            pr_color = (
                '#FF0000' if pr_value >= 80 else
                '#FF4444' if pr_value >= 60 else
                'black' if pr_value >= 40 else
                '#00AA00' if pr_value >= 20 else
                '#006600'
            )

            # 添加趨勢符號
            def get_trend_symbol(value):
                return '↑' if value >= 0 else '↓'

            yoy_symbol = get_trend_symbol(yoy_value)
            mom_symbol = get_trend_symbol(mom_value)
            qoq_symbol = get_trend_symbol(qoq_value)

            # 格式化營收數字
            def format_revenue(value):
                try:
                    if pd.isna(value):
                        return "N/A"
                    return format(int(value/1000), ',')
                except (ValueError, TypeError) as e:
                    logger.error(f"格式化營收時發生錯誤: {str(e)}")
                    return "N/A"

            formatted_rev = format_revenue(latest_rev)

            # 格式化各項指標文字
            def format_indicator(name, value, symbol):
                return f'{name}: {value:+.1f}%{symbol}'

            yoy_text = format_indicator('年增率', yoy_value, yoy_symbol)
            mom_text = format_indicator('月增率', mom_value, mom_symbol)
            qoq_text = format_indicator('季增率', qoq_value, qoq_symbol)
            ma_ratio_text = f'3MA/12MA: {ma3_ma12_ratio:+.1f}%'
            pr_text = f'PR值: {pr_value:.0f} ({pr_status})'

            # 獲取股票資訊
            try:
                stock_info = self.stock_topics_df[self.stock_topics_df['stock_no'] == int(stock_id)]
                
                if not stock_info.empty:
                    stock_name = stock_info["stock_name"].values[0]
                    topic = stock_info["topic"].values[0] if "topic" in stock_info.columns else "無"

                    main_title = f'{stock_name}({stock_id})'
                    subtitle = f'{topic} | {overall} {new_high_text}'
                    revenue_line = f'{month_prefix}{revenue_month}營收 {formatted_rev}百萬元 | {revenue_status}'
                else:
                    logger.warning(f"找不到股票 {stock_id} 的資訊")
                    main_title = f'{stock_id}'
                    subtitle = f'{overall}'
                    revenue_line = f'{month_prefix}{revenue_month}營收 {formatted_rev}百萬元 | {revenue_status}'
            except Exception as e:
                logger.error(f"處理股票資訊時發生錯誤: {str(e)}")
                main_title = f'{stock_id}'
                subtitle = f'{overall}'
                revenue_line = f'{month_prefix}{revenue_month}營收 {formatted_rev}百萬元 | {revenue_status}'

            return {
                'main_title': main_title,
                'subtitle': subtitle,
                'revenue_line': revenue_line,
                'yoy_text': yoy_text,
                'mom_text': mom_text,
                'qoq_text': qoq_text,
                'ma_ratio_text': ma_ratio_text,
                'pr_text': pr_text,
                'yoy_color': yoy_color,
                'mom_color': mom_color,
                'qoq_color': qoq_color,
                'ma_ratio_color': ma_ratio_color,
                'pr_color': pr_color,
                'yoy_value': yoy_value,
                'mom_value': mom_value,
                'qoq_value': qoq_value,
                'ma_ratio_value': ma3_ma12_ratio,
                'pr_value': pr_value,
                'yoy_symbol': yoy_symbol,
                'mom_symbol': mom_symbol,
                'qoq_symbol': qoq_symbol,
                'is_new_high': is_new_high,
                'new_high_text': new_high_text
            }

        except Exception as e:
            logger.error(f"生成標題時發生錯誤: {str(e)}")
            logger.exception("詳細錯誤資訊：")
            return self._get_default_title_dict(stock_id)


    def _get_default_title_dict(self, stock_id):
        """
        返回預設的標題字典
        """
        return {
            'main_title': f'{stock_id}',
            'subtitle': '月營收走勢圖',
            'revenue_line': 'N/A',
            'yoy_text': 'N/A',
            'mom_text': 'N/A',
            'qoq_text': 'N/A',
            'ma_ratio_text': 'N/A',
            'pr_text': 'N/A',
            'yoy_color': 'black',
            'mom_color': 'black',
            'qoq_color': 'black',
            'ma_ratio_color': 'black',
            'pr_color': 'black',
            'yoy_value': 0,
            'mom_value': 0,
            'qoq_value': 0,
            'ma_ratio_value': 0,
            'pr_value': 50,
            'yoy_symbol': '',
            'mom_symbol': '',
            'qoq_symbol': '',
            'is_new_high': False,
            'new_high_text': ''
        }


    def _add_yoy_labels(self, ax, bars, yoy, rev):
        """
        添加年增率標籤
        """
        try:
            # 獲取技術指標
            indicators = self._calculate_technical_indicators(rev)
            
            # 安全地獲取 growth 值
            def safe_get_value(data_dict, key, idx=-1, default=False):
                try:
                    if key not in data_dict:
                        return default
                    value = data_dict[key]
                    if isinstance(value, (pd.Series, np.ndarray)):
                        return value.iloc[idx] if isinstance(value, pd.Series) else value[idx]
                    return value
                except Exception as e:
                    logger.error(f"獲取 {key} 數據時發生錯誤: {str(e)}")
                    return default

            # 獲取 growth 數據
            growth = indicators.get('growth', pd.Series([False] * len(rev)))
            
            for i, bar in enumerate(bars[-4:]):
                idx = i - 4
                if not pd.isna(yoy.iloc[idx]):
                    height = bar.get_height()
                    # 使用與柱狀圖相同的邏輯
                    is_growing = safe_get_value({'growth': growth}, 'growth', idx)
                    text_color = '#FF4444' if (yoy.iloc[idx] > 0 and is_growing) else '#00CC00'

                    ax.text(bar.get_x() + bar.get_width()/2.,
                        height * 1.02,
                        f'{yoy.iloc[idx]:.1f}%',
                        ha='center',
                        va='bottom',
                        fontsize=12,
                        color=text_color,
                        fontweight='bold',
                        rotation=45,
                        zorder=5)

        except Exception as e:
            logger.error(f"添加年增率標籤時發生錯誤: {str(e)}")


    def _set_axes_format(self, ax1, ax2):
        """設置軸的格式"""
        # 設置軸標籤
        ax1.set_ylabel('股\n價\n (元)', fontsize=15, rotation=0, labelpad=5)
        # ax2.set_ylabel('營\n收\n 百\n萬\n元', fontsize=15, rotation=0, labelpad=95)

        # 調整Y軸位置
        ax1.yaxis.set_label_coords(-0.08, 0.5)
        # ax2.yaxis.set_label_coords(1.1, 0.75)

        # 設置x軸格式
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 設置刻度字體大小
        ax1.tick_params(axis='both', labelsize=14)
        ax2.tick_params(axis='y', labelsize=14)

        # 設置Y軸格式
        ax1.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        # ax2.yaxis.set_major_formatter(
        #     ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    def _set_legend(self, ax, line_monthly_close, line3ma, line12ma, line24ma):
        """設置圖例"""
        try:
            if all([line_monthly_close, line3ma, line12ma, line24ma]):
                # 修改圖例，加入更多顏色說明
                legend_handles = [
                    line_monthly_close,
                    line3ma,
                    line12ma,
                    line24ma,
                    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='k', markersize=15,label='創新高')
                ]

                # 創建圖例
                legend = ax.legend(
                    handles=legend_handles,
                    labels=[
             
                        '月均價',
                        '3月均線',
                        '12月均線',
                        '24月均線',
                        '創新高'

                        
                    ],
                    loc='upper left',
                    bbox_to_anchor=(0.02, -0.19),
                    ncol=5,
                    fontsize=20,
                    frameon=True,
                    edgecolor='gray'
                )

                legend.get_frame().set_alpha(0.9)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_linewidth(1)
                # 調整圖例的間距
                plt.tight_layout()

        except Exception as e:
            logger.error(f"設置圖例時發生錯誤: {str(e)}")
            logger.exception("詳細錯誤信息：")


# ============================================集保分析================================================================================================

    def calculate_period_pr(self, ratio_diff, ratio_data, stock_id, date):
        """
        計算特定期間的PR值
        """
        try:
            current_ratio = ratio_data.loc[date, stock_id]
            diff = ratio_diff.loc[date, stock_id]

            if pd.isna(diff) or pd.isna(current_ratio):
                return 0

            # 建議修改：加入安全檢查和更精確的計算
            if abs(current_ratio) < 0.0001:  # 避免除以接近0的值
                return 0
                
            pr = int(diff * 100 / current_ratio)  # 移除+0.0001，改用上面的檢查
            return pr

        except Exception as e:
            print(f"計算PR值時發生錯誤: {str(e)}")
            return 0

    def get_pr_strength(self, pr_value):
        """
        根據PR值判斷強度，使用紅色系表示多頭、綠色系表示空頭
        """
        if pr_value > 5:
            return "極強", "#FF0000"  # 純紅
        elif pr_value > 3:
            return "強", "#FF3333"    # 較亮紅
        elif pr_value > 1:
            return "偏強", "#FF6666"  # 淺紅
        elif pr_value < -5:
            return "極弱", "#006400"  # 深綠
        elif pr_value < -3:
            return "弱", "#228B22"    # 較深綠
        elif pr_value < -1:
            return "偏弱", "#90EE90"  # 淺綠
        else:
            return "持平", "#808080"  # 灰色

    def calculate_statistics(self, ratio_data, weekly_change):
        """
        計算統計指標
        """
        # 使用最近8週的數據
        data_8w = ratio_data[-8:]
        changes_8w = weekly_change[-8:]

        # 趨勢分析（線性回歸）
        x = np.arange(len(data_8w))
        slope, intercept = np.polyfit(x, data_8w.values * 100, 1)
        y_pred = slope * x + intercept

        # 趨勢可信度（R平方值）
        correlation_matrix = np.corrcoef(data_8w.values * 100, y_pred)
        r2 = correlation_matrix[0, 1]**2

        # 動能分析
        momentum = (data_8w.iloc[-1] - data_8w.iloc[0]) * 100

        return {
            'slope': slope,
            'r2': r2,
            'momentum': momentum
        }

    def calculate_trend_analysis(self, data_8w, weekly_change):
        """
        進行更全面的走勢分析
        """
        # 加入數據有效性檢查
        if len(data_8w) < 2:
            raise ValueError("數據點數不足")
        
        # 標準化數據以改善分析
        normalized_data = (data_8w - data_8w.mean()) / data_8w.std()
        
        x = np.arange(len(data_8w))
        
        # 使用更穩健的回歸方法
        try:
            slope, intercept = np.polyfit(x, data_8w.values * 100, 1)
            y_pred = slope * x + intercept
            
            # 使用Pearson相關係數
            correlation = np.corrcoef(data_8w.values * 100, y_pred)[0,1]
            r2 = correlation**2
        except Exception:
            slope, r2 = 0, 0
        
        # 加入更多統計指標
        return {
            'slope': slope,
            'r2': r2,
            'momentum': (data_8w.iloc[-1] - data_8w.iloc[0]) * 100,
            'volatility': weekly_change.std(),
            'recent_trend': weekly_change.tail(4).mean(),
            'is_breakthrough_up': data_8w.iloc[-1] > (data_8w.mean() + data_8w.std() * 1.5),
            'is_breakthrough_down': data_8w.iloc[-1] < (data_8w.mean() - data_8w.std() * 1.5),
            'trend_consistency': np.sum(np.diff(data_8w) > 0) / (len(data_8w) - 1)  # 趨勢一致性
        }


    def get_trend_description(self, trend_analysis):
        """
        根據走勢分析結果生成詳細描述
        """
        # 1. 趨勢方向判斷
        if trend_analysis['slope'] > 0.3 and trend_analysis['r2'] > 0.6:
            trend = "強勢上升"
        elif trend_analysis['slope'] > 0.1 and trend_analysis['r2'] > 0.5:
            trend = "緩步上升"
        elif trend_analysis['slope'] < -0.3 and trend_analysis['r2'] > 0.6:
            trend = "強勢下降"
        elif trend_analysis['slope'] < -0.1 and trend_analysis['r2'] > 0.5:
            trend = "緩步下降"
        else:
            trend = "橫盤整理"

        # 2. 動能強度判斷
        if trend_analysis['momentum'] > 5 and trend_analysis['recent_trend'] > 0:
            momentum = "強勢上漲"
        elif trend_analysis['momentum'] > 2:
            momentum = "溫和上漲"
        elif trend_analysis['momentum'] < -5 and trend_analysis['recent_trend'] < 0:
            momentum = "強勢下跌"
        elif trend_analysis['momentum'] < -2:
            momentum = "溫和下跌"
        else:
            momentum = "盤整"

        # 3. 突破判斷
        breakthrough = ""
        if trend_analysis['is_breakthrough_up']:
            breakthrough = "｜突破上軌"
        elif trend_analysis['is_breakthrough_down']:
            breakthrough = "｜突破下軌"

        # 4. 波動特徵
        volatility_str = ("高波動" if trend_analysis['volatility'] > 2 else
                          "中等波動" if trend_analysis['volatility'] > 1 else
                          "低波動")

        return (f"走勢:{trend}({trend_analysis['r2']:.2f})|"
                f"動能:{momentum}|"
                f"波動:{volatility_str}"
                f"{breakthrough}")

    def get_overall_pr_score(self, pr_values, latest_super_ratio):
        """
        計算綜合評分
        """
        # 權重設定
        weights = {1: 0.5, 3: 0.3, 6: 0.2}

        # 計算中實戶PR值的加權分數
        mid_score = sum(pr_values['mid'][period] * weights[period]
                        for period in weights.keys())

        # 計算超大戶PR值的加權分數
        super_score = sum(pr_values['super'][period] * weights[period]
                          for period in weights.keys())

        # 綜合評分（可以根據需求調整計算方式）
        overall_score = mid_score * 0.7 + super_score * 0.3 + latest_super_ratio * 0.1

        # 評分等級判定
        if overall_score > 8:
            return "ALL_IN買起來", "darkred", overall_score
        elif overall_score > 5:
            return "優", "red", overall_score
        elif overall_score > 2:
            return "佳", "orange", overall_score
        elif overall_score < -8:
            return "極差", "darkgreen", overall_score
        elif overall_score < -5:
            return "差", "green", overall_score
        elif overall_score < -2:
            return "弱", "lightgreen", overall_score
        else:
            return "中性", "gray", overall_score
        
        
        
        # ==========================================畫出中實戶=========================
        
    def plot_holding_ratio(self, ax, stock_id):
        """
        繪製持股比例分析圖，包含8週統計分析
        
        Parameters:
            ax: matplotlib axes object, 繪圖用的軸物件
            stock_id: str, 股票代碼
        """
        try:
            # 1. 獲取股票名稱
            stock_info = self.stock_topics_df[self.stock_topics_df['stock_no'] == int(stock_id)]
            stock_name = stock_info['stock_name'].values[0] if not stock_info.empty else stock_id

            # 2. 獲取持股資料
            inv = self.data.get('inventory')
            if inv is None:
                raise ValueError("無法獲取持股資料")
            
            # 檢查缺失值並填補
            if inv.isnull().any().any():
                print("警告: 持股資料中存在缺失值，將進行填補")
                inv.fillna(0, inplace=True)

            # 確保日期格式一致
            inv['date'] = pd.to_datetime(inv['date'], errors='coerce')
            if inv['date'].isnull().any():
                raise ValueError("持股資料中存在無效的日期格式")
            
            
            # ======================10到15張以下============================


            # 3. 計算一般大戶持股比例
            small_holders = FinlabDataFrame(
                inv[inv.持股分級.astype(int) <= 4]
                .reset_index()
                .groupby(['date', 'stock_id'], observed=True)
                .agg({'持有股數': 'sum'})
                .reset_index()
                .pivot(index='date', columns='stock_id', values='持有股數')
            )
            

            # ==================200到800張大戶==================================
            

            large_holders = FinlabDataFrame(
                inv[(inv.持股分級.astype(int) >= 11) &
                    (inv.持股分級.astype(int) <= 14)]
                .reset_index()
                .groupby(['date', 'stock_id'], observed=True)
                .agg({'持有股數': 'sum'})
                .reset_index()
                .pivot(index='date', columns='stock_id', values='持有股數')
            )
            
            
            # ==================400到1000張大戶持股變動============================
            
            mid_range_holders = FinlabDataFrame(
                inv[(inv.持股分級.astype(int) >= 12) & 
                    (inv.持股分級.astype(int) <= 15)]
                .reset_index()
                .groupby(['date', 'stock_id'], observed=True)
                .agg({'持有股數': 'sum'})
                .reset_index()
                .pivot(index='date', columns='stock_id', values='持有股數')
            )

            # =======================大於1000張==============================


            # 4. 計算超大戶持股比例
            super_large_holders = FinlabDataFrame(
                inv[(inv.持股分級.astype(int) == 15) ]
                .reset_index()
                .groupby(['date', 'stock_id'], observed=True)
                .agg({'持有股數': 'sum'})
                .reset_index()
                .pivot(index='date', columns='stock_id', values='持有股數')
            )

            # ============================全部持股=================================

            all_holders = FinlabDataFrame(
                inv[(inv.持股分級.astype(int) == 17)]
                .reset_index()
                .groupby(['date', 'stock_id'],observed=True)
                .agg({'持有股數': 'sum'})
                .reset_index()
                .pivot(index='date', columns='stock_id', values='持有股數')
            )

            if stock_id not in small_holders.columns or stock_id not in large_holders.columns:
                raise ValueError(f"無法獲取 {stock_id} 的持股資料")

            # 5. 計算持股比例
            total_shares = small_holders + large_holders
            holding_ratio = (large_holders / total_shares).fillna(0)
            super_large_ratio = (super_large_holders / all_holders).fillna(0)
            mid_small_ratio = (mid_range_holders / (mid_range_holders + small_holders)).fillna(0)

            # 6. 計算PR值和移動平均
            periods = [1, 3, 6]
            pr_values = {'mid': {}, 'super': {}}
            
            for period in periods:
                ratio_diff = holding_ratio.diff(period)
                latest_date = ratio_diff.index[-1]
                pr_values['mid'][period] = self.calculate_period_pr(
                    ratio_diff, 
                    holding_ratio,
                    stock_id, 
                    latest_date
                )

                super_ratio_diff = super_large_ratio.diff(period)
                super_latest_date = super_ratio_diff.index[-1]
                pr_values['super'][period] = self.calculate_period_pr(
                    super_ratio_diff,
                    super_large_ratio,
                    stock_id, 
                    super_latest_date
                )

            # 7. 統一日期格式和數據處理
            holding_ratio.index = pd.to_datetime(holding_ratio.index)
            super_large_ratio.index = pd.to_datetime(super_large_ratio.index)
            mid_small_ratio.index = pd.to_datetime(mid_small_ratio.index)

            # 8. 過濾最近8周數據
            end_date = holding_ratio.index[-1]
            start_date = end_date - pd.Timedelta(weeks=8)
            mask = (holding_ratio.index >= start_date) & (holding_ratio.index <= end_date)
            ratio_data = holding_ratio.loc[mask, stock_id].copy()
            mid_small_data = mid_small_ratio.loc[mask, stock_id].copy()

            if ratio_data.empty:
                raise ValueError("過濾後的數據為空")

            # 9. 計算技術指標
            weekly_change = ratio_data.diff() * 100
            mid_small_change = mid_small_data.diff() * 100

            # 10. 獲取和處理收盤價數據
            close_data = self.data.get("price:收盤價")[stock_id]
            close_data.index = pd.to_datetime(close_data.index)
            ratio_dates = ratio_data.index
            close_prices = close_data.reindex(ratio_dates, method='ffill')

            # 11. 繪製主圖表
            ax2 = ax.twinx()

            # 計算柱狀圖的寬度和位置
            bar_width = 2
            
            # 計算散戶持股比例
            small_holders_ratio = (small_holders / all_holders).fillna(0)
            small_holders_data = small_holders_ratio.loc[mask, stock_id].copy()

            # 繪製兩組柱狀圖
            bars1 = ax.bar(ratio_data.index - pd.Timedelta(days=bar_width/2), 
                        ratio_data.values * 100,
                        width=bar_width, 
                        alpha=0.6, 
                        label='中實戶',
                        edgecolor='black',
                        linewidth=1.5)

            bars2 = ax.bar(ratio_data.index + pd.Timedelta(days=bar_width/2), 
                        mid_small_data.values * 100,
                        width=bar_width, 
                        alpha=0.6, 
                        label='大戶',
                        edgecolor='black',
                        linewidth=1.5)
            
            small_line = ax.plot(small_holders_data.index,
                                small_holders_data.values * 100,
                                color='#800080',          # 深紫色
                                linestyle='--',            # 改為實線
                                marker='o',               # 加入圓形標記
                                markersize=7,             # 標記大小
                                markerfacecolor='white',  # 標記內部填充為白色
                                markeredgewidth=1,        # 標記邊框寬度
                                linewidth=2,              # 線條寬度
                                label='韭菜持股',
                                zorder=6)

            # 繪製收盤價
            line = ax2.plot(close_prices.index, close_prices.values, 
                        color='blue', linewidth=2, linestyle=':', 
                        marker='D', label='收盤價', zorder=7)

            # 12. 設置柱狀圖顏色和標籤
            for i, (date, value) in enumerate(weekly_change.items()):
                try:
                    current_ratio = ratio_data.values[i] * 100
                    current_mid_small = mid_small_data.values[i] * 100
                    current_mid_small_change = mid_small_change.iloc[i]
                    
                    # 第一組柱狀圖（中實戶）
                    if pd.isna(current_ratio):
                        color1, alpha1 = 'gray', 0.3
                    else:
                        # 根據持股比例設定透明度
                        base_alpha = 0.4  # 基礎透明度
                        ratio_max = max(ratio_data.values) * 100  # 最大持股比例
                        ratio_min = min(ratio_data.values) * 100  # 最小持股比例
                        
                        # 將當前值標準化到 0.4-0.9 的範圍
                        alpha1 = base_alpha + (current_ratio - ratio_min) / (ratio_max - ratio_min) * 0.5
                        alpha1 = min(max(alpha1, 0.4), 0.9)  # 確保透明度在合理範圍內
                        
                        # 顏色仍然根據變化方向決定
                        color1 = '#FF4D4D' if value > 0 else '#4DAF4A'
                    
                    bars1[i].set_color(color1)
                    bars1[i].set_alpha(alpha1)
                    
                    # 第二組柱狀圖（中實戶II）
                    if pd.isna(current_mid_small):
                        color2, alpha2 = 'gray', 0.3
                    else:
                        # 類似的方法計算第二組透明度
                        mid_small_max = max(mid_small_data.values) * 100
                        mid_small_min = min(mid_small_data.values) * 100
                        
                        alpha2 = base_alpha + (current_mid_small - mid_small_min) / (mid_small_max - mid_small_min) * 0.5
                        alpha2 = min(max(alpha2, 0.4), 0.9)
                        
                        color2 = '#FF4D4D' if current_mid_small_change > 0 else '#4DAF4A'
                    
                    # 顏色記住:#FF9999  #90EE90
                    
                    bars2[i].set_color(color2)
                    bars2[i].set_alpha(alpha2)
                    
                    # 標籤顯示
                    if not pd.isna(current_ratio):
                        # 第一組標籤
                        ax.text(date - pd.Timedelta(days=bar_width/2), 
                            current_ratio + 0.5,
                            f'{current_ratio:.1f}%',
                            ha='center', va='bottom',
                            color='black', fontsize=8)
                        
                        # 第二組標籤
                        ax.text(date + pd.Timedelta(days=bar_width/2), 
                            current_mid_small + 0.5,
                            f'{current_mid_small:.1f}%',
                            ha='center', va='bottom',
                            color='black', fontsize=8)
                        
                        # 變動標籤
                        if abs(value) >= 0.3:
                            ax.text(date - pd.Timedelta(days=bar_width/2), 
                                current_ratio - 1.5,
                                f'{value:+.1f}%',
                                ha='center', va='top',
                                color='darkred' if value > 0 else 'darkgreen', 
                                fontsize=8)
                        
                        if abs(current_mid_small_change) >= 0.3:
                            ax.text(date + pd.Timedelta(days=bar_width/2), 
                                current_mid_small - 1.5,
                                f'{current_mid_small_change:+.1f}%',
                                ha='center', va='top',
                                color='darkred' if current_mid_small_change > 0 else 'darkgreen', 
                                fontsize=8)

                except Exception as e:
                    print(f"處理第 {i} 個柱子時發生錯誤: {str(e)}")
                    continue

            # 13. 計算分析指標
            stats = self.calculate_statistics(ratio_data, weekly_change)
            trend_analysis = self.calculate_trend_analysis(ratio_data, weekly_change)
            trend_text = self.get_trend_description(trend_analysis)

            # 14. 計算評分和強度
            latest_super_ratio = super_large_ratio.loc[end_date, stock_id] * 100
            score_text, score_color, overall_score = self.get_overall_pr_score(
                pr_values, latest_super_ratio)
            latest_pr = pr_values['mid'][1]
            pr_strength_text, pr_color = self.get_pr_strength(latest_pr)

            # 15. 設置圖表標題和標籤
            ax.set_title('')
            
            # 設置多行標題
            title_lines = [
                f"評等: {score_text} ",
                f"{stock_id} {stock_name} 中實戶持股比例變化",
                f"千張戶比例: {latest_super_ratio:.1f}% | 綜合評分: {overall_score:.1f}",
                f"強度({latest_pr:+d}): {pr_strength_text} |{trend_text}"
            ]
            
            # 計算每行的y位置
            y_positions = [1.35, 1.25, 1.15, 1.05]
            
            # 分別設置每行標題
            for i, (y_pos, title_line) in enumerate(zip(y_positions, title_lines)):
                color = score_color if i == 0 else ('black' if i == 1 else pr_color)
                ax.text(0.5, y_pos, title_line, 
                    horizontalalignment='center',
                    transform=ax.transAxes,
                    fontsize=30,
                    color=color)

            # 16. 設置軸標籤和格式
            ax.set_xlabel('')
            ax.set_ylabel('持\n股\n比\n例\n(%)', rotation=0, labelpad=20, fontsize=18)
            ax2.set_ylabel('股\n價', rotation=0, labelpad=20, fontsize=18)

            # 設置圖例在子圖外面右側
            # 獲取所有圖例元素
            legend_elements = [
                bars1.patches[0],  # 中實戶變動
                bars2.patches[0],  # 中實戶II變動
                small_line[0],        # 散戶持股
                line[0]           # 收盤價
            ]
            legend_labels = ['中實戶變動', '大戶變動','韭菜持股','收盤價']

            # 將圖例放在子圖外面右側
            ax.legend(legend_elements, 
                    legend_labels,
                    loc='center left',
                    bbox_to_anchor=(1.15, 0.5),
                    frameon=True,
                    fontsize=18)

            # 調整布局以確保圖例不會被裁剪
            plt.subplots_adjust(right=0.85)  # 留出右側空間給圖例

            # 設置網格
            ax.grid(True, alpha=0.3)
            
            # 設置x軸日期格式
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=20)

            # 調整圖表佈局
            plt.tight_layout()

        except Exception as e:
            error_msg = f"繪製股票 {stock_id} 的持股比例圖表時發生錯誤: {str(e)}"
            print(error_msg)
            ax.text(0.5, 0.5, error_msg, 
                    ha='center', va='center',
                    transform=ax.transAxes)






# =================================三大法人=================================================


    def plot_institutional_investors(self, ax, stock_id):
        """
        繪製三大法人買賣超分析圖，包含PR值變化和增強的評分系統
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            繪圖用的axes物件
        stock_id : str
            股票代碼
            
        Returns:
        --------
        bool
            繪圖成功返回True，失敗返回False
        """
        try:
            # 1. 數據獲取和預處理 (保持不變)
            all_big_player1 = data.get(
                'institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)')
            all_big_player1 += data.get(
                'institutional_investors_trading_summary:外資自營商買賣超股數')
            all_big_player2 = data.get(
                'institutional_investors_trading_summary:投信買賣超股數')
            all_big_player3 = data.get(
                'institutional_investors_trading_summary:自營商買賣超股數(自行買賣)')
            all_volume = data.get('price:成交股數')
            all_close = data.get('price:收盤價')
            all_cap = data.get('etl:market_value')  # 獲取市值數據

            # 檢查股票是否存在於資料中
            if stock_id not in all_big_player1.columns or stock_id not in all_volume.columns:
                raise ValueError(f"無法獲取 {stock_id} 的三大法人資料或成交量資料")

            # 確保所有數據使用相同的日期索引
            common_dates = (all_big_player1.index
              .intersection(all_big_player2.index)
              .intersection(all_big_player3.index)
              .intersection(all_volume.index)
              .intersection(all_close.index)
              .intersection(all_cap.index))

            if len(common_dates) == 0:
                raise ValueError("沒有共同的交易日期")

            # 使用共同日期重新索引所有數據
            days = min(60, len(common_dates))
            common_dates = common_dates[-days:]

            # 2. 提取並處理數據 (保持不變)
            big_player1 = all_big_player1.loc[common_dates, stock_id]
            big_player2 = all_big_player2.loc[common_dates, stock_id]
            big_player3 = all_big_player3.loc[common_dates, stock_id]
            volume = all_volume.loc[common_dates, stock_id]
            close = all_close.loc[common_dates, stock_id]
            cap = all_cap.loc[common_dates, stock_id]
            cap = cap.fillna(method='ffill')  # 處理缺失值


            # 處理缺失值
            big_player1 = big_player1.fillna(0)
            big_player2 = big_player2.fillna(0)
            big_player3 = big_player3.fillna(0)
            volume = volume.fillna(0)
            close = close.fillna(method='ffill')
          
            big_player1_ratio = (big_player1 * close) / cap
            big_player2_ratio = (big_player2 * close) / cap
            big_player3_ratio = (big_player3 * close) / cap

            # 填補可能的 NaN 值
            big_player1_ratio = big_player1_ratio.fillna(0)
            big_player2_ratio = big_player2_ratio.fillna(0)
            big_player3_ratio = big_player3_ratio.fillna(0)

            # 4. 計算連續買賣天數
            def calculate_consecutive_days(ratios):
                current_streak = 0
                for ratio in ratios[::-1]:
                    if (current_streak >= 0 and ratio > 0) or (current_streak <= 0 and ratio < 0):
                        current_streak = current_streak + 1 if ratio > 0 else current_streak - 1
                    else:
                        break
                return current_streak

            foreign_consecutive_days = calculate_consecutive_days(big_player1_ratio)
            investment_consecutive_days = calculate_consecutive_days(big_player2_ratio)

            # 5. 計算20日平均比例
            last_20_days_fi_ratio = big_player1_ratio.tail(20).mean()
            last_20_days_sit_ratio = big_player2_ratio.tail(20).mean()

            # 6. 計算市場整體法人動向
            market_trends = pd.DataFrame()
            valid_stocks = set(all_volume.columns) & set(all_big_player1.columns)

            for stock in valid_stocks:
                stock_volume = all_volume[stock].loc[common_dates]
                if stock_volume.sum() > 0:
                    foreign_ratio = all_big_player1[stock].loc[common_dates] / stock_volume
                    sit_ratio = all_big_player2[stock].loc[common_dates] / stock_volume
                    dealer_ratio = all_big_player3[stock].loc[common_dates] / stock_volume
                    market_trends[stock] = foreign_ratio + sit_ratio + dealer_ratio

            # 計算市場平均
            market_avg = market_trends.mean(axis=1)
            market_stats = pd.DataFrame({
                '市場平均買超比例': market_avg,
                '市場20日累積': market_avg.rolling(20).sum(),
                '市場5日累積': market_avg.rolling(5).sum()
            })
            market_stats = market_stats.loc[common_dates]
            
            # 7. 計算累積買賣超比例
            total_buy_sell_ratio = big_player1_ratio + big_player2_ratio + big_player3_ratio
            cumsum_total = total_buy_sell_ratio.cumsum()
            cumsum_foreign = big_player1_ratio.cumsum()
            cumsum_investment = big_player2_ratio.cumsum()

            # 計算每日買賣超比例總和
            daily_total_ratio = big_player1_ratio + big_player2_ratio + big_player3_ratio

            # 準備繪圖用的數據字典
            data_dict = {
                'cumsum_total': cumsum_total.values,
                'cumsum_investment': cumsum_investment.values,
                'cumsum_foreign': cumsum_foreign.values,
                'daily_total_ratio': daily_total_ratio.values,
                'big_player1_ratio': big_player1_ratio.values,
                'big_player2_ratio': big_player2_ratio.values
            }

            # 8. 計算評分指標 (更新部分)
            # 新增: PR值計算函數
            def calculate_pr_scores(total_buy_sell_ratio, market_stats, window):
                """計算PR值"""
                stock_sum = total_buy_sell_ratio.rolling(window).sum()
                market_sum = market_stats[f'市場{window}日累積']
                relative_strength = stock_sum - market_sum
                return stats.percentileofscore(relative_strength.dropna(), relative_strength.iloc[-1])

            # 計算PR值
            pr_5d = calculate_pr_scores(total_buy_sell_ratio, market_stats, 5)
            pr_20d = calculate_pr_scores(total_buy_sell_ratio, market_stats, 20)
            pr_change = pr_5d - pr_20d

            # 相對強度評分
            stock_20d_ratio = total_buy_sell_ratio.rolling(20).sum()
            market_20d_ratio = market_stats['市場20日累積']
            relative_strength = stock_20d_ratio - market_20d_ratio
            relative_percentile = stats.percentileofscore(
                relative_strength.dropna(), relative_strength.iloc[-1])

            # 法人一致性評分
            def calculate_consistency_score(foreign_ratio, investment_ratio):
                recent_days = 5
                foreign_direction = np.sign(foreign_ratio.tail(recent_days))
                investment_direction = np.sign(investment_ratio.tail(recent_days))
                same_direction = (foreign_direction == investment_direction) & \
                            (foreign_direction != 0) & \
                            (investment_direction != 0)
                return same_direction.mean() * 100

            consistency_score = calculate_consistency_score(
                big_player1_ratio, big_player2_ratio)

            # 買賣動能評分
            def calculate_momentum_score(ratios):
                recent_5d = ratios.tail(5).mean()
                recent_20d = ratios.tail(20).mean()
                if recent_20d == 0:
                    return 50
                momentum_change = (recent_5d / recent_20d - 1)
                if momentum_change > 0.2:
                    return 100
                elif momentum_change > 0.1:
                    return 80
                elif momentum_change > 0:
                    return 60
                elif momentum_change > -0.1:
                    return 40
                elif momentum_change > -0.2:
                    return 20
                else:
                    return 0

            momentum_score = np.mean([
                calculate_momentum_score(big_player1_ratio),
                calculate_momentum_score(big_player2_ratio)
            ])

            # 市場氛圍評分
            def calculate_market_score(market_stats):
                recent_5d = market_stats['市場5日累積'].iloc[-1]
                recent_20d = market_stats['市場20日累積'].iloc[-1]
                if recent_5d > 0 and recent_20d > 0:
                    return 100
                elif recent_5d > 0 and recent_20d <= 0:
                    return 75
                elif recent_5d <= 0 and recent_20d > 0:
                    return 25
                else:
                    return 0

            market_score = calculate_market_score(market_stats)
            relative_score = relative_percentile

            # 更新權重配置
            weights = {
                'relative': 0.3,     # 降低原相對強度權重
                'consistency': 0.2,
                'momentum': 0.2,
                'market': 0.2,
                'pr_value': 0.1      # 新增PR值權重
            }

            # 計算PR值評分
            pr_score = (pr_5d + pr_20d) / 2

            # 更新最終評分計算
            final_score = (
                relative_score * weights['relative'] +
                consistency_score * weights['consistency'] +
                momentum_score * weights['momentum'] +
                market_score * weights['market'] +
                pr_score * weights['pr_value']
            )

            # 9. 評級判定 (更新部分)
            def format_pr_change(change):
                """格式化PR值變化的顯示"""
                if change > 0:
                    return f"↑{change:.1f}%", "red"
                elif change < 0:
                    return f"↓{abs(change):.1f}%", "green"
                return f"{change:.1f}%", "black"

            def get_final_rating(score, pr_change):
                """整合PR值變化的評級判定"""
                if score >= 80:
                    rating = "強勢買入"
                    color = "red"
                    description = "法人持續買超,動能強勁"
                elif score >= 60:
                    rating = "建議買入"
                    color = "indianred"
                    description = "法人買超,趨勢向上"
                elif score >= 40:
                    rating = "中性持平"
                    color = "black"
                    description = "法人動向觀望"
                elif score >= 20:
                    rating = "建議賣出"
                    color = "green"
                    description = "法人賣超,趨勢向下"
                else:
                    rating = "強勢賣出"
                    color = "darkgreen"
                    description = "法人持續賣超,動能疲弱"

                # PR值變化影響評級描述
                if abs(pr_change) > 20:
                    if pr_change > 0:
                        description += " | PR值大幅上升"
                    else:
                        description += " | PR值大幅下降"
                elif abs(pr_change) > 10:
                    if pr_change > 0:
                        description += " | PR值上升"
                    else:
                        description += " | PR值下降"

                return rating, color, description

            def get_strength_description(percentile):
                if percentile >= 80:
                    return "非常強勢", "red", "法人持續買超,強勢股"
                elif percentile >= 60:
                    return "相對強勢", "indianred", "法人買超,趨勢向上"
                elif percentile >= 40:
                    return "持平", "black", "法人動向觀望"
                elif percentile >= 20:
                    return "相對弱勢", "green", "法人賣超,趨勢向下"
                else:
                    return "非常弱勢", "darkgreen", "法人持續賣超,弱勢股"

            rating, color, description = get_final_rating(final_score, pr_change)
            strength, strength_color, strength_desc = get_strength_description(
                relative_percentile)

            # 10. 繪製圖表 (保持原有繪圖邏輯)
            def plot_with_scaled_axes(ax, common_dates, data_dict):
                """使用調整後的比例尺繪製圖表"""
                # 創建雙軸
                ax1 = ax
                ax2 = ax.twinx()

                # 計算累積數據的合適比例範圍
                cumsum_data = pd.DataFrame({
                    'total': data_dict['cumsum_total'],
                    'investment': data_dict['cumsum_investment'],
                    'foreign': data_dict['cumsum_foreign']
                })
                
                y1_min = cumsum_data.min().min()
                y1_max = cumsum_data.max().max()
                y1_range = y1_max - y1_min
                y1_padding = y1_range * 0.1
                
                # 設置左側Y軸範圍
                ax1.set_ylim(y1_min - y1_padding, y1_max + y1_padding)

                # 繪製主要折線圖
                ax1.plot(common_dates, data_dict['cumsum_total'],
                        label='三大法人', 
                        color='#1f77b4',
                        linewidth=3.0,
                        linestyle='-')

                ax1.plot(common_dates, data_dict['cumsum_investment'],
                        label='投信', 
                        color='#ff7f0e',
                        linewidth=2.5,
                        linestyle='--',
                        dashes=(5, 2))

                ax1.plot(common_dates, data_dict['cumsum_foreign'],
                        label='外資', 
                        color='#9467bd',
                        linewidth=2.5,
                        linestyle=':',
                        dashes=(2, 2))

                # 添加重要點位標記
                for line_data, color in [
                    (data_dict['cumsum_total'], '#1f77b4'),
                    (data_dict['cumsum_investment'], '#ff7f0e'),
                    (data_dict['cumsum_foreign'], '#9467bd')
                ]:
                    # 標記極值點
                    max_idx = np.argmax(line_data)
                    min_idx = np.argmin(line_data)
                    
                    ax1.plot(common_dates[max_idx], line_data[max_idx], 
                            'o', color=color, markersize=8)
                    ax1.plot(common_dates[min_idx], line_data[min_idx], 
                            'o', color=color, markersize=8)

                # 計算右側Y軸（每日買賣超）的比例範圍
                daily_data = pd.DataFrame({
                    'total': data_dict['daily_total_ratio'],
                    'foreign': data_dict['big_player1_ratio'],
                    'investment': data_dict['big_player2_ratio']
                }) * 100  # 轉換為百分比

                y2_min = daily_data.min().min()
                y2_max = daily_data.max().max()
                y2_range = y2_max - y2_min
                y2_padding = y2_range * 0.1

                # 設置右側Y軸範圍
                ax2.set_ylim(y2_min - y2_padding, y2_max + y2_padding)

                # 繪製柱狀圖
                bar_width = 0.35
                ax2.bar(common_dates, data_dict['big_player1_ratio'] * 100,
                        width=bar_width, alpha=0.3,
                        color='purple', label='外資每日')
                ax2.bar(common_dates, data_dict['big_player2_ratio'] * 100,
                        width=bar_width, alpha=0.3,
                        color='orange', label='投信每日')

                # 設置格式化器
                ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
                ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

                # 添加參考線和網格
                ax1.axhline(y=0, color='black', linestyle='-', alpha=0.2)
                ax1.grid(True, which='major', linestyle='--', alpha=0.2)

                # 設置刻度密度
                set_axis_density(ax1, y1_min, y1_max)
                set_axis_density(ax2, y2_min, y2_max)

                return ax1, ax2

            def set_axis_density(ax, min_val, max_val):
                """設置軸的刻度密度"""
                value_range = max_val - min_val
                if value_range <= 0.01:
                    step = 0.001
                elif value_range <= 0.1:
                    step = 0.01
                elif value_range <= 1:
                    step = 0.1
                else:
                    step = 0.5
                
                ticks = np.arange(
                    math.floor(min_val/step)*step,
                    math.ceil(max_val/step)*step + step,
                    step
                )
                ax.set_yticks(ticks)

            # 在主繪圖程式中使用
            data_dict = {
                'cumsum_total': cumsum_total.values,
                'cumsum_investment': cumsum_investment.values,
                'cumsum_foreign': cumsum_foreign.values,
                'daily_total_ratio': daily_total_ratio.values,
                'big_player1_ratio': big_player1_ratio.values,
                'big_player2_ratio': big_player2_ratio.values
            }

            ax1, ax2 = plot_with_scaled_axes(ax, common_dates, data_dict)




            # 11. 更新圖表標題和說明
            plt.subplots_adjust(top=0.85)

            # 格式化PR值變化
            pr_text, pr_color = format_pr_change(pr_change)

      
            ax.text(0.5, 1.35,
                    f'綜合評級: {rating} ({final_score:.0f}分) | 5日PR:{pr_5d:.0f}%| PR值變化: {pr_text}',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=30,
                    color=color)
            
            ax.text(0.5, 1.25,
                    f'法人動向: {strength_desc} ',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=30,
                    color=strength_color)


            # 設置買賣超信息
            sit_action = "買入" if investment_consecutive_days > 0 else "賣出"
            fi_action = "買入" if foreign_consecutive_days > 0 else "賣出"
            sit_sign = "+" if last_20_days_sit_ratio > 0 else ""
            fi_sign = "+" if last_20_days_fi_ratio > 0 else ""
            sit_color = "red" if last_20_days_sit_ratio > 0 else "green"
            fi_color = "red" if last_20_days_fi_ratio > 0 else "green"

            ax.text(0.5, 1.15,
                    f'投信/市值: {sit_sign}{last_20_days_sit_ratio:.2%} (連續{abs(investment_consecutive_days)}天{sit_action})',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=30,
                    color=sit_color)

            ax.text(0.5, 1.05,
                    f'外資/市值: {fi_sign}{last_20_days_fi_ratio:.2%} (連續{abs(foreign_consecutive_days)}天{fi_action})',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=30,
                    color=fi_color)

            # 12. 設置圖例
            custom_lines = [Line2D([0], [0], color='blue', lw=2.5),
                            Line2D([0], [0], color='orange', lw=2, linestyle='--'),
                            Line2D([0], [0], color='purple', lw=2, linestyle=':'),
                            Line2D([0], [0], color='gray', linestyle='--')]

            custom_labels = ['三大法人', '投信', '外資', '市場平均']

            ax.legend(custom_lines, custom_labels,
                    fontsize=20,
                    loc='upper left',
                    bbox_to_anchor=(0, 1))

            # 13. 設置座標軸
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(),
                    rotation=45,
                    ha='right')

            # 設置y軸標籤
            ax.set_ylabel('累\n積\n買\n賣\n超\n比\n例',
                        rotation=0,
                        fontsize=16,
                        labelpad=20)
            ax2.set_ylabel('每\n日\n買\n賣\n超\n比\n例',
                        rotation=0,
                        fontsize=16,
                        labelpad=20)

            # 14. 添加PR值變化指示器 (新增)
            def add_pr_indicator(ax, pr_change, y_pos):
                """添加PR值變化指示器"""
                if abs(pr_change) > 10:
                    marker = '▲' if pr_change > 0 else '▼'
                    color = 'red' if pr_change > 0 else 'green'
                    ax.text(1.02, y_pos, marker,
                            transform=ax.transAxes,
                            color=color,
                            fontsize=15)

            add_pr_indicator(ax, pr_change, 0.95)

            # 15. 優化圖表顯示
            # 調整y軸範圍以確保數據完全顯示
            y1_min, y1_max = ax.get_ylim()
            y2_min, y2_max = ax2.get_ylim()

            # 為了確保圖表美觀，稍微擴大顯示範圍
            ax.set_ylim(y1_min * 1.1, y1_max * 1.1)
            ax2.set_ylim(y2_min * 1.1, y2_max * 1.1)

            # 設置網格線
            ax.grid(True, which='both', linestyle='--', alpha=0.2)

            # 確保所有文字元素不會重疊
            plt.tight_layout()

            return True

        except Exception as e:
            # 詳細的錯誤處理
            logger.error(f"繪製三大法人圖表時發生錯誤: {str(e)}")
            logger.exception("詳細錯誤信息：")

            # 清除圖表並顯示錯誤信息
            ax.clear()
            ax.text(0.5, 0.5,
                    f'繪圖錯誤: {str(e)}',
                    ha='center',
                    va='center',
                    transform=ax.transAxes,
                    fontsize=20)

            return False



# ==========================================三大法人變化=================================================

    def plot_institutional_analysis(self, ax, stock_id):
        """
        三大法人累積買賣超變化 & 股價趨勢視覺化（增強版）
        功能：
        - 分析連續買賣天數和買賣連貫性
        - 判斷法人協同性（同步做多/做空）
        - 主力資金方向判斷
        - 柱狀圖加大放大倍數
        - 添加更完整的圖例
        - 分析法人買入成本分布
        """
        try:
            # 1. 取得數據
            close = self.data.get('price:收盤價')
            cap = self.data.get('etl:market_value')

            big_player1 = self.data.get(
                'institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)')
            big_player1 += self.data.get(
                'institutional_investors_trading_summary:外資自營商買賣超股數')

            big_player2 = self.data.get(
                'institutional_investors_trading_summary:投信買賣超股數')
            big_player3 = self.data.get(
                'institutional_investors_trading_summary:自營商買賣超股數(自行買賣)')

            big_player_all = big_player1 + big_player2 + big_player3  # 三大法人總買賣超

            # 填充NaN值
            close = close.fillna(method='ffill').fillna(method='bfill')
            cap = cap.fillna(method='ffill').fillna(method='bfill')
            big_player1 = big_player1.fillna(0)
            big_player2 = big_player2.fillna(0)
            big_player3 = big_player3.fillna(0)
            big_player_all = big_player_all.fillna(0)

            # 2. 計算法人影響力標準化指標
            def normalize_buysell(big_player, close_price, cap_data):
                return (big_player * close_price) / cap_data

            impact_1 = normalize_buysell(big_player1, close, cap)
            impact_2 = normalize_buysell(big_player2, close, cap)
            impact_3 = normalize_buysell(big_player3, close, cap)
            impact_all = normalize_buysell(big_player_all, close, cap)

            # 3. 選擇單一股票數據
            stock_price = close[stock_id]
            stock_cap_rank = cap.rank(axis=1)[stock_id].iloc[-1]  # 取得實際排名
            stock_cap_percentile = cap.rank(pct=True, axis=1)[stock_id]  # 市值排名百分比

            impact_1_stock = impact_1[stock_id]
            impact_2_stock = impact_2[stock_id]
            impact_3_stock = impact_3[stock_id]
            impact_all_stock = impact_all[stock_id]

            # 4. 計算累積買賣超 (只計算60天)
            impact_1_60 = impact_1_stock.rolling(60).sum()
            impact_2_60 = impact_2_stock.rolling(60).sum()
            impact_3_60 = impact_3_stock.rolling(60).sum()
            impact_all_60 = impact_all_stock.rolling(60).sum()

            # 5. 計算法人買超與股價相關性 (修正版)
            corr_window = 60
            # 修正：直接計算價格變動與法人買賣超的相關性
            price_returns = stock_price.pct_change()  # 股價日報酬率
            # 直接計算相關性，不再做滾動平均處理
            rolling_corr = price_returns.rolling(
                corr_window).corr(impact_all_stock)
            # 取得最近有效相關性值
            recent_corr = rolling_corr.iloc[-1]
            # 如果最後一個值是NaN，嘗試獲取最後一個有效值
            if pd.isna(recent_corr):
                valid_corr = rolling_corr.dropna()
                recent_corr = valid_corr.iloc[-1] if len(valid_corr) > 0 else 0

            # 6. 只保留最近120天的數據
            last_120_days = stock_price.index[-120:]
            stock_price = stock_price.loc[last_120_days]
            impact_1_stock = impact_1_stock.loc[last_120_days]
            impact_2_stock = impact_2_stock.loc[last_120_days]
            impact_3_stock = impact_3_stock.loc[last_120_days]
            impact_1_60 = impact_1_60.loc[last_120_days]
            impact_2_60 = impact_2_60.loc[last_120_days]
            impact_3_60 = impact_3_60.loc[last_120_days]
            impact_all_60 = impact_all_60.loc[last_120_days]

            # 7. 【新功能】計算連續買賣天數和連貫性
            def calc_consecutive_days(series):
                """計算連續買賣天數"""
                # 創建買賣標記序列 (1=買, -1=賣, 0=不動)
                sign = pd.Series(np.sign(series), index=series.index).replace(
                    0, np.nan).ffill()

                # 計算連續天數 (當符號改變時重置)
                consecutive = sign.groupby(
                    (sign != sign.shift()).cumsum()).cumcount() + 1

                # 獲取最近的連續天數
                if len(consecutive) > 0:
                    latest_sign = sign.iloc[-1] if not pd.isna(
                        sign.iloc[-1]) else 0
                    latest_days = consecutive.iloc[-1]
                    return int(latest_sign), int(latest_days)
                return 0, 0

            # 計算連續買賣天數
            foreign_sign, foreign_days = calc_consecutive_days(
                impact_1_stock.iloc[-20:])  # 僅看最近20天
            sitc_sign, sitc_days = calc_consecutive_days(impact_2_stock.iloc[-20:])
            dealer_sign, dealer_days = calc_consecutive_days(
                impact_3_stock.iloc[-20:])
            all_sign, all_days = calc_consecutive_days(impact_all_stock.iloc[-20:])

            # 8. 【新功能】計算買賣連貫性（穩定度）
            def calc_consistency(series, window=20):
                """計算買賣連貫性 (一致性)"""
                if len(series) < window:
                    return 0

                # 計算最近window天中買入/賣出的天數比例
                recent = series.iloc[-window:]
                buy_days = sum(recent > 0)
                sell_days = sum(recent < 0)

                # 一致性 = 主導方向的天數佔比 (0.5-1.0)
                if buy_days > sell_days:
                    return buy_days / window
                else:
                    return sell_days / window

            foreign_consistency = calc_consistency(impact_1_stock)
            sitc_consistency = calc_consistency(impact_2_stock)
            dealer_consistency = calc_consistency(impact_3_stock)
            all_consistency = calc_consistency(impact_all_stock)

            # 9. 【新功能】計算法人協同性/背離性
            def calc_synchronization(series1, series2, window=20):
                """計算兩個序列的協同性"""
                if len(series1) < window or len(series2) < window:
                    return 0

                # 提取最近window天數據
                recent1 = series1.iloc[-window:]
                recent2 = series2.iloc[-window:]

                # 計算方向一致的天數比例
                same_direction = sum((recent1 > 0) & (
                    recent2 > 0) | (recent1 < 0) & (recent2 < 0))
                return same_direction / window

            # 計算主要法人之間的協同性
            sync_foreign_sitc = calc_synchronization(
                impact_1_stock, impact_2_stock)
            sync_foreign_dealer = calc_synchronization(
                impact_1_stock, impact_3_stock)
            sync_sitc_dealer = calc_synchronization(impact_2_stock, impact_3_stock)

            # 獲取主要的協同/背離關係
            sync_pairs = [
                ("外資-投信", sync_foreign_sitc),
                ("外資-自營商", sync_foreign_dealer),
                ("投信-自營商", sync_sitc_dealer)
            ]
            max_sync = max(sync_pairs, key=lambda x: x[1])
            min_sync = min(sync_pairs, key=lambda x: x[1])

            # 10. 縮放係數計算
            all_daily_impacts = pd.concat(
                [impact_1_stock, impact_2_stock, impact_3_stock])
            data_range = all_daily_impacts.max() - all_daily_impacts.min()

            if data_range < 0.001:
                scale_factor = 200
            elif data_range < 0.005:
                scale_factor = 50
            elif data_range < 0.01:
                scale_factor = 25
            elif data_range < 0.03:
                scale_factor = 10
            else:
                scale_factor = 5

            # 11. 分析主力資金來源與方向
            recent_impact_1 = impact_1_60.iloc[-1]
            recent_impact_2 = impact_2_60.iloc[-1]
            recent_impact_3 = impact_3_60.iloc[-1]
            recent_impact_all = impact_all_60.iloc[-1]

            recent_impacts = [recent_impact_1, recent_impact_2, recent_impact_3]
            recent_abs_impacts = [abs(impact) for impact in recent_impacts]

            if all(impact == 0 for impact in recent_abs_impacts):
                # 所有值都是0時，設定一個預設值
                max_impact_idx = 0
                main_player = "無明顯主力"
                main_player_direction = "觀望"
                main_player_strength = "無明顯資金流向"  # 直接設定主力強度描述
            else:
                max_value = max(recent_abs_impacts)
                # 找出第一個等於最大值的索引
                max_impact_idx = next((i for i, v in enumerate(recent_abs_impacts) if v == max_value), 0)
                main_player = "外資" if max_impact_idx == 0 else "投信" if max_impact_idx == 1 else "自營商"
                main_player_direction = "做多" if recent_impacts[max_impact_idx] > 0 else "做空"
                
                # 進一步判斷主力強度
                main_impact = recent_abs_impacts[max_impact_idx]
                second_impact = sorted(recent_abs_impacts, reverse=True)[1]

                if main_impact > second_impact * 3:
                    main_player_strength = f"以{main_player}{main_player_direction}為絕對主導"
                elif main_impact > second_impact * 1.5:
                    main_player_strength = f"以{main_player}{main_player_direction}為主要資金來源"
                else:
                    # 檢查是否有接近的第二強力量
                    second_max_idx = sorted(range(len(recent_abs_impacts)),
                                        key=lambda i: recent_abs_impacts[i],
                                        reverse=True)[1]
                    second_player = "外資" if second_max_idx == 0 else "投信" if second_max_idx == 1 else "自營商"
                    second_direction = "做多" if recent_impacts[second_max_idx] > 0 else "做空"
                    main_player_strength = f"{main_player}{main_player_direction}和{second_player}{second_direction}共同主導"



            # 11.5 【新功能】分析法人買入成本分布
            # 將120天股價切分為10等分，計算法人在不同價格區間的買賣情況
            price_min = stock_price.min()
            price_max = stock_price.max()
            price_range = price_max - price_min

            # 創建10個等分的價格區間
            # 股價範圍被分為10個等分區間，索引從0到9
            # 每個區間代表股價從最低到最高的10%區間

            # 初始化累積買賣超數據
            institutional_buy_sell_by_price = {
                '外資': np.zeros(10),
                '投信': np.zeros(10),
                '自營商': np.zeros(10),
                '三大法人': np.zeros(10)
            }

            # 計算每個價格區間的買賣超
            for i in range(len(stock_price)):
                current_price = stock_price.iloc[i]
                # 找到對應的價格區間
                bin_idx = min(
                    9, max(0, int((current_price - price_min) / price_range * 10)))

                # 累加該價格區間的買賣超
                institutional_buy_sell_by_price['外資'][bin_idx] += impact_1_stock.iloc[i]
                institutional_buy_sell_by_price['投信'][bin_idx] += impact_2_stock.iloc[i]
                institutional_buy_sell_by_price['自營商'][bin_idx] += impact_3_stock.iloc[i]
                institutional_buy_sell_by_price['三大法人'][bin_idx] += impact_all_stock.iloc[i]

            # 計算法人買入集中的價格區間
            total_buy_sell = institutional_buy_sell_by_price['三大法人']
            main_buy_bins = [i for i, val in enumerate(total_buy_sell) if val > 0]
            main_sell_bins = [i for i, val in enumerate(total_buy_sell) if val < 0]

            # 分析法人買入成本與當前價格的關係
            current_price = stock_price.iloc[-1]
            current_bin = min(
                9, max(0, int((current_price - price_min) / price_range * 10)))

            # 判斷買入的主要價格區間
            if len(main_buy_bins) > 0:
                # 計算買入加權平均區間
                buy_values = [institutional_buy_sell_by_price['三大法人'][b] for b in main_buy_bins
                            if institutional_buy_sell_by_price['三大法人'][b] > 0]

                if sum(buy_values) > 0:  # 確保分母不為零
                    avg_buy_bin = sum([b * institutional_buy_sell_by_price['三大法人'][b] for b in main_buy_bins
                                    if institutional_buy_sell_by_price['三大法人'][b] > 0]) / \
                                sum(buy_values)

                    # 判斷法人買入策略
                    if avg_buy_bin < 3 and current_bin > 7:  # 低檔買入，目前高檔
                        cost_strategy = f"三大法人低檔布局({int(avg_buy_bin*10)}-{int(avg_buy_bin*10)+10}%區間)，目前處於高檔"
                    elif avg_buy_bin > 7 and current_bin < 3:  # 高檔買入，目前低檔
                        cost_strategy = f"三大法人高檔買入({int(avg_buy_bin*10)}-{int(avg_buy_bin*10)+10}%區間)，目前處於低檔"
                    elif abs(avg_buy_bin - current_bin) <= 1:  # 近期買入
                        cost_strategy = "三大法人近期買入，成本接近當前價格"
                    elif avg_buy_bin < current_bin:
                        cost_strategy = f"三大法人在較低位置({int(avg_buy_bin*10)}%區間)買入，目前處於{int(current_bin*10)}%區間"
                    else:
                        cost_strategy = f"三大法人在較高位置({int(avg_buy_bin*10)}%區間)買入，目前處於{int(current_bin*10)}%區間"
                else:
                    cost_strategy = "三大法人交易量不足，無法分析買入成本"
            else:
                cost_strategy = "三大法人淨賣出，無法分析買入成本"

            # 12. 繪圖設定
            ax.clear()
            ax2 = ax.twinx()

            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax.fill_between(impact_all_60.index, 0, impact_all_60,
                            where=(impact_all_60 >= 0), color='lightcoral', alpha=0.2,
                            label='買超區')
            ax.fill_between(impact_all_60.index, 0, impact_all_60,
                            where=(impact_all_60 < 0), color='lightgreen', alpha=0.2,
                            label='賣超區')

            # 使用柱狀圖顯示每日法人買賣超
            width = 0.8
            ax.bar(impact_1_stock.index, impact_1_stock * scale_factor,
                width=width, color='blue', alpha=0.6, label='外資買賣超')
            ax.bar(impact_2_stock.index, impact_2_stock * scale_factor,
                width=width, color='orange', alpha=0.6, label='投信買賣超')
            ax.bar(impact_3_stock.index, impact_3_stock * scale_factor,
                width=width, color='green', alpha=0.6, label='自營商買賣超')

            # 使用線圖顯示累積買賣超趨勢
            ax.plot(impact_all_60, color='black', linestyle='--',
                    label='三大法人累積買超(60天)', linewidth=2.5)

            # 繪製股價走勢
            ax2.plot(stock_price, color='red',
                    label='股價', linewidth=2.5, alpha=0.8)

            # 13. 【新功能】標記連續買賣模式
            # 為最近的連續買賣添加特殊標記
            consecutive_end_date = impact_all_stock.index[-1]
            consecutive_start_date = impact_all_stock.index[max(
                -all_days, -len(impact_all_stock))]

            # 連續買賣區域高亮
            if all_days >= 3:
                # 連續買入超過3天時使用淺紅色高亮
                if all_sign > 0:
                    ax.axvspan(consecutive_start_date, consecutive_end_date,
                            color='red', alpha=0.15)
                    ax.text(consecutive_end_date, ax.get_ylim()[1]*0.9,
                            f"連續買入{all_days}天", color='darkred', fontsize=14,
                            ha='right', fontweight='bold')
                # 連續賣出超過3天時使用淺綠色高亮
                elif all_sign < 0:
                    ax.axvspan(consecutive_start_date, consecutive_end_date,
                            color='green', alpha=0.15)
                    ax.text(consecutive_end_date, ax.get_ylim()[1]*0.9,
                            f"連續賣出{all_days}天", color='darkgreen', fontsize=14,
                            ha='right', fontweight='bold')

            # 14. 【修改】只保留協同買賣的視覺指示，移除背離標示
            if max_sync[1] > 0.7:  # 70%以上的時間方向一致，表示高度協同
                ax.text(impact_all_stock.index[-40], ax.get_ylim()[1]*0.8,
                        f"{max_sync[0]}高度協同性 {max_sync[1]:.0%}",
                        color='purple', fontsize=14, fontweight='bold')

            # 15. 判斷連貫買賣的強度描述


            def get_consistency_desc(consistency):
                if consistency > 0.85:
                    return "極高連貫性"
                elif consistency > 0.7:
                    return "高連貫性"
                elif consistency > 0.55:
                    return "中等連貫性"
                else:
                    return "低連貫性"


            # 16. 獲取市值排名描述
            cap_rank_pct = stock_cap_percentile.iloc[-1]
            if cap_rank_pct >= 0.9:
                cap_size = f"大型股 (市值PR{int(cap_rank_pct*100)})"
            elif cap_rank_pct >= 0.7:
                cap_size = f"中大型股 (市值PR{int(cap_rank_pct*100)})"
            elif cap_rank_pct >= 0.3:
                cap_size = f"中型股 (市值PR{int(cap_rank_pct*100)})"
            elif cap_rank_pct >= 0.1:
                cap_size = f"中小型股 (市值PR{int(cap_rank_pct*100)})"
            else:
                cap_size = f"小型股 (市值PR{int(cap_rank_pct*100)})"

            # 17. 整理連續買賣天數描述
            consecutive_desc = []
            if foreign_days >= 3:
                action = "買入" if foreign_sign > 0 else "賣出"
                consecutive_desc.append(
                    f"外資連續{action}{foreign_days}天({get_consistency_desc(foreign_consistency)})")
            if sitc_days >= 3:
                action = "買入" if sitc_sign > 0 else "賣出"
                consecutive_desc.append(
                    f"投信連續{action}{sitc_days}天({get_consistency_desc(sitc_consistency)})")
            if dealer_days >= 3:
                action = "買入" if dealer_sign > 0 else "賣出"
                consecutive_desc.append(
                    f"自營商連續{action}{dealer_days}天({get_consistency_desc(dealer_consistency)})")

            consecutive_text = " | ".join(
                consecutive_desc) if consecutive_desc else "無明顯連續買賣模式"

            # 18. 判斷訊號文字
            if recent_impact_all > 0.02:
                signal_text = "強烈買入"
            elif recent_impact_all > 0.005:
                signal_text = "買入"
            elif recent_impact_all < -0.02:
                signal_text = "強烈賣出"
            elif recent_impact_all < -0.005:
                signal_text = "賣出"
            else:
                signal_text = "觀望"

            # 19. 標題設計 (修改，將背離信息移至標題)
            # 準備背離信息文字
            divergence_info = ""
            if min_sync[1] < 0.3:  # 30%以下的時間方向一致，表示明顯背離
                divergence_info = f" | {min_sync[0]}明顯背離({min_sync[1]:.0%})"

            corr_text = f"法人-股價相關性: {recent_corr:.2f}"
            # 繼續完成標題設定
            title = (f"{stock_id} | {cap_size} | {main_player_strength}{divergence_info} | {corr_text}\n"
                f"訊號: {signal_text} | {cost_strategy} | 近60日累積買超: 外資 {recent_impact_1:.2%} | 投信 {recent_impact_2:.2%} | "
                f"自營商 {recent_impact_3:.2%} | 三大法人 {recent_impact_all:.2%}\n"
                f"{consecutive_text}")

            # 在標題加上縮放提示
            title += f"\n(註: 柱狀圖已放大{scale_factor}倍以增強可視性)"

            ax.set_title(title, fontsize=35, fontweight='bold')

            # 20. Y軸標籤
            ax.set_ylabel("法\n人\n買\n賣\n超\n標\n準\n化\n(%)", fontsize=20, rotation=0, labelpad=20)
            ax2.set_ylabel("股\n價", fontsize=20, color='red', rotation=0, labelpad=20)

            # 21. X軸設置 (每5天顯示一次)
            import matplotlib.dates as mdates
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.tick_params(axis='x', labelrotation=45, labelsize=18)

            # Y軸百分比顯示
            import matplotlib.ticker as mtick
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

            # 設置刻度顏色與大小
            ax.tick_params(axis='y', labelsize=14)
            ax2.tick_params(axis='y', labelcolor='red', labelsize=14)

            # 22. 添加額外的圖例項目
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()

            # 確保圖例中包含所有元素
            all_labels = labels1 + labels2
            all_handles = handles1 + handles2

            # 添加連續買賣模式圖例項目
            if all_days >= 3:
                import matplotlib.lines as mlines
                if all_sign > 0:
                    continuous_marker = mlines.Line2D([], [], color='darkred', marker='s', 
                                                linestyle='None', markersize=10, alpha=0.5,
                                                label=f'連續買入區塊')
                else:
                    continuous_marker = mlines.Line2D([], [], color='darkgreen', marker='s', 
                                                linestyle='None', markersize=10, alpha=0.5,
                                                label=f'連續賣出區塊')
                all_handles.append(continuous_marker)
                all_labels.append(f'連續{abs(all_days)}天{"買入" if all_sign > 0 else "賣出"}區塊')

            # 確保"投信買賣超"在圖例中
            if '投信買賣超' not in all_labels:
                import matplotlib.patches as mpatches
                orange_bar = mpatches.Patch(color='orange', alpha=0.6, label='投信買賣超')
                all_handles.append(orange_bar)
                all_labels.append('投信買賣超')

            # 組合圖例
            ax.legend(all_handles, all_labels, loc='upper left', fontsize=18, framealpha=0.7)

            # 顯示網格
            ax.grid(True, linestyle='--', alpha=0.4)

            # 添加圖表邊框
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
                
            # 自動調整Y軸範圍
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            ax.set_ylim(y_min - y_range*0.15, y_max + y_range*0.15)

            return True

        except Exception as e:
            print(f"繪圖錯誤: {e}")
            import traceback
            traceback.print_exc()
            ax.clear()
            ax.text(0.5, 0.5, f'繪圖錯誤: {str(e)}', ha='center', va='center', transform=ax.transAxes)
            return False




# ==========================================融資判斷==============================================================


    def plot_margin_trading_focus(self, ax, stock_id):
        try:
            # 1. 資料準備
            margin_prev = data.get('margin_transactions:融資前日餘額').fillna(method='ffill')
            margin_today = data.get('margin_transactions:融資今日餘額').fillna(method='ffill')
            short_today = data.get('margin_transactions:融券今日餘額').fillna(method='ffill')
            volume = data.get('price:成交股數')
            cap = data.get('etl:market_value')
            amt = data.get('price:成交金額')
            intraday_vol = data.get("intraday_trading:當日沖銷交易成交股數").fillna(method='ffill')
            margin_transaction_ratio = data.get('margin_transactions:融資使用率').fillna(method='ffill')

            # 2. 計算成交值比率
            turnover = amt / cap

            # 3. 成交值波動分析函數
            def analyze_turnover_volatility(turnover, stock_id):
                ma5 = turnover.rolling(5).mean()
                ma20 = turnover.rolling(20).mean()
                std5 = turnover.rolling(5).std()
                std20 = turnover.rolling(20).std()
                
                z_score5 = (turnover - ma5) / std5
                z_score20 = (turnover - ma20) / std20
                
                current_z5 = z_score5[stock_id].iloc[-1]
                current_z20 = z_score20[stock_id].iloc[-1]
                volatility_ratio = std5[stock_id].iloc[-1] / std20[stock_id].iloc[-1]
                
                return current_z5, current_z20, volatility_ratio

            # 4. 市場狀態判斷函數
            def get_market_status(z5, z20, vol_ratio):
                if z5 > 2 and z20 > 2:
                    return "大量進場", "短期和中期都出現大量", 'red'
                elif z5 > 1 and z20 > 1:
                    return "持續放量", "買盤持續活躍", 'lightcoral'
                elif z5 > 1 > z20:
                    return "突然放量", "近期買盤增加", 'pink'
                elif z5 < -2 and z20 < -2:
                    return "量能枯竭", "短期和中期都極度清淡", 'green'
                elif z5 < -1 and z20 < -1:
                    return "持續清淡", "交投持續冷清", 'lightgreen'
                elif z5 < -1 < z20:
                    return "近期觀望", "交投轉趨保守", 'palegreen'
                else:
                    if vol_ratio > 1.5:
                        return "波動加大", "盤勢轉趨活躍", 'orange'
                    elif vol_ratio < 0.7:
                        return "波動收斂", "盤勢趨於穩定", 'blue'
                    else:
                        return "正常波動", "交投穩定", 'black'

            # 5. 資料有效性檢查
            if stock_id not in margin_today.columns:
                logger.warning(f"無法獲取 {stock_id} 的融資融券資料")
                ax.clear()
                ax.text(0.5, 0.5, f'無 {stock_id} 融資融券資料',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=20)
                return False

            # 6. 資料處理與計算
            days = 60
            df = pd.DataFrame({
                '融資餘額': margin_today[stock_id].tail(days),
                '融券餘額': short_today[stock_id].tail(days),
                '成交量': volume[stock_id].tail(days) / 1000,
                '當沖量': intraday_vol[stock_id].tail(days) / 1000,
                '融資使用率': margin_transaction_ratio[stock_id].tail(days),
            })

            # 資料清理
            df['成交量'] = df['成交量'].replace(0, np.nan)
            df['融資餘額'] = df['融資餘額'].replace(0, np.nan)
            df = df.dropna()

            if len(df) < 5:
                logger.warning(f"{stock_id} 有效資料筆數不足")
                ax.clear()
                ax.text(0.5, 0.5, f'{stock_id} 有效資料不足',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=20)
                return False

            # 7. 市場數據計算
            market_df = pd.DataFrame({
                stock: margin_transaction_ratio[stock].fillna(method='ffill')
                for stock in margin_transaction_ratio.columns
            })
            market_avg = market_df.mean(axis=1)
            market_trends = pd.DataFrame({
                '市場平均融資使用率': market_avg,
                '市場日變化': market_avg.diff(),
                '市場五日變化': market_avg.diff(5)
            })

            # 8. 個股指標計算
            df['融資變動'] = df['融資餘額'] - margin_prev[stock_id].loc[df.index]
            df['融資變動率'] = (df['融資變動'] / margin_prev[stock_id].loc[df.index] * 100).fillna(0).clip(-20, 20)
            df['當沖率'] = (df['當沖量'] / 2 / df['成交量'] * 100).fillna(0)
            df['券資比'] = (df['融券餘額'] / df['融資餘額'] * 100).fillna(0)
            
            # 變化率指標
            df['融資使用率變化'] = df['融資使用率'].diff()
            df['五日融資使用率變化'] = df['融資使用率'].diff(5)
            
            # 市場比較指標
            df['市場平均融資使用率'] = market_trends['市場平均融資使用率']
            df['市場五日變化'] = market_trends['市場五日變化']
            df['相對強度'] = df['融資使用率'] - df['市場平均融資使用率']
            df['相對五日變化'] = df['五日融資使用率變化'] - df['市場五日變化']

            # 9. 統計值計算
            last_margin_usage = df['融資使用率'].iloc[-1]
            last_individual_change = df['五日融資使用率變化'].iloc[-1]
            last_market_change = df['市場五日變化'].iloc[-1]
            last_relative_change = df['相對五日變化'].iloc[-1]
            relative_percentile = stats.percentileofscore(
                df['相對五日變化'].dropna(), last_relative_change)

            # 10. 強度判斷函數
            def get_market_strength(individual_change, market_change, percentile):
                if percentile >= 80:
                    return "明顯強於大盤", "red", "融資動能大幅領先市場"
                elif percentile >= 60:
                    return "略強於大盤", "indianred", "融資動能優於市場"
                elif percentile >= 40:
                    return "與大盤同步", "black", "融資動能與市場同步"
                elif percentile >= 20:
                    return "略弱於大盤", "green", "融資動能略遜於市場"
                else:
                    return "明顯弱於大盤", "darkgreen", "融資動能明顯落後"

            # 11. 繪圖
            dates = df.index
            
            # 主圖（融資變動率）
            ax.fill_between(dates, df['融資變動率'], 0,
                        where=(df['融資變動率'] >= 0),
                        color='lightcoral', alpha=0.3, label='融資增加')
            ax.fill_between(dates, df['融資變動率'], 0,
                        where=(df['融資變動率'] < 0),
                        color='lightgreen', alpha=0.3, label='融資減少')

            # 移動平均線
            ax.plot(dates, df['融資變動率'].rolling(window=5).mean(),
                    color='navy', linestyle='-',
                    marker='.', markersize=2, label='5日均線',
                    linewidth=1.5, alpha=0.8)
            ax.plot(dates, df['融資變動率'].rolling(window=20).mean(),
                    color='darkred', linestyle='-.',
                    marker='.', markersize=2, label='20日均線',
                    linewidth=1.5, alpha=0.8)
            ax.plot(dates, df['市場五日變化'],
                    color='gray', linestyle='--',
                    marker='', label='市場平均',
                    linewidth=1, alpha=0.5)

            # 券資比（右軸1）
            ax2 = ax.twinx()
            ax2.spines['right'].set_position(('outward', 60))
            ax2.plot(dates, df['券資比'],
                    color='purple', linestyle=':',
                    marker='s', markersize=3, label='券資比',
                    linewidth=1.5, alpha=0.7)
            ax2.axhline(y=15, color='red', linestyle=':',
                        alpha=0.4, label='券資比警戒線')

            # 融資使用率（右軸2）  
            ax3 = ax.twinx()
            ax3.spines['right'].set_position(('outward', 120))
            ax3.spines['right'].set_color('#FF1493')  # 對應軸線顏色

            ax3.plot(dates, df['融資使用率'],
                    color='#FF1493',  # 使用亮粉紅色 (Deep Pink)
                    linestyle='-',
                    marker='o',
                    markersize=3,
                    label='融資使用率',
                    linewidth=2.5,  # 加粗線條
                    alpha=1.0)      # 提高不透明度

            # 12. 圖表樣式設置
            # 軸標籤
            ax.set_ylabel('融\n資\n日\n變\n動\n率\n%',
                        rotation=0, labelpad=25, color='navy', fontsize=20)
            ax2.set_ylabel('券\n資\n比\n%',
                        rotation=0, labelpad=25, color='purple', fontsize=20)
        
            ax3.set_ylabel('融\n資\n使\n用\n率\n%',
            rotation=0,
            labelpad=25,
            color='#FF1493',  # 對應軸標籤顏色
            fontsize=20,
            fontweight='bold')  # 加粗字體

            # 軸範圍和顏色
            ax.set_ylim(-20, 20)
            ax2.set_ylim(0, max(30, df['券資比'].max()*1.2))
            ax3.set_ylim(0, max(100, df['融資使用率'].max()*1.2))

            # 軸顏色設置
            ax.tick_params(axis='y', colors='navy')
            ax2.tick_params(axis='y', colors='purple')
            ax3.tick_params(axis='y',
                colors='#FF1493',  # 對應刻度顏色
                labelsize=10)

            # 網格和背景
            ax.grid(True, linestyle='--', alpha=0.2)
            ax.set_facecolor('white')
            ax.xaxis.set_major_locator(plt.MaxNLocator(10))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

            # 13. 圖例設置
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines3, labels3 = ax3.get_legend_handles_labels()
            ax.legend(lines1 + lines2 + lines3,
                    labels1 + labels2 + labels3,
                    loc='upper left',
                    bbox_to_anchor=(0, 1.02),
                    ncol=4)

            # 14. 標題和說明文字
            strength, color, description = get_market_strength(
                last_individual_change, last_market_change, relative_percentile)
            z5, z20, vol_ratio = analyze_turnover_volatility(turnover, stock_id)
            status, detail, turnover_color = get_market_status(z5, z20, vol_ratio)
            
            main_title = f'{stock_id} 融資融券力道分析'
            usage_text = f'融資使用率: {last_margin_usage:.1f}% | 市場平均: {df["市場平均融資使用率"].iloc[-1]:.1f}%'
            change_text = f'個股五日變化: {last_individual_change:+.1f}% | 市場五日變化: {last_market_change:+.1f}%'
            strength_text = f'相對強度: {strength} ({relative_percentile:.0f}%) - {description}'
            ratio_text = f'券資比: {df["券資比"].iloc[-1]:.1f}% | 當沖率: {df["當沖率"].iloc[-1]:.1f}%'
            turnover_text = f"{status} ({detail})"

            # 文字位置設置
            center_x = 0.5
            ax.text(0.5, 1.42, main_title,
                    ha='center', va='bottom',
                    transform=ax.transAxes,
                    fontsize=30, color='black',
                    fontweight='bold')
            ax.text(0.5, 1.32, usage_text,
                    ha='center', va='bottom',
                    transform=ax.transAxes,
                    fontsize=30, color='black',
                    fontweight='bold')
            ax.text(0.5, 1.22, change_text,
                    ha='center', va='bottom',
                    transform=ax.transAxes,
                    fontsize=30, color=color,
                    fontweight='bold')
            ax.text(0.5, 1.12, strength_text,
                    ha='center', va='bottom',
                    transform=ax.transAxes,
                    fontsize=30, color=color,
                    fontweight='bold')
            ax.text(center_x, 1.02, f"{ratio_text} | ",
                    ha='right', va='bottom',
                    transform=ax.transAxes,
                    fontsize=30, color='black',
                    fontweight='bold')
            ax.text(center_x, 1.02, turnover_text,
                    ha='left', va='bottom',
                    transform=ax.transAxes,
                    fontsize=30, color=turnover_color,
                    fontweight='bold')

            # X軸標籤旋轉
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            return True

        except Exception as e:
            logger.error(f"繪製融資融券圖表時發生錯誤: {str(e)}")
            ax.clear()
            ax.text(0.5, 0.5, f'繪圖錯誤',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=20)
            return False







# =================================券商分點===============================================////


    def plot_broker_analysis(self, ax, stock_id):
        """繪製淨交易量與相對強度信號分析圖"""
        try:
            # 獲取數據
            close = data.get("price:收盤價")
            buy_vol = data.get(
                'etl:broker_transactions:top15_buy').fillna(method='ffill')
            sell_vol = data.get(
                'etl:broker_transactions:top15_sell').fillna(method='ffill')

            # 計算所有股票的淨交易量和相對強度
            market_forces = pd.DataFrame()

            for stock in close.columns:
                if stock in buy_vol.columns and stock in sell_vol.columns:
                    stock_close = close[stock].tail(480)
                    stock_buy = buy_vol[stock].tail(480)
                    stock_sell = sell_vol[stock].tail(480)

                    net_vol = (stock_buy - stock_sell) * stock_close
                    force = net_vol.rolling(
                        60).mean() / net_vol.rolling(60).std()
                    market_forces[stock] = force

            # 計算市場平均相對強度
            market_avg_force = market_forces.mean(axis=1)

            # 獲取目標股票的數據
            stock_close = close[stock_id].tail(480)
            stock_buy = buy_vol[stock_id].tail(480)
            stock_sell = sell_vol[stock_id].tail(480)
            net_volume = (stock_buy - stock_sell) * stock_close
            force = net_volume.rolling(
                60).mean() / net_volume.rolling(60).std()
            
            rolling_mean = net_volume.rolling(60).mean()
            rolling_std = net_volume.rolling(60).std()
            
            # 防止除以零或極小值
            force = pd.Series(index=net_volume.index, dtype=float)
            valid_std = rolling_std > 1e-10  # 設定一個很小的閾值
            force[valid_std] = rolling_mean[valid_std] / rolling_std[valid_std]
            force[~valid_std] = 0  # 當標準差太小時,設為0

            # 處理極端值
            force = force.replace([np.inf, -np.inf], np.nan)
            force = force.fillna(method='ffill').fillna(method='bfill')

            # 只選擇最近120天的數據
            dates = net_volume.index[-120:]
            net_volume_to_plot = net_volume[-120:]
            force_to_plot = force[-120:]
            close_to_plot = stock_close[-120:]
            market_force_to_plot = market_avg_force[-120:]

            # 計算相對於市場的強度並處理nan值
            relative_to_market = force_to_plot - market_force_to_plot
            relative_to_market = relative_to_market.fillna(method='ffill')

            # 計算市場波動度
            market_volatility = market_forces.std(axis=1)[-120:]
            market_vol_ma = market_volatility.rolling(20).mean()
            vol_factor = 1 + (market_vol_ma - market_vol_ma.mean()
                            ) / market_vol_ma.std()

            # 定義調整後的PR值計算函數
            def calculate_adjusted_pr(relative_strength, vol_factor, window_data):
                """
                根據市場波動度調整PR值計算
                """
                base_pr = stats.percentileofscore(
                    window_data.dropna(),
                    relative_strength
                )

                # 調整PR值：高波動時向中間值收斂，低波動時強化極端值
                vol_adj = vol_factor.iloc[-1] if not np.isnan(
                    vol_factor.iloc[-1]) else 1
                adjusted_pr = 50 + (base_pr - 50) / vol_adj
                return np.clip(adjusted_pr, 0, 100)

            # 計算近5天的PR值變化
            recent_5days_pr = []
            for i in range(5):
                if len(relative_to_market) > i:
                    daily_relative = relative_to_market.iloc[-(i+1)]
                    window_data = relative_to_market.iloc[:-(i+1)]
                    pr_i = calculate_adjusted_pr(
                        daily_relative,
                        vol_factor,
                        window_data
                    )
                    recent_5days_pr.append(pr_i)
                else:
                    recent_5days_pr.append(np.nan)

            # 計算PR值變化趨勢
            pr_change = recent_5days_pr[0] - recent_5days_pr[-1]
            pr_daily_changes = np.diff(recent_5days_pr[::-1])

            # 更新PR值
            pr_value = recent_5days_pr[0]  # 使用調整後的最新PR值

            # 計算統計數據
            latest_force = force_to_plot.iloc[-1]
            if np.isnan(latest_force):
                latest_force = force_to_plot.fillna(method='ffill').iloc[-1]

            latest_net_volume = net_volume_to_plot.iloc[-1]
            latest_market_force = market_force_to_plot.iloc[-1]

            # PR值變化趨勢的解釋
            pr_trend = (
                "強勢加速" if pr_change > 20 and sum(x > 0 for x in pr_daily_changes) >= 3 else
                "穩定增強" if pr_change > 10 and sum(x > 0 for x in pr_daily_changes) >= 2 else
                "緩慢增強" if pr_change > 5 else
                "持平" if abs(pr_change) <= 5 else
                "緩慢走弱" if pr_change > -10 else
                "明顯走弱" if pr_change > -20 else
                "加速下跌"
            )

            # 強度解釋
            force_interpretation = (
                "極度強勢 (>2σ)" if latest_force > 2 else
                "強勢 (1~2σ)" if latest_force > 1 else
                "偏強 (0~1σ)" if latest_force > 0 else
                "偏弱 (-1~0σ)" if latest_force > -1 else
                "弱勢 (-2~-1σ)" if latest_force > -2 else
                "極度弱勢 (<-2σ)"
            )

            # 市場相對強度解釋
            market_interpretation = (
                "明顯強於大盤" if pr_value >= 80 else
                "略強於大盤" if pr_value >= 60 else
                "與大盤同步" if pr_value >= 40 else
                "略弱於大盤" if pr_value >= 20 else
                "明顯弱於大盤"
            )

            # 優化趨勢計算：使用加權平均
            weights = np.linspace(1, 2, 10)
            recent_force = force_to_plot.fillna(method='ffill').tail(10)
            recent_trend = np.average(recent_force, weights=weights)

            trend_interpretation = (
                "明顯轉強" if recent_trend > 1 else
                "逐漸走強" if recent_trend > 0.5 else
                "略為走強" if recent_trend > 0 else
                "略為走弱" if recent_trend > -0.5 else
                "逐漸走弱" if recent_trend > -1 else
                "明顯轉弱"
            )

            # 統計評分系統 (0-100分)
            def calculate_score():
                try:
                    scores = []
                    weights = []

                    # 1. 相對強度分數 (30%)
                    if not np.isnan(latest_force):
                        force_score = stats.norm.cdf(latest_force) * 100
                    else:
                        force_score = 50
                    scores.append(force_score)
                    weights.append(0.3)

                    # 2. 市場相對位置分數 (30%)
                    if not np.isnan(pr_value):
                        market_score = pr_value
                    else:
                        market_score = 50
                    scores.append(market_score)
                    weights.append(0.3)

                    # 3. 趨勢分數 (20%)
                    if not np.isnan(recent_trend):
                        trend_score = stats.norm.cdf(recent_trend) * 100
                    else:
                        trend_score = 50
                    scores.append(trend_score)
                    weights.append(0.2)

                    # 4. 波動性分數 (20%)
                    volatility = net_volume_to_plot.std()
                    if not np.isnan(volatility):
                        # 計算20天和60天的波動率
                        vol_20d = net_volume_to_plot.rolling(20).std().iloc[-1]
                        vol_60d = net_volume_to_plot.rolling(60).std().iloc[-1]

                        # 計算波動率變化
                        vol_change = (vol_20d / vol_60d - 1) * \
                            100 if vol_60d != 0 else 0

                        # 計算相對市場波動率
                        market_vol = market_forces.std()
                        relative_vol = volatility / market_vol.mean() if market_vol.mean() != 0 else 1

                        # 波動分數計算
                        vol_score = 0

                        # 波動率變化評分 (40%)
                        if abs(vol_change) < 20:  # 波動穩定
                            vol_score += 40
                        elif abs(vol_change) < 50:  # 波動適中
                            vol_score += 30
                        else:  # 波動過大
                            vol_score += 20

                        # 相對市場波動評分 (60%)
                        if relative_vol < 0.8:  # 低波動
                            vol_score += 60
                        elif relative_vol < 1.2:  # 中等波動
                            vol_score += 50
                        elif relative_vol < 1.5:  # 高波動
                            vol_score += 40
                        else:  # 極高波動
                            vol_score += 30

                        vol_percentile = vol_score  # 使用新的評分替換原本的percentileofscore
                    else:
                        vol_percentile = 50

                    scores.append(vol_percentile)
                    weights.append(0.2)

                    # 確保所有分數都不是 NaN
                    scores = np.array(scores)
                    weights = np.array(weights)
                    valid_mask = ~np.isnan(scores)

                    if not any(valid_mask):
                        return 50

                    # 重新計算權重比例
                    weights = weights[valid_mask]
                    weights = weights / weights.sum()

                    # 記錄計算過程
                    print(f"Force score: {force_score}")
                    print(f"Market score: {market_score}")
                    print(f"Trend score: {trend_score}")
                    print(f"Volatility score: {vol_percentile}")
                    print(f"Final scores: {scores}")
                    print(f"Weights: {weights}")

                    final_score = np.average(
                        scores[valid_mask], weights=weights)
                    print(f"Final score: {final_score}")

                    return final_score

                except Exception as e:
                    print(f"計算評分時發生錯誤: {str(e)}")
                    return 50

            # 計算總分並處理可能的 NaN
            total_score = calculate_score()
            if np.isnan(total_score):
                total_score = 50
                grade = "無法評分"
            else:
                grade = (
                    "極為優異" if total_score >= 90 else
                    "表現優異" if total_score >= 80 else
                    "表現良好" if total_score >= 70 else
                    "表現平穩" if total_score >= 60 else
                    "表現平平" if total_score >= 50 else
                    "表現欠佳" if total_score >= 40 else
                    "表現不佳"
                )

            # 創建雙軸圖表
            ax2 = ax.twinx()

            # 柱狀圖顏色（紅色表示強勢，綠色表示弱勢）
            bar_colors = np.where(net_volume_to_plot >= 0,
                                '#FF3333',  # 正值用紅色
                                '#00AA00')  # 負值用綠色

            bars = ax.bar(dates, net_volume_to_plot,
                        alpha=0.4,
                        color=bar_colors,
                        label='淨交易量',
                        width=0.8)

            # 繪製相對強度線
            ax2.plot(dates, force_to_plot,
                    color='black',
                    linewidth=2.5,
                    label='相對強度',
                    zorder=5)

            # 添加市場平均線
            ax2.plot(dates, market_force_to_plot,
                    color='gray',
                    linewidth=1.5,
                    linestyle='--',
                    label='市場平均',
                    alpha=0.6)
            
            # 創建第三個Y軸用於繪製收盤價
            ax3 = ax.twinx()
            ax3.spines['right'].set_position(('outward', 60))

            close_normalized = ((close_to_plot - close_to_plot.min()) / 
                            (close_to_plot.max() - close_to_plot.min()) * 
                            (force_to_plot.max() - force_to_plot.min()) + 
                            force_to_plot.min())

            ax3.plot(dates, close_normalized, 
                    color='blue',
                    linestyle='--',
                    linewidth=2,
                    label='收盤價',
                    zorder=4)

            def price_formatter(x, p):
                actual_price = (x - force_to_plot.min()) / (force_to_plot.max() - force_to_plot.min()) * \
                            (close_to_plot.max() - close_to_plot.min()) + close_to_plot.min()
                return f'{actual_price:.1f}'

            ax3.yaxis.set_major_formatter(plt.FuncFormatter(price_formatter))

            # 隱藏ax3的軸線和刻度
            ax3.set_yticklabels([])  # 隱藏刻度標籤
            ax3.spines['right'].set_visible(False)  # 隱藏右側軸線
            ax3.tick_params(right=False)  # 隱藏刻度線

                

            # === 收盤價相關代碼結束 ===
            

            # 定義漸層顏色函數
            def get_gradient_colors(force_values):
                """根據強度值返回漸層顏色"""
                colors = []
                for value in force_values:
                    if np.isnan(value):
                        colors.append('#FFFFFF00')  # 透明色處理NaN值
                    elif value >= 2:
                        colors.append('#FF0000')    # 極度強勢 >2σ
                    elif value >= 1:
                        colors.append('#FF3333')    # 強勢 1~2σ
                    elif value >= 0:
                        colors.append('#FF6666')    # 偏強 0~1σ
                    elif value >= -1:
                        colors.append('#66CC66')    # 偏弱 -1~0σ
                    elif value >= -2:
                        colors.append('#339933')    # 弱勢 -2~-1σ
                    else:
                        colors.append('#006600')    # 極度弱勢 <-2σ
                return colors

            # 設置零線
            zero_line = np.zeros_like(force_to_plot)

            # 正向強度填充
            for i in range(len(dates)-1):
                if force_to_plot.iloc[i] >= 0:
                    ax2.fill_between([dates[i], dates[i+1]], 
                                    [force_to_plot.iloc[i], force_to_plot.iloc[i+1]], 
                                    [0, 0],
                                    color=get_gradient_colors([force_to_plot.iloc[i]])[0],
                                    alpha=0.4)

            # 負向強度填充
            for i in range(len(dates)-1):
                if force_to_plot.iloc[i] <= 0:
                    ax2.fill_between([dates[i], dates[i+1]], 
                                    [force_to_plot.iloc[i], force_to_plot.iloc[i+1]], 
                                    [0, 0],
                                    color=get_gradient_colors([force_to_plot.iloc[i]])[0],
                                    alpha=0.4)


            # 標題顏色邏輯
            def get_title_colors():
                # 強度顏色
                force_color = (
                    '#FF0000' if latest_force > 1 else
                    '#FF6666' if latest_force > 0 else
                    '#66CC66' if latest_force > -1 else
                    '#009900'
                )

                # PR值顏色
                pr_color = (
                    '#FF0000' if pr_value >= 80 else
                    '#FF6666' if pr_value >= 60 else
                    'black' if pr_value >= 40 else
                    '#66CC66' if pr_value >= 20 else
                    '#009900'
                )

                # 趨勢顏色
                trend_color = (
                    '#FF0000' if recent_trend > 1 else
                    '#FF6666' if recent_trend > 0 else
                    '#66CC66' if recent_trend > -1 else
                    '#009900'
                )

                return force_color, pr_color, trend_color

            force_color, pr_color, trend_color = get_title_colors()

            # 更新標題（使用不同顏色）
            title_lines = [
                f'{stock_id} 券商15大分點淨交易量與相對強度分析',
                f'目前強度: {latest_force:.2f}σ ({force_interpretation})({trend_interpretation})',
                f'相對市場: {market_interpretation} (PR值: {pr_value:.0f}, {pr_trend})',
                f'綜合評分: {total_score:.0f}分 ({grade})'
                ]

            # 使用多行標題，每行不同顏色
            ax.text(0.5, 1.28, title_lines[0],
                    horizontalalignment='center',
                    transform=ax.transAxes,
                    fontsize=30,
                    weight='bold')

            ax.text(0.5, 1.19, title_lines[1],
                    horizontalalignment='center',
                    transform=ax.transAxes,
                    fontsize=26,
                    color=force_color)

            ax.text(0.5, 1.1, title_lines[2],
                    horizontalalignment='center',
                    transform=ax.transAxes,
                    fontsize=26,
                    color=pr_color)

            ax.text(0.5, 1.01, title_lines[3],
                    horizontalalignment='center',
                    transform=ax.transAxes,
                    fontsize=26,
                    color=trend_color)

            # 優化軸標籤
            ax.set_ylabel('淨\n交\n易\n量',
                            fontsize=16,
                            rotation=0,
                            labelpad=20,
                            weight='bold')
            ax2.set_ylabel('相\n對\n強\n度',
                            fontsize=16,
                            rotation=0,
                            labelpad=20,
                            weight='bold')

            # 優化圖例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines3, labels3 = ax3.get_legend_handles_labels()
            ax2.legend(lines1 + lines2 + lines3, 
                    labels1 + labels2 + labels3,
                    loc='upper left',
                    fontsize=14,
                    framealpha=0.8,
                    edgecolor='gray',
                    facecolor='white')



            # 優化網格和軸設置
            ax.grid(True, linestyle='--', alpha=0.3, color='gray')
            ax.xaxis.set_major_locator(
                mdates.WeekdayLocator(interval=2, byweekday=mdates.MO))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

            # 優化刻度標籤
            plt.setp(ax.xaxis.get_majorticklabels(),
                        rotation=45,
                        fontsize=12,
                        weight='bold')
            ax.tick_params(axis='both', labelsize=12)
            ax2.tick_params(axis='both', labelsize=12)
            
            
            # 在設定 y 軸範圍之前加入這些除錯資訊
            print(f"force_to_plot statistics:")
            print(force_to_plot.describe())
            print(f"NaN values in force_to_plot: {force_to_plot.isna().sum()}")
            print(f"Inf values in force_to_plot: {np.isinf(force_to_plot).sum()}")


            # 動態調整y軸範圍
            force_max = force_to_plot.max()
            force_min = force_to_plot.min()

            if pd.isna(force_max) or pd.isna(force_min):
                # 如果有 NaN 值,嘗試使用 fillna 處理
                force_to_plot_clean = force_to_plot.fillna(method='ffill').fillna(method='bfill')
                force_max = force_to_plot_clean.max()
                force_min = force_to_plot_clean.min()

            if pd.isna(force_max) or pd.isna(force_min) or np.isinf(force_max) or np.isinf(force_min):
                # 如果還是有 NaN 或 Inf,使用預設值
                ax2.set_ylim(-3, 3)
            else:
                max_force = max(abs(force_max), abs(force_min))
                ax2.set_ylim(-max_force * 1.1, max_force * 1.1)

            # 美化圖表邊框
            ax.spines['top'].set_visible(False)
            ax2.spines['top'].set_visible(False)

            # 添加最新值標註
            last_date = dates[-1]
            ax2.annotate(f'{latest_force:.2f}σ',
                            xy=(last_date, latest_force),
                            xytext=(10, 0),
                            textcoords='offset points',
                            fontsize=12,
                            weight='bold')

            # 添加PR值變化趨勢標註
            if abs(pr_change) > 5:  # 只有當PR值變化顯著時才添加標註
                ax2.annotate(f'PR變化: {pr_change:+.1f}',
                                xy=(last_date, latest_force),
                                xytext=(10, 20),
                                textcoords='offset points',
                                fontsize=12,
                                weight='bold',
                                color=pr_color)

        except Exception as e:
            print(f"Error in plot_broker_analysis: {e}")
            raise



# =========================================broker=====================================================
 
 
    def plot_broker_analysis2(self, ax, stock_id):
        """
        繪製券商買賣超第一張圖：股價和買賣超對比
        加強自適應放大邏輯，使細微變化更加明顯
        CVD折線圖和柱狀圖同時放大，保持一致比例
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            matplotlib 圖表物件
        stock_id : str or int
            股票代碼
        """
        try:
            # 1. 基礎函數定義
            def safe_process_series(series):
                if series is None or series.empty:
                    return pd.Series()
                return series.fillna(0)
            
            # 2. 數據獲取和預處理
            if not isinstance(stock_id, str):
                stock_id = str(stock_id)
                        
            # 獲取基礎數據
            buy_vol = safe_process_series(data.get('etl:broker_transactions:top15_buy'))
            sell_vol = safe_process_series(data.get('etl:broker_transactions:top15_sell'))
            close_price = safe_process_series(data.get("price:收盤價"))
            days = 60

            # 數據處理
            stock_buy = buy_vol[stock_id]
            stock_sell = sell_vol[stock_id]
            stock_price = close_price[stock_id]
            net_vol = (stock_buy - stock_sell)*stock_price
        
            # 取最近期間數據
            recent_data = net_vol.tail(days)
            stock_price = stock_price.tail(days)

            # 計算買賣反轉點
            buy_reversal = (net_vol > 0) & (net_vol.shift(1) < 0) & (stock_price > stock_price.rolling(5).mean())
            sell_reversal = (net_vol < 0) & (net_vol.shift(1) > 0) & (stock_price < stock_price.rolling(5).mean())

            # 3. 繪製圖形：股價和買賣超
            ax_twin = ax.twinx()

            # 股價曲線
            ax.plot(range(len(stock_price)), stock_price, 'b-', linewidth=2, label='股價')
            ax.set_ylabel('股\n價', fontsize=20, rotation=0)
            ax.yaxis.set_label_coords(-0.02, 0.5)
            
            # 添加股價網格線
            ax.grid(True, alpha=0.2)
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            
            # 計算不同時間週期的CVD (先計算原始CVD)
            cvd_60d_original = recent_data.cumsum()
            
            # 計算20日CVD (從最近60天數據中取最後20天重新計算)
            recent_data_20d = recent_data.iloc[-20:]
            cvd_20d_original = recent_data_20d.cumsum()
            
            # 計算5日CVD
            recent_data_5d = recent_data.iloc[-5:]
            cvd_5d_original = recent_data_5d.cumsum()
            
            # ========= 改進的自適應放大縮小邏輯 =========
            
            # 計算平均交易價格，用於標準化買賣超金額
            avg_price = stock_price.mean()
            
            # 計算買賣超佔股價比例
            recent_ratio = recent_data / avg_price
            
            # 檢查原始數據區間
            data_range = recent_ratio.max() - recent_ratio.min()
            original_data_range = data_range  # 保存原始範圍供報告使用
            
            # 更精細的放大係數劃分，尤其加強對細小變動的放大
            if data_range < 0.05:  # 極度微小變動 (<0.05%)
                scale_factor = 200  # 放大 200 倍
            elif data_range < 0.1:  # 極小變動 (<0.1%)
                scale_factor = 150  # 放大 150 倍
            elif data_range < 0.2:  # 非常小變動 (<0.2%)
                scale_factor = 100  # 放大 100 倍
            elif data_range < 0.5:  # 很小變動 (<0.5%)
                scale_factor = 50   # 放大 50 倍
            elif data_range < 1.0:  # 小變動 (<1%)
                scale_factor = 25   # 放大 25 倍
            elif data_range < 2.0:  # 中等變動 (<2%)
                scale_factor = 15   # 放大 15 倍
            elif data_range < 3.0:  # 較大變動 (<3%)
                scale_factor = 8    # 放大 8 倍
            elif data_range < 5.0:  # 大變動 (<5%)
                scale_factor = 5    # 放大 5 倍
            else:                   # 巨大變動 (≥5%)
                scale_factor = 2    # 放大 2 倍，確保即使大變動也有一定視覺效果
            
            # 根據比例係數調整買賣超顯示值
            display_recent_data = recent_data * scale_factor
            
            # 同樣對CVD數據應用放大係數
            cvd_60d = cvd_60d_original * scale_factor
            cvd_20d = cvd_20d_original * scale_factor
            cvd_5d = cvd_5d_original * scale_factor
            
            # 繪製CVD線圖 (使用放大後的CVD數據)
            ax_twin.plot(range(len(cvd_60d)), cvd_60d, 
                        color='orange', 
                        linewidth=2,
                        label=f'60日CVD' + (f'(×{scale_factor})' if scale_factor > 1 else ''),
                        zorder=3,
                        alpha=0.8)
                        
            ax_twin.plot(range(days-20, days), cvd_20d.values, 
                        color='purple', 
                        linewidth=2, 
                        label=f'20日CVD' + (f'(×{scale_factor})' if scale_factor > 1 else ''), 
                        zorder=4,
                        alpha=0.8)
                        
            ax_twin.plot(range(days-5, days), cvd_5d.values, 
                        color='red', 
                        linewidth=2, 
                        linestyle='--',
                        label=f'5日CVD' + (f'(×{scale_factor})' if scale_factor > 1 else ''), 
                        zorder=5,
                        alpha=1)
            
            # 買賣超柱狀圖（使用調整後的數值）
            colors = ['red' if x >= 0 else 'green' for x in display_recent_data]
            bars = ax_twin.bar(range(len(display_recent_data)), display_recent_data, color=colors, alpha=0.3, 
                    label='買賣超' + (f'(×{scale_factor})' if scale_factor > 1 else ''))
            
            # 設置一個合理的 y 軸範圍，確保放大後的圖形能充分顯示
            # 同時考慮柱狀圖和折線圖的範圍
            y_max_bars = max(abs(display_recent_data.max()), abs(display_recent_data.min()))
            y_max_cvd = max(abs(cvd_60d.max()), abs(cvd_60d.min()))
            y_max = max(y_max_bars, y_max_cvd)
            y_buffer = y_max * 0.2  # 增加20%的緩衝空間
            ax_twin.set_ylim(-y_max - y_buffer, y_max + y_buffer)
            
            # 更新 y 軸標籤，添加放大信息
            y_label_text = f'買\n賣\n超\n金\n額\n(元)'
            if scale_factor > 1:
                y_label_text += f'\n(×{scale_factor})'
            ax_twin.set_ylabel(y_label_text, fontsize=18, rotation=0)
            ax_twin.yaxis.set_label_coords(1.03, 0.5)
            
            # 標記買入和賣出反轉點（使用調整後的數據來定位）
            for i, (buy, sell) in enumerate(zip(buy_reversal.tail(days), sell_reversal.tail(days))):
                if buy:
                    ax_twin.annotate('買', 
                        xy=(i, display_recent_data.iloc[i]), 
                        xytext=(i, display_recent_data.iloc[i] + abs(display_recent_data.max() * 0.4)),
                        arrowprops=dict(facecolor='blue', arrowstyle='->'), 
                        fontsize=12, 
                        color='blue',
                        ha='center'
                    )
                if sell:
                    ax_twin.annotate('賣', 
                        xy=(i, display_recent_data.iloc[i]), 
                        xytext=(i, display_recent_data.iloc[i] + abs(display_recent_data.max() * 0.2)),
                        arrowprops=dict(facecolor='purple', arrowstyle='->'), 
                        fontsize=12, 
                        color='purple',
                        ha='center'
                    )

            # 設置買賣超y軸的次要刻度
            ax_twin.yaxis.set_minor_locator(AutoMinorLocator())

            # 買賣超累積量面積圖 - 使用放大後的60日CVD
            ax_twin.fill_between(range(len(cvd_60d)), 0, cvd_60d, 
                                color='orange', alpha=0.1, 
                                label='累積買賣超' + (f'(×{scale_factor})' if scale_factor > 1 else ''))
            
            # 更新圖例位置和大小
            ax_twin.legend(loc='upper left', fontsize=20)

            # 相關性分析
            correlation = recent_data.corr(stock_price)
            
            # 分析資金流向狀態 (使用原始未放大數據進行判斷)
            latest_5d_cvd = cvd_5d_original.iloc[-1]
            latest_20d_cvd = cvd_20d_original.iloc[-1]
            latest_60d_cvd = cvd_60d_original.iloc[-1]
            
            # 判斷資金流向趨勢
            if latest_5d_cvd > 0 and latest_20d_cvd > 0 and latest_60d_cvd > 0:
                flow_trend = "多頭資金持續流入"
            elif latest_5d_cvd < 0 and latest_20d_cvd < 0 and latest_60d_cvd < 0:
                flow_trend = "空頭資金持續流出"
            elif latest_5d_cvd > 0 > latest_60d_cvd:
                flow_trend = "短期資金開始流入"
            elif latest_5d_cvd < 0 < latest_60d_cvd:
                flow_trend = "短期資金開始流出"
            else:
                flow_trend = "資金流向混亂"
            
            # 判斷買賣超強度 (使用原始未放大數據)
            recent_strength = abs(recent_data.iloc[-5:]).mean()
            historical_strength = abs(recent_data).mean()
            if recent_strength > historical_strength * 1.5:
                strength_text = "，力道強勁"
            elif recent_strength < historical_strength * 0.5:
                strength_text = "，力道減弱"
            else:
                strength_text = ""
            
            # 設置標題（包含分析結論）- 加強放大信息的顯示
            title = f'前15大券商買賣超資金流向分析 - {flow_trend}{strength_text}\n'
            if scale_factor > 1:
                title += f'所有數據已放大{scale_factor}倍 (原始變動範圍:{original_data_range:.4f}%) | '
            title += f'相關性:{correlation:.2f}'
            
            ax.set_title(title, fontsize=35, pad=15)

            # 設置日期刻度
            dates = stock_price.index.strftime('%Y-%m-%d')
            show_dates = [''] * len(dates)
            for i in range(0, len(dates), 5):
                show_dates[i] = dates[i]

            ax.set_xticks(range(0, len(dates), 5))
            ax.set_xticklabels(show_dates[::5], rotation=45, fontsize=18)

            # 如果放大係數很大，增加說明文字
            if scale_factor >= 50:
                ax_twin.text(0.02, 0.97, 
                        f"變動微小 ({original_data_range:.4f}%),\n所有數據已放大{scale_factor}倍以便觀察", 
                        transform=ax_twin.transAxes,
                        fontsize=20, 
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                        zorder=10)

            return True

        except Exception as e:
            logger.error(f"繪製券商買賣超分析圖表時發生錯誤: {str(e)}")
            ax.clear()
            ax.text(0.5, 0.5,
                    f'繪製圖表時發生錯誤:\n{str(e)}',
                    ha='center',
                    va='center',
                    transform=ax.transAxes)
            return False





        
        
    def plot_broker_analysis3(self, ax, stock_id):
        """
        繪製券商買賣超第二張圖：Z分數趨勢分析，加強放大效果
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            matplotlib 圖表物件
        stock_id : str or int
            股票代碼
        """
        try:
            def safe_process_series(series):
                if series is None or series.empty:
                    return pd.Series()
                return series.fillna(0)
            
            def calculate_trend_persistence(z_scores, window=5):
                direction = np.sign(z_scores)
                persistence = pd.Series(0, index=z_scores.index)
                current_count = 0
                current_direction = 0
                
                for i, value in enumerate(direction):
                    if i == 0 or value != current_direction:
                        current_count = 1
                        current_direction = value
                    else:
                        current_count += 1
                    persistence.iloc[i] = current_count
                
                strength = z_scores.abs() * (persistence / window)
                return persistence, strength

            # 數據獲取和預處理
            if not isinstance(stock_id, str):
                stock_id = str(stock_id)
                        
            buy_vol = safe_process_series(data.get('etl:broker_transactions:top15_buy'))
            sell_vol = safe_process_series(data.get('etl:broker_transactions:top15_sell'))
            days = 60

            stock_buy = buy_vol[stock_id]
            stock_sell = sell_vol[stock_id]
            net_vol = stock_buy - stock_sell
            
            # Z分數計算
            rolling_mean = net_vol.rolling(days).mean()
            rolling_std = net_vol.rolling(days).std()
            
            z_score = pd.Series(0, index=net_vol.index)
            mask = rolling_std != 0
            z_score[mask] = (net_vol[mask] - rolling_mean[mask]) / rolling_std[mask]

            recent_z = z_score.tail(days)
            ma5 = recent_z.rolling(window=5).mean()
            ma20 = recent_z.rolling(window=20).mean()

            # 計算趨勢持續性指標
            persistence, strength = calculate_trend_persistence(recent_z)

            # 自適應放大邏輯 - 大幅加強放大效果
            z_range = recent_z.max() - recent_z.min()
            
            # 顯著提高放大係數，參考 plot_broker_analysis2 的放大策略
            if z_range < 0.1:  # 極小波動
                scale_factor = 100  # 放大100倍
            elif z_range < 0.3:  # 非常小波動
                scale_factor = 50   # 放大50倍
            elif z_range < 0.5:  # 很小波動
                scale_factor = 20   # 放大20倍
            elif z_range < 1.0:  # 小波動
                scale_factor = 10   # 放大10倍
            elif z_range < 1.5:  # 適中波動
                scale_factor = 5    # 放大5倍
            else:
                scale_factor = 2    # 基礎放大2倍，讓變化更明顯
            
            # 創建用於顯示的放大Z分數
            display_z = recent_z * scale_factor
            
            # 繪製Z分數趨勢分析圖
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

            # Z分數柱狀圖
            bars = ax.bar(range(len(display_z)), display_z,
                        color=['red' if x >= 0 else 'green' for x in display_z], 
                        alpha=0.3, 
                        label='Z分數' + (f'(×{scale_factor})' if scale_factor > 1 else ''))

            # 設置動態的y軸範圍 - 進一步放大視覺效果
            y_max = max(2 * scale_factor, max(abs(display_z.max()), abs(display_z.min()))) * 1.2  # 增加20%的空間
            y_max = np.ceil(y_max * 10) / 10
            ax.set_ylim(-y_max, y_max)

            # 添加網格線和次要刻度
            ax.grid(True, which='both', alpha=0.2)
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.set_axisbelow(True)

            # 均線（也需要放大）
            ax.plot(range(len(ma5)), ma5 * scale_factor, 'r-', 
                label='5日均線' + (f'(×{scale_factor})' if scale_factor > 1 else ''), 
                linewidth=1.5)
            ax.plot(range(len(ma20)), ma20 * scale_factor, 'b-', 
                label='20日均線' + (f'(×{scale_factor})' if scale_factor > 1 else ''), 
                linewidth=1.5)

            # 設置標籤
            y_label_text = f'Z\n分\n數'
            if scale_factor > 1:
                y_label_text += f'\n(×{scale_factor})'
            ax.set_ylabel(y_label_text, fontsize=18, rotation=0)
            ax.yaxis.set_label_coords(-0.02, 0.5)

            # 標記Z分數值（顯示原始值，但位置使用放大後的坐標）
            for i, z in enumerate(recent_z):
                # 降低標記門檻使更多Z分數值顯示出來
                if abs(z) > recent_z.std() * 0.7:  # 降低門檻
                    ax.text(i, display_z.iloc[i], f'{z:.2f}', 
                            ha='center', 
                            va='bottom' if z > 0 else 'top',
                            fontsize=8,
                            color='red' if z > 0 else 'green')

            # 標記強烈趨勢區域
            for i in range(len(recent_z)):
                if abs(strength.iloc[i]) > strength.mean() + strength.std() * 0.7:  # 降低門檻
                    color = 'red' if recent_z.iloc[i] > 0 else 'green'
                    ax.fill_between([i-0.4, i+0.4], 
                                ax.get_ylim()[0], 
                                ax.get_ylim()[1], 
                                color=color, 
                                alpha=0.1)
                    
                    if persistence.iloc[i] >= 2:  # 降低門檻
                        ax.text(i, ax.get_ylim()[1]*0.8,
                                f'{persistence.iloc[i]}天',
                                ha='center',
                                color=color,
                                fontsize=10)

            # 趨勢強度判斷
            latest_persistence = persistence.iloc[-1]
            latest_z = recent_z.iloc[-1]

            if latest_persistence >= 5:
                trend_strength = '強勢{}超持續中'.format('買' if latest_z > 0 else '賣')
            elif latest_persistence >= 3:
                trend_strength = '溫和{}超趨勢'.format('買' if latest_z > 0 else '賣')
            else:
                trend_strength = '趨勢不明確'

            # 分析Z分數趨勢
            ma5_latest = ma5.iloc[-1] if not pd.isna(ma5.iloc[-1]) else 0
            ma20_latest = ma20.iloc[-1] if not pd.isna(ma20.iloc[-1]) else 0
            
            # 趨勢強度判斷
            if abs(latest_z) > 2:
                z_status = "極端" + ("買超" if latest_z > 0 else "賣超")
            elif abs(latest_z) > 1:
                z_status = "明顯" + ("買超" if latest_z > 0 else "賣超")
            else:
                z_status = "處於中性區間"
            
            # 趨勢方向判斷
            if ma5_latest > ma20_latest and ma5_latest > 0:
                trend_direction = "，買力增強中"
            elif ma5_latest < ma20_latest and ma5_latest < 0:
                trend_direction = "，賣力增強中"
            elif ma5_latest > 0 > ma20_latest:
                trend_direction = "，轉向買超"
            elif ma5_latest < 0 < ma20_latest:
                trend_direction = "，轉向賣超"
            else:
                trend_direction = ""
            
            # 設置標題（包含分析結論與放大資訊）
            title = f'Z分數趨勢分析 - {z_status}{trend_direction}\n'
            if scale_factor > 1:
                title += f'Z分數圖表已放大{scale_factor}倍 (原波動範圍:{z_range:.3f}) | '
            title += f'當前Z值:{latest_z:.2f}'
            ax.set_title(title, fontsize=35, pad=15)

            # 合併趨勢資訊和圖例
            legend_text = (f'趨勢狀態: {trend_strength}\n'
                        f'持續天數: {latest_persistence}天\n'
                        f'Z分數波動範圍: {z_range:.3f}')

            # 創建自定義圖例
            handles, labels = ax.get_legend_handles_labels()
            legend_label = labels + ['趨勢資訊']
            legend_handles = handles + [plt.Rectangle((0, 0), 1, 1, fc='none', ec='none')]
            legend_texts = labels + [legend_text]

            # 設置圖例
            legend = ax.legend(legend_handles, legend_texts, 
                            loc='upper left', 
                            fontsize=20, 
                            bbox_to_anchor=(0.02, 0.98),
                            bbox_transform=ax.transAxes)

            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('gray')
            legend.get_frame().set_alpha(0.8)

            # 設置日期刻度
            dates = recent_z.index.strftime('%Y-%m-%d')
            show_dates = [''] * len(dates)
            for i in range(0, len(dates), 5):
                show_dates[i] = dates[i]

            ax.set_xticks(range(0, len(dates), 5))
            ax.set_xticklabels(show_dates[::5], rotation=45, fontsize=18)

            return True

        except Exception as e:
            logger.error(f"繪製Z分數趨勢分析圖時發生錯誤: {str(e)}")
            ax.clear()
            ax.text(0.5, 0.5,
                    f'繪製圖表時發生錯誤:\n{str(e)}',
                    ha='center',
                    va='center',
                    transform=ax.transAxes)
            return False





    def plot_broker_analysis4(self, ax, stock_id):
        """
        券商力量視覺化 - 極致優化 (針對極小幅度變化)
        顯示券商淨買超/成交金額比例，加入統計學分析及持續性判斷
        增強功能：
        1. 券商連續買賣行為分析
        2. 股價分段建倉成本分析
        3. 股價與買超相關性分析
        4. 買入與賣出券商背離分析
        """
        try:
            # 1. 基礎函數定義
            def safe_process_series(series):
                if series is None or series.empty:
                    return pd.Series()
                return series.fillna(0)

            # 輔助函數：判斷條件是否連續N天成立
            def sustained_condition(series, condition_func, days=2):
                """檢查條件是否連續N天成立"""
                if len(series) < days:
                    return False
                return all(condition_func(series.iloc[-i]) for i in range(1, days+1))

            # 券商連續買賣判斷函數
            def analyze_broker_continuity(net_buys, days=5):
                """分析券商買賣連續性"""
                if len(net_buys) < days:
                    return {'current_streak': 0, 'continuity_score': 0,
                            'buy_percentage': 0, 'sell_percentage': 0}

                signs = np.sign(net_buys.tail(days))
                consecutive_buys = 0
                consecutive_sells = 0
                current_streak = 0

                # 計算當前連續買入或賣出天數
                for i in range(len(signs)-1, -1, -1):
                    if signs.iloc[i] > 0:  # 買入
                        if current_streak >= 0:
                            current_streak += 1
                        else:
                            break
                    elif signs.iloc[i] < 0:  # 賣出
                        if current_streak <= 0:
                            current_streak -= 1
                        else:
                            break
                    else:  # 不變
                        break

                # 分析整體連續性
                for i in range(1, len(signs)):
                    if signs.iloc[i] > 0 and signs.iloc[i-1] > 0:
                        consecutive_buys += 1
                    if signs.iloc[i] < 0 and signs.iloc[i-1] < 0:
                        consecutive_sells += 1

                continuity_score = (consecutive_buys +
                                    consecutive_sells) / max(1, (len(signs)-1))

                result = {
                    'current_streak': current_streak,  # 正數表示連續買入天數，負數表示連續賣出天數
                    'continuity_score': continuity_score,  # 0-1之間，越高表示連續性越強
                    'buy_percentage': sum(signs > 0) / len(signs),
                    'sell_percentage': sum(signs < 0) / len(signs)
                }

                return result

            # 價格區間券商買賣行為分析
            def analyze_cost_distribution(stock_price, net_buys, segments=10):
                """分析券商在不同價位區間的買賣行為"""
                if len(stock_price) < 30 or len(net_buys) < 30:
                    return None

                # 取最近可用數據
                length = min(60, len(stock_price), len(net_buys))
                recent_price = stock_price.tail(length)
                recent_buys = net_buys.tail(length)

                # 計算價格區間
                price_min = recent_price.min()
                price_max = recent_price.max()
                price_range = price_max - price_min
                if price_range == 0:  # 防止除以零
                    price_range = 0.01

                segment_size = price_range / segments

                # 創建價格區間
                price_segments = [(price_min + i*segment_size, price_min + (i+1)*segment_size)
                                for i in range(segments)]

                # 計算每個價格區間的買超情況
                segment_buys = []
                segment_days = []
                segment_avg_prices = []

                for low, high in price_segments:
                    mask = (recent_price >= low) & (recent_price < high)
                    segment_buys.append(recent_buys[mask].sum())
                    segment_days.append(mask.sum())
                    segment_avg_prices.append((low+high)/2)

                # 計算最新價格所在區間
                current_price = recent_price.iloc[-1]
                current_segment = 0
                for i, (low, high) in enumerate(price_segments):
                    if low <= current_price < high:
                        current_segment = i
                        break

                # 找出買入最多的區間
                max_buy_segment = np.argmax(segment_buys) if any(
                    buy > 0 for buy in segment_buys) else -1
                max_sell_segment = np.argmin(segment_buys) if any(
                    buy < 0 for buy in segment_buys) else -1

                return {
                    'price_segments': price_segments,
                    'segment_buys': segment_buys,
                    'segment_days': segment_days,
                    'current_segment': current_segment,
                    'max_buy_segment': max_buy_segment,
                    'max_sell_segment': max_sell_segment,
                    'segment_avg_prices': segment_avg_prices
                }

            # 股價與買超相關性分析
            def analyze_price_buy_correlation(stock_price, net_buys, window=60):
                """分析股價與買超的相關性"""
                if len(stock_price) < 10 or len(net_buys) < 10:
                    return {'long_term_correlation': 0, 'recent_correlation': 0, 'avg_price_change_after_buy': 0}

                # 取最近可用數據
                length = min(window, len(stock_price), len(net_buys))
                recent_price = stock_price.tail(length)
                recent_buys = net_buys.tail(length)

                # 計算相關係數
                # 使用try-except避免因數據問題造成錯誤
                try:
                    correlation = recent_price.corr(recent_buys)
                except:
                    correlation = 0

                # 計算買超後股價表現
                price_changes = []
                for i in range(len(recent_buys)-5):
                    if recent_buys.iloc[i] > 0:  # 買超日
                        price_change = recent_price.iloc[i +
                                                        5] / recent_price.iloc[i] - 1
                        price_changes.append(price_change)

                avg_price_change = np.mean(price_changes) if price_changes else 0

                # 計算最近10天趨勢
                recent_length = min(10, length)
                try:
                    recent_correlation = recent_price.tail(
                        recent_length).corr(recent_buys.tail(recent_length))
                except:
                    recent_correlation = 0

                return {
                    'long_term_correlation': correlation,
                    'recent_correlation': recent_correlation,
                    'avg_price_change_after_buy': avg_price_change
                }

            # 券商背離分析函數 (使用總買賣量替代明細)
            def analyze_simple_divergence(buy_vol, sell_vol, days=5):
                """使用買賣量分析券商背離情況"""
                recent_buy = buy_vol.tail(days)
                recent_sell = sell_vol.tail(days)

                # 計算買入和賣出的變化趨勢
                buy_trend = recent_buy.iloc[-1] / \
                    recent_buy.iloc[0] if recent_buy.iloc[0] != 0 else 1
                sell_trend = recent_sell.iloc[-1] / \
                    recent_sell.iloc[0] if recent_sell.iloc[0] != 0 else 1

                # 計算買賣背離程度 (1表示完全背離, 0表示完全一致)
                if buy_trend > 1 and sell_trend > 1:  # 都增加
                    divergence_score = 0
                    divergence_type = "一致看多"
                elif buy_trend < 1 and sell_trend < 1:  # 都減少
                    divergence_score = 0
                    divergence_type = "一致看空"
                elif buy_trend > 1 and sell_trend < 1:  # 買入增加，賣出減少
                    divergence_score = (buy_trend - 1) + (1 - sell_trend)
                    divergence_type = "極度看多"
                elif buy_trend < 1 and sell_trend > 1:  # 買入減少，賣出增加
                    divergence_score = (1 - buy_trend) + (sell_trend - 1)
                    divergence_type = "極度看空"
                else:
                    divergence_score = 0
                    divergence_type = "中性"

                # 計算買賣比例變化
                buy_sell_ratio_start = recent_buy.iloc[0] / \
                    recent_sell.iloc[0] if recent_sell.iloc[0] != 0 else 1
                buy_sell_ratio_end = recent_buy.iloc[-1] / \
                    recent_sell.iloc[-1] if recent_sell.iloc[-1] != 0 else 1
                ratio_change = buy_sell_ratio_end / \
                    buy_sell_ratio_start if buy_sell_ratio_start != 0 else 1

                return {
                    'divergence_score': divergence_score,
                    'divergence_type': divergence_type,
                    'buy_trend': buy_trend,
                    'sell_trend': sell_trend,
                    'buy_sell_ratio_change': ratio_change
                }

            # 2. 數據獲取和預處理
            if not isinstance(stock_id, str):
                stock_id = str(stock_id)

            # 獲取基礎數據
            buy_vol = safe_process_series(
                data.get('etl:broker_transactions:top15_buy'))
            sell_vol = safe_process_series(
                data.get('etl:broker_transactions:top15_sell'))
            close_price = safe_process_series(data.get("price:收盤價"))
            amt = safe_process_series(data.get("price:成交金額"))
            days = 60

            # 數據處理
            stock_buy = buy_vol[stock_id]
            stock_sell = sell_vol[stock_id]
            stock_price = close_price[stock_id]
            daily_amt = amt[stock_id]

            # 計算買賣超金額
            net_amount = (stock_buy - stock_sell) * stock_price

            # 計算買賣超佔成交金額比例
            ratio = (net_amount / daily_amt) * 100
            recent_ratio = ratio.tail(days)

            # 計算移動平均
            ma5 = recent_ratio.rolling(window=5).mean()
            ma20 = recent_ratio.rolling(window=20).mean()

            # 3. 執行新增的分析
            continuity_results = analyze_broker_continuity(recent_ratio)
            cost_results = analyze_cost_distribution(stock_price, net_amount)
            correlation_results = analyze_price_buy_correlation(
                stock_price, net_amount)
            divergence_results = analyze_simple_divergence(stock_buy, stock_sell)

            # 4. 繪製圖形
            ax.clear()

            # === 極致視覺化設計 ===

            # 1. **Y 軸範圍自適應調整** (核心優化)
            data_range = recent_ratio.max() - recent_ratio.min()
            if data_range < 0.1:  # 數據範圍極小，小於 0.1%
                scale_factor = 100  # 放大 100 倍
            elif data_range < 0.5:  # 數據範圍很小，小於 0.5%
                scale_factor = 20  # 放大 20 倍
            elif data_range < 1:  # 如果數據範圍小於 1%，則放大顯示
                scale_factor = 10  # 放大 10 倍
            elif data_range < 3:  # 如果數據範圍小於 3%，則放大顯示
                scale_factor = 5  # 放大 5 倍
            else:
                scale_factor = 1  # 否則不放大

            scaled_ratio = recent_ratio * scale_factor
            scaled_ma5 = ma5 * scale_factor
            scaled_ma20 = ma20 * scale_factor

            max_abs_ratio = abs(scaled_ratio).max()
            y_max = max(max_abs_ratio * 1.1, 1)  # 確保至少顯示到 +/- 1% (放大後的數值)
            y_min = -y_max
            ax.set_ylim(y_min, y_max)

            # 2. **繪製買賣力道柱狀圖**
            bar_width = 0.6
            for i, date in enumerate(scaled_ratio.index):
                ratio_value = scaled_ratio[date]
                if ratio_value >= 0:
                    color = '#FF4500'  # 鮮豔的橘紅色，代表買超
                else:
                    color = '#32CD32'  # 鮮豔的草綠色，代表賣超
                ax.bar(date, ratio_value, width=bar_width, color=color, alpha=0.8)

            # 3. **繪製移動平均線**
            ax.plot(scaled_ma5.index, scaled_ma5, color='#1E90FF',
                    linewidth=2, label='5日均線')  # 鮮明的藍色
            ax.plot(scaled_ma20.index, scaled_ma20, color='#8A2BE2',
                    linewidth=2, label='20日均線')  # 神秘的紫色

            # 4. **繪製水平零軸**
            ax.axhline(0, color='black', linestyle='--', linewidth=1)

            # 5. **標記最新數據點**
            last_date = scaled_ratio.index[-1]
            last_ratio = scaled_ratio.iloc[-1]
            ax.scatter(last_date, last_ratio, color='gold', s=100,
                    edgecolors='black', linewidth=0.5, zorder=5)

            # 6. **統計學分析**
            # 6.1 Z-Score計算 - 顯示當前值的異常程度
            mean_ratio = recent_ratio.mean()
            std_ratio = recent_ratio.std()
            if std_ratio != 0:  # 防止除以零
                z_score = (recent_ratio.iloc[-1] - mean_ratio) / std_ratio
            else:
                z_score = 0

            # 6.2 短期顯著性檢驗 - 最近10天的平均值是否顯著不為零
            recent_10_values = recent_ratio.iloc[-10:].dropna()
            if len(recent_10_values) >= 5:  # 確保有足夠樣本
                import scipy.stats as stats
                # 單樣本t檢驗，檢驗均值是否顯著不為零
                t_stat, p_value = stats.ttest_1samp(recent_10_values, 0)
                significant_trend = p_value < 0.05  # 95%信心水平
            else:
                significant_trend = False
                t_stat = 0
                p_value = 1.0

            # 6.3 分析資金流向的持續性 - 最近資金流向連續性
            if len(recent_ratio) >= 5:
                # 計算最近5天資金流向的符號連續性
                recent_5_signs = np.sign(recent_ratio.tail(5))
                sign_changes = sum(abs(
                    recent_5_signs[i] - recent_5_signs[i-1]) > 0 for i in range(1, len(recent_5_signs)))
                high_persistence = sign_changes <= 1  # 最多只有1次變向表示高持續性
            else:
                high_persistence = False

            # 7. **判斷資金流向與趨勢 (加入持續性與統計學觀點)**
            # 7.1 均線交叉判斷 - 確保連續兩天以上的趨勢
            ma5_above_ma20 = sustained_condition(
                pd.DataFrame({'ma5': ma5, 'ma20': ma20}),
                lambda x: x['ma5'] > x['ma20'],
                days=2  # 連續2天確認
            )

            ma5_below_ma20 = sustained_condition(
                pd.DataFrame({'ma5': ma5, 'ma20': ma20}),
                lambda x: x['ma5'] < x['ma20'],
                days=2  # 連續2天確認
            )

            # 7.2 趨勢方向判斷
            if ma5_above_ma20:
                trend_direction = "多頭"
            elif ma5_below_ma20:
                trend_direction = "空頭"
            else:
                trend_direction = "盤整"

            # 7.3 資金流向判斷 (結合統計學觀點)
            recent_5_values = recent_ratio.iloc[-5:]
            recent_5_sum = recent_5_values.sum()
            ma5_trend = ma5.iloc[-1] - ma5.iloc[-5] if len(ma5) >= 5 else 0

            # 7.4 結合統計學和持續性的資金流向判斷
            if significant_trend and t_stat > 0 and high_persistence and ma5_trend > 0:
                flow_status = "資金顯著持續流入"
            elif significant_trend and t_stat > 0:
                flow_status = "資金顯著流入"
            elif significant_trend and t_stat < 0 and high_persistence and ma5_trend < 0:
                flow_status = "資金顯著持續流出"
            elif significant_trend and t_stat < 0:
                flow_status = "資金顯著流出"
            # 非顯著情況下的資金流向判斷
            elif ma5_trend > 0 and recent_5_sum > 0:
                flow_status = "資金趨勢向上"
            elif ma5_trend < 0 and recent_5_sum < 0:
                flow_status = "資金趨勢向下"
            elif abs(z_score) > 1.5:  # 使用z-score判斷異常值
                flow_status = f"資金異常波動 (Z={z_score:.1f})"
            else:
                flow_status = "資金流向中性"

            # 8. **組織券商連續買賣行為分析文字**
            continuity_info = ""
            if continuity_results['current_streak'] > 0:
                continuity_info = f"連買{continuity_results['current_streak']}天"
            elif continuity_results['current_streak'] < 0:
                continuity_info = f"連賣{abs(continuity_results['current_streak'])}天"

            # 9. **組織價格區間分析文字**
            # 組織價格區間分析文字
            price_info = ""
            if cost_results is not None and cost_results['max_buy_segment'] >= 0:
                # 獲取主力買入最多的價格區間
                buy_low, buy_high = cost_results['price_segments'][cost_results['max_buy_segment']]
                buy_segment = cost_results['max_buy_segment']
                segment_buys = cost_results['segment_buys']
                segment_days = cost_results['segment_days']
                current_price = stock_price.iloc[-1]
                
                # 計算該區間的買入強度
                total_buy = sum([b for b in segment_buys if b > 0])
                buy_strength = segment_buys[buy_segment] / total_buy * 100 if total_buy > 0 else 0
                
                # 判斷區間位置特性
                price_position = ""
                if current_price < buy_low:
                    price_position = "現價低於主力成本"
                elif current_price > buy_high:
                    price_position = "現價高於主力成本"
                else:
                    price_position = "現價處於主力成本區間"
                
                # 分析該區間的交易天數佔比
                days_percentage = segment_days[buy_segment] / sum(segment_days) * 100 if sum(segment_days) > 0 else 0
                
                # 判斷是否為近期建倉
                recent_days = 10  # 定義「近期」為最近10天
                recent_price = stock_price.tail(recent_days)
                recent_in_segment = sum((recent_price >= buy_low) & (recent_price <= buy_high))
                is_recent = recent_in_segment >= (recent_days * 0.3)  # 如果最近30%的時間都在此區間，視為近期建倉
                
                # 組合完整的價格區間分析
                price_info = f"主力買入${buy_low:.1f}-${buy_high:.1f}"
                price_info += f" | {price_position}"
                price_info += f" | 買入強度:{buy_strength:.1f}%"
                price_info += f" | 交易佔比:{days_percentage:.1f}%"
                
                if is_recent:
                    price_info += " | 近期建倉"
                
                # 判斷當前價格區間與主力買入區間的關係
                if cost_results['current_segment'] == cost_results['max_buy_segment']:
                    price_info += " | 與散戶成本相同"


            # 10. **組織相關性分析文字**
            corr_info = ""
            long_corr = correlation_results['long_term_correlation']
            recent_corr = correlation_results['recent_correlation']
            if abs(long_corr) > 0.5 or abs(recent_corr) > 0.5:
                corr_direction = "正" if (long_corr + recent_corr) > 0 else "負"
                corr_info = f"股價買超{corr_direction}相關"

            # 11. **組織券商背離分析文字**
            divergence_info = ""
            if divergence_results['divergence_score'] > 0.3:
                divergence_info = f"{divergence_results['divergence_type']}"

            # 12. **建立綜合標題**
            # 統計顯著性信息
            stat_info = ""
            if significant_trend:
                stat_info = f" | 顯著性:95%+ | Z值:{z_score:.1f}"

            # 主標題 - 基本資訊
            title = f"{stock_id} 券商淨買超/成交額比例 | {trend_direction}趨勢 | {flow_status}{stat_info} | 最新:{recent_ratio.iloc[-1]:.2f}%"
            
            # 副標題 - 四大分析結果
            subtitle = ""
            analysis_parts = []
            
            # 只加入有意義的分析結果
            if continuity_info:
                analysis_parts.append(continuity_info)
            if price_info:
                analysis_parts.append(price_info)
            if corr_info:
                analysis_parts.append(corr_info)
            if divergence_info:
                analysis_parts.append(divergence_info)
                
            # 組合副標題
            if analysis_parts:
                subtitle = " | ".join(analysis_parts)
                
            # 完整標題
            full_title = title
            if subtitle:
                full_title += "\n" + subtitle
                
            # 加入放大倍率說明
            if scale_factor > 1:
                full_title += f" | ({scale_factor}倍放大)"

            ax.set_title(full_title, fontsize=35, fontweight='bold')

            # 13. **設定軸標籤**
            ax.set_ylabel("淨\n買\n超\n額\n/\n成\n交\n額\n比\n例\n(%)",
                        fontsize=20, rotation=0, labelpad=20)

           # 14. **格式調整**
            ax.tick_params(axis='x', rotation=45, labelsize=20)  # 放大X軸日期字體
            ax.tick_params(axis='y', labelsize=14)
            ax.grid(True, linestyle=':', alpha=0.6)

            # 新增：添加股價線(黑色虛線)
            # 創建次要Y軸
            ax2 = ax.twinx()

            # 取得與recent_ratio相同時間範圍的股價數據
            recent_stock_price = stock_price.tail(days)

            # 繪製股價走勢線（黑色虛線）
            ax2.plot(recent_stock_price.index, recent_stock_price.values, 
                    color='black', linestyle='--', linewidth=1.5, label='股價')

            # 設置次要Y軸的範圍和格式
            min_price = recent_stock_price.min()
            max_price = recent_stock_price.max()
            price_range = max_price - min_price
            # 避免股價範圍過小導致顯示異常
            if price_range < 0.01:
                price_range = recent_stock_price.mean() * 0.01
            ax2.set_ylim(min_price - price_range*0.1, max_price + price_range*0.1)

            # 設置次要Y軸的標籤和格式
            ax2.set_ylabel('股\n價', fontsize=20, rotation=0, labelpad=20)
            ax2.tick_params(axis='y', labelsize=14)
            ax2.yaxis.label.set_color('black')
            ax2.tick_params(axis='y', colors='black')
            ax2.spines['right'].set_color('black')

            # 設置圖例（合併兩組圖例）
            handles, labels = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(handles + handles2, labels + labels2, loc='upper left', 
                    fontsize=18, frameon=True, facecolor='white', edgecolor='grey')


            # 15. **移除邊框**
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)

            # 16. **增加 Y 軸的百分比標示**
            import matplotlib.ticker as mtick
            ax.yaxis.set_major_formatter(
                mtick.PercentFormatter(decimals=1, symbol='%'))

            # 17. **設定X軸每五日顯示一次**
            import matplotlib.dates as mdates
            date_list = list(scaled_ratio.index)
            if len(date_list) > 0:
                plt_dates = [date_list[0]]  # 第一天
                for i in range(4, len(date_list), 5):  # 每5天顯示一次
                    if i < len(date_list):
                        plt_dates.append(date_list[i])
                if date_list[-1] not in plt_dates:
                    plt_dates.append(date_list[-1])  # 最後一天

                # 設置顯示這些日期
                ax.set_xticks(plt_dates)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

            # 18. **添加統計學資訊到圖表**
            if significant_trend:
                sign_color = 'red' if t_stat > 0 else 'green'
                stats_text = f"統計顯著 (p={p_value:.3f})" if p_value < 0.05 else "無顯著趨勢"
                ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                    fontsize=15, color=sign_color,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='grey'))

            return True

        except Exception as e:
            logger.error(f"繪製券商力量柱狀圖時發生錯誤: {str(e)}")
            ax.clear()
            ax.text(0.5, 0.5,
                    f'繪製圖表時發生錯誤:\n{str(e)}',
                    ha='center',
                    va='center',
                    transform=ax.transAxes)
            return False



# =======================================峰度圖==============================4


    def plot_trend_analysis(self, ax, stock_id):
        """繪製多重趨勢分析圖，整合長中短期趨勢和波動特徵
        
        Parameters:
            ax: matplotlib axes物件，繪圖區域
            stock_id: str，股票代碼
        """
        try:
            # 參數驗證
            if ax is None:
                raise ValueError("繪圖區域(ax)不能為None")

            if stock_id is None:
                raise ValueError("股票代碼不能為None")

            if not hasattr(self, 'data') or self.data is None:
                raise ValueError("數據對象未初始化")

            # 確保 data 是類的屬性
            data = self.data  # 使用類的 data 屬性
            # 1. 定義技術指標計算函數

            def wma(price, n):
                """指數加權移動平均"""
                return price.ewm(span=n).mean()

            def zlma(price, n):
                """零延遲移動平均"""
                lag = (n - 1) // 2
                series = 2 * price - price.shift(lag)
                return wma(series, n)

            def calculate_volatility(stock_price, window=20):
                """計算波動率並返回分類結果和動態閾值"""
                if stock_price is None or len(stock_price) < window:
                    raise ValueError(f"股價數據不足 {window} 個交易日")

                # 計算收益率
                returns = stock_price.pct_change().dropna()

                # 1. 傳統波動率計算
                rolling_vol = returns.rolling(window).std() * np.sqrt(240)

                # 2. 動態分位數計算
                def calculate_dynamic_quantiles(volatility_series):
                    recent_vol = volatility_series.iloc[-window:]  # 使用最近期間的數據
                    current_mean = recent_vol.mean()
                    historical_mean = volatility_series.expanding().mean().iloc[-1]

                    # 根據當前波動率水平調整分位數
                    if current_mean > historical_mean * 1.2:  # 高波動期
                        upper_quantile = 0.85
                        lower_quantile = 0.35
                    elif current_mean < historical_mean * 0.8:  # 低波動期
                        upper_quantile = 0.70
                        lower_quantile = 0.20
                    else:  # 正常期
                        upper_quantile = 0.75
                        lower_quantile = 0.25

                    return (volatility_series.quantile(upper_quantile),
                            volatility_series.quantile(lower_quantile))

                vol_quantile_75, vol_quantile_25 = calculate_dynamic_quantiles(
                    rolling_vol)
                latest_vol = rolling_vol.iloc[-1]

                def classify_volatility(vol):
                    base_result = {
                        'value': vol,
                        'quantile_75': vol_quantile_75,
                        'quantile_25': vol_quantile_25,
                        'rolling_vol': rolling_vol.iloc[-1]
                    }

                    # 極端值檢測（使用四分位距）
                    IQR = vol_quantile_75 - vol_quantile_25
                    extreme_threshold = vol_quantile_75 + 1.5 * IQR

                    if vol > extreme_threshold:
                        return {
                            **base_result,
                            'state': '極端波動',
                            'description': f'波動率 {vol:.1%}, 已超過極端閾值 {extreme_threshold:.1%}',
                            'risk_level': '極高'
                        }
                    elif vol > vol_quantile_75:
                        return {
                            **base_result,
                            'state': '高波動',
                            'description': f'波動率 {vol:.1%}, 高於75%分位數 {vol_quantile_75:.1%}',
                            'risk_level': '高'
                        }
                    elif vol < vol_quantile_25:
                        return {
                            **base_result,
                            'state': '低波動',
                            'description': f'波動率 {vol:.1%}, 低於25%分位數 {vol_quantile_25:.1%}',
                            'risk_level': '低'
                        }
                    else:
                        return {
                            **base_result,
                            'state': '正常波動',
                            'description': f'波動率 {vol:.1%}, 處於正常區間',
                            'risk_level': '中'
                        }

                volatility_state = classify_volatility(latest_vol)

                # 3. 改進的動態閾值計算
                def calculate_dynamic_threshold(vol_series, current_state):
                    base_threshold = vol_series * 100

                    # 根據風險等級調整因子
                    adjustment_factors = {
                        '極高': 2.0,
                        '高': 1.8,
                        '中': 1.5,
                        '低': 1.2
                    }

                    adjustment_factor = adjustment_factors[current_state['risk_level']]
                    dynamic_threshold = base_threshold * adjustment_factor

                    # 根據市場狀態調整閾值範圍
                    if current_state['risk_level'] in ['極高', '高']:
                        return dynamic_threshold.clip(lower=8, upper=20)
                    else:
                        return dynamic_threshold.clip(lower=5, upper=15)

                dynamic_threshold = calculate_dynamic_threshold(
                    rolling_vol, volatility_state)

                return volatility_state, dynamic_threshold

            # 在計算波動特徵部分，添加更詳細的分類

            def analyze_market_state(kurtosis, volatility_state, bias_250, bias_120, bias_60, dynamic_threshold):
                """
                根據峰度、波動率和偏離度進行更細緻的市場狀態分析
                """
                # 原有的峰度分類保持不變
                if kurtosis < -0.5:
                    kurt_state = {
                        'state': '扁平分布',
                        'description': '價格波動較為分散，市場分歧較大',
                        'risk_level': '中等'
                    }
                elif -0.5 <= kurtosis <= 0.5:
                    kurt_state = {
                        'state': '常態分布',
                        'description': '價格波動呈現正常市場特徵',
                        'risk_level': '低'
                    }
                elif 0.5 < kurtosis <= 3:
                    kurt_state = {
                        'state': '輕度尖峰',
                        'description': '價格波動開始集中，可能形成趨勢',
                        'risk_level': '中低'
                    }
                elif 3 < kurtosis <= 6:
                    kurt_state = {
                        'state': '中度尖峰',
                        'description': '價格波動高度集中，趨勢明顯',
                        'risk_level': '中高'
                    }
                else:
                    kurt_state = {
                        'state': '極度尖峰',
                        'description': '價格波動極度集中，可能存在異常',
                        'risk_level': '高'
                    }

                # 趨勢持續性分析
                if (bias_60 > bias_120 > bias_250):
                    trend_persistence = '上升趨勢加速'
                elif (bias_60 < bias_120 < bias_250):
                    trend_persistence = '下降趨勢加速'
                elif (bias_60 > bias_120 and bias_120 < bias_250):
                    trend_persistence = '短期反彈'
                elif (bias_60 < bias_120 and bias_120 > bias_250):
                    trend_persistence = '短期回檔'
                else:
                    trend_persistence = '趨勢不明確'

                # 趨勢強度分析
                trend_strength = 0
                for bias in [bias_250, bias_120, bias_60]:
                    if abs(bias) > dynamic_threshold:
                        trend_strength += 1

                # 市場狀態綜合分析
                if trend_strength >= 2 and volatility_state['risk_level'] == '高':
                    market_state = '過熱警告'
                elif trend_strength >= 2 and volatility_state['risk_level'] in ['低', '中']:
                    market_state = '健康趨勢'
                elif trend_strength == 1:
                    market_state = '趨勢發展中'
                else:
                    market_state = '盤整階段'

                # 風險警告等級
                risk_level = 0
                # 安全地檢查波動率狀態
                if volatility_state.get('risk_level') == '高':
                    risk_level += 2
                    # 使用 get 方法安全地訪問字典值，提供預設值
                    current_vol = volatility_state.get('value', 0)
                    vol_75 = volatility_state.get('quantile_75', 0)

                    if vol_75 > 0 and current_vol > vol_75 * 1.2:  # 避免除以零
                        risk_level += 1
                elif volatility_state.get('risk_level') == '中':
                    risk_level += 1

                if kurt_state['risk_level'] == '高':
                    risk_level += 2
                if market_state == '過熱警告':
                    risk_level += 1
                if trend_persistence in ['上升趨勢加速', '下降趨勢加速']:
                    risk_level += 1

                risk_warning = ['低風險', '警示', '高度警示', '極度警示'][min(risk_level, 3)]

                trend_analysis = []

                # 趨勢共振分析
                if all(bias > dynamic_threshold for bias in [bias_250, bias_120, bias_60]):
                    trend_analysis.append("多重趨勢共振(短中長期均處於超買區，建議降低持倉)")
                elif all(bias > 0 for bias in [bias_250, bias_120, bias_60]):
                    trend_analysis.append("全面多頭格局(短中長期均處於均線上方)")
                elif bias_250 > dynamic_threshold and bias_120 > dynamic_threshold:
                    trend_analysis.append("中長期趨勢強勁向上，短期可能出現回檔")
                elif bias_60 > dynamic_threshold and bias_120 > dynamic_threshold:
                    trend_analysis.append("短中期趨勢向上，注意長期壓力")
                elif all(bias < -dynamic_threshold for bias in [bias_250, bias_120, bias_60]):
                    trend_analysis.append("多重下跌信號(短中長期均處於超賣區，可考慮逢低布局)")
                elif all(bias < 0 for bias in [bias_250, bias_120, bias_60]):
                    trend_analysis.append("全面空頭格局(短中長期均處於均線下方)")
                elif bias_60 > 0 and bias_120 < 0 and bias_250 < 0:
                    trend_analysis.append("短期反彈，但中長期仍在下跌趨勢中")
                elif bias_60 < 0 and bias_120 > 0 and bias_250 > 0:
                    trend_analysis.append("短期回檔，但中長期仍在上升趨勢中")
                else:
                    trend_analysis.append("趨勢混沌，建議觀望")
                # 波動異常分析
                if abs(kurtosis) > 6:
                    trend_analysis.append("極度異常波動，建議暫停交易")
                elif abs(kurtosis) > 3:
                    trend_analysis.append("波動明顯擴大，建議降低倉位")
                elif abs(kurtosis) > 1.5:
                    trend_analysis.append("波動較大，需要謹慎操作")

                # 趨勢強度與速度分析
                if trend_persistence == '上升趨勢加速':
                    if bias_60 > dynamic_threshold * 1.5:
                        trend_analysis.append("趨勢加速上揚，但已達極度超買，注意回檔風險")
                    else:
                        trend_analysis.append("趨勢加速上揚，動能強勁")
                elif trend_persistence == '下降趨勢加速':
                    if bias_60 < -dynamic_threshold * 1.5:
                        trend_analysis.append("趨勢加速下跌，已達極度超賣，可能出現反彈")
                    else:
                        trend_analysis.append("趨勢加速下跌，建議觀望")
                elif trend_persistence == '短期反彈':
                    trend_analysis.append("出現短期反彈信號，關注能否突破中期趨勢")
                elif trend_persistence == '短期回檔':
                    trend_analysis.append("出現短期回檔，可能是買入機會")

                # 趨勢背離分析
                if bias_60 > bias_120 > bias_250:
                    trend_analysis.append("趨勢完美共振，多頭動能強勁")
                elif bias_60 < bias_120 < bias_250:
                    trend_analysis.append("趨勢完美共振，空頭壓力較大")
                elif bias_60 > 0 and bias_250 < 0:
                                    trend_analysis.append("短期反轉信號，關注能否突破長期趨勢")

                # 波動率狀態分析
                if volatility_state['risk_level'] == '高':
                    if volatility_state['value'] > volatility_state['quantile_75'] * 1.5:
                        trend_analysis.append("波動率異常高")
                    else:
                        trend_analysis.append("波動率偏高")
                elif volatility_state['risk_level'] == '低':
                    if trend_persistence in ['上升趨勢加速', '下降趨勢加速']:
                        trend_analysis.append("波動率低位，趨勢穩定，可適度加倉")

                # 添加明確的交易建議
                if market_state == '過熱警告' or risk_warning == '極度警示':
                    trade_action = '建議減倉保守'
                elif market_state == '健康趨勢' and all(bias > 0 for bias in [bias_250, bias_120, bias_60]):
                    trade_action = '建議做多進場'
                elif market_state == '健康趨勢' and all(bias < 0 for bias in [bias_250, bias_120, bias_60]):
                    trade_action = '建議做空進場'
                elif all(bias < -dynamic_threshold for bias in [bias_250, bias_120, bias_60]):
                    trade_action = '建議逢低布局'
                elif all(bias > dynamic_threshold for bias in [bias_250, bias_120, bias_60]):
                    trade_action = '建議減倉觀望'
                elif volatility_state['risk_level'] == '高' and kurt_state['state'] == '極度尖峰':
                    trade_action = '建議暫停交易'
                elif trend_persistence == '上升趨勢加速' and bias_60 > 0:
                    trade_action = '建議順勢做多'
                elif trend_persistence == '下降趨勢加速' and bias_60 < 0:
                    trade_action = '建議順勢做空'
                elif '趨勢混沌' in ' '.join(trend_analysis):
                    trade_action = '建議觀望為主'
                elif bias_60 > 0 and bias_120 < 0 and bias_250 < 0:
                    trade_action = '建議短線試做多'
                elif bias_60 < 0 and bias_120 > 0 and bias_250 > 0:
                    trade_action = '建議短線試做空'
                else:
                    trade_action = '建議觀望等待信號'

                return kurt_state, market_state, trend_persistence, risk_warning, volatility_state, trend_analysis, trade_action
            
            
            # 2. 數據準備
            if not isinstance(stock_id, str):
                stock_id = str(stock_id)
                
            close_price = data.get("price:收盤價")
            if close_price is None or close_price.empty:
                raise ValueError("無法獲取收盤價數據")
                
            stock_price = close_price[stock_id]
            
            # 3. 計算技術指標
            # 移動平均線
            sma_250 = stock_price.rolling(250).mean()  # 長期趨勢
            zlma_120 = zlma(stock_price, 120)         # 中期趨勢
            wma_60 = wma(stock_price, 60)             # 短期趨勢
            
            volatility_state, dynamic_threshold = calculate_volatility(stock_price)

            # 計算偏離度
            bias_250 = (stock_price / sma_250 - 1) * 100  # 長期偏離度
            bias_120 = (stock_price / zlma_120 - 1) * 100 # 中期偏離度
            bias_60 = (stock_price / wma_60 - 1) * 100    # 短期偏離度
            
        
            # 計算波動特徵
            kurtosis_250 = stock_price.pct_change().rolling(250).kurt()
            
            # 4. 準備繪圖數據
            plot_days = 120  # 顯示最近120天的數據
            bias_250_plot = bias_250.tail(plot_days)
            bias_120_plot = bias_120.tail(plot_days)
            bias_60_plot = bias_60.tail(plot_days)
            dynamic_threshold_plot = dynamic_threshold.tail(plot_days)
            dynamic_threshold_plot = dynamic_threshold_plot.reindex(bias_250_plot.index).fillna(method='ffill')

            # 5. 繪製主圖
            # 填充超買超賣區域
            ax.fill_between(bias_250_plot.index, bias_250_plot, 0, 
                            where=(bias_250_plot >= dynamic_threshold_plot),
                color='r', alpha=0.2, label='長期超買區')  # 改為紅色
            ax.fill_between(bias_250_plot.index, bias_250_plot, 0,
                where=(bias_250_plot <= -dynamic_threshold_plot),
                color='g', alpha=0.2, label='長期超賣區')  # 改為綠色
            
            # 繪製趨勢線
            ax.plot(bias_250_plot.index, bias_250_plot, 'g-', 
                    label='長期趨勢(250天)', linewidth=2)
            ax.plot(bias_120_plot.index, bias_120_plot, 'b-', 
                    label='中期趨勢(120天)', linewidth=2)
            ax.plot(bias_60_plot.index, bias_60_plot, 'r-', 
                    label='短期趨勢(60天)', linewidth=2)
            
            # 6. 標註重要點位
            # 標註轉折點
            latest_dynamic_threshold = dynamic_threshold.iloc[-1]

            # 設定最小變化閾值，防止過小變化被標記
            min_threshold = 1  # 最小閾值，單位是%
            threshold = max(latest_dynamic_threshold * 0.2, min_threshold)

            for i in range(2, len(bias_60_plot) - 2):  # 讓轉折點範圍更廣泛
                # 標註超買轉折點（高點）
                if (bias_60_plot.iloc[i-2] < bias_60_plot.iloc[i-1] < bias_60_plot.iloc[i] > bias_60_plot.iloc[i+1] > bias_60_plot.iloc[i+2]) and \
                (bias_60_plot.iloc[i] - bias_60_plot.iloc[i-1]) > threshold:
                    ax.annotate('超\n買\n轉\n折', 
                                xy=(bias_60_plot.index[i], bias_60_plot.iloc[i]),
                                xytext=(10, 10), textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                                arrowprops=dict(arrowstyle='->'),
                                fontsize=12)

                # 標註超賣轉折點（低點）
                elif (bias_60_plot.iloc[i-2] > bias_60_plot.iloc[i-1] > bias_60_plot.iloc[i] < bias_60_plot.iloc[i+1] < bias_60_plot.iloc[i+2]) and \
                    (bias_60_plot.iloc[i-1] - bias_60_plot.iloc[i]) > threshold:
                    ax.annotate('超\n賣\n轉\n折', 
                                xy=(bias_60_plot.index[i], bias_60_plot.iloc[i]),
                                xytext=(-30, -10), textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.5', fc='cyan', alpha=0.5),
                                arrowprops=dict(arrowstyle='->'),
                                fontsize=12)

            # 7. 繪製參考線，改為動態閾值
            ax.axhline(y=latest_dynamic_threshold, color='r', linestyle='--', alpha=0.5, label='超買警戒線', linewidth=2)
            ax.axhline(y=-latest_dynamic_threshold, color='g', linestyle='--', alpha=0.5, label='超賣警戒線', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.2, label='基準線', linewidth=2)
            
            # 8. 趨勢分析
            # 獲取最新數據
            latest_bias_250 = bias_250.iloc[-1]
            latest_bias_120 = bias_120.iloc[-1]
            latest_bias_60 = bias_60.iloc[-1]
            latest_kurtosis = kurtosis_250.iloc[-1]
            
            # 趨勢判斷
            # 8. 趨勢分析               
            kurt_state, market_state, trend_persistence, risk_warning, volatility_state, trend_analysis, trade_action = analyze_market_state(
                latest_kurtosis, 
                volatility_state, 
                latest_bias_250,
                latest_bias_120,
                latest_bias_60,
                latest_dynamic_threshold
            )
            
            # 9. 設置圖表標題和說明
            title = (f'{stock_id} 趨勢分析 | 【{trade_action}】\n'
                    f'市場狀態: {market_state} | 風險警告: {risk_warning}\n'
                    f'趨勢持續性: {trend_persistence} | 波動特徵: {kurt_state["state"]}\n'
                    f'波動率: {volatility_state["description"]}\n'
                    f'趨勢分析: {" | ".join(trend_analysis)}')

            # 使用annotation替代標題，保持整體居中但文字靠左對齊
            ax.set_title("")  # 清除原標題
            ax.annotate(title, xy=(0.5, 1), xycoords='axes fraction', 
                        xytext=(0, 20), textcoords='offset points',
                        ha='center', va='bottom', fontsize=30)

            ax.set_ylabel('偏\n離\n均\n線\n程\n度\n(%)', 
            fontsize=20, 
            rotation=0,  # 保持水平
            labelpad=30, # 增加到 60 使標籤往左移
            ha='right', # 右對齊
            va='center') # 垂直居中
            
            # 10. 添加說明文字
            explanation = (
            "圖表說明：\n"
            "1. 線條越往上代表股價越高於平均\n"
            "2. 超過上方紅線表示超買區域\n"
            "3. 低於下方綠線表示超賣區域\n"
            "4. 三條線同時向上是強勢信號\n"
        
                )
            ax.text(1.03, 0, explanation, 
                    transform=ax.transAxes, fontsize=20,
                    bbox=dict(facecolor='white', alpha=0.8))
            
            # 11. 圖表美化
            ax.grid(True, alpha=0.2)
            ax.legend(loc='upper left', fontsize=14, bbox_to_anchor=(1.02, 1))
            
            # 優化刻度顯示
            ax.xaxis.set_major_locator(plt.MultipleLocator(5))  # 每5天顯示一次
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))  # 設定日期格式為 月/日

            # 設置字體大小和旋轉角度
            for label in ax.get_xticklabels():
                label.set_rotation(45)  # 旋轉45度
                label.set_fontsize(20)  # 字體大小14
            
            # Y軸刻度優化
            ax.yaxis.set_major_locator(plt.MultipleLocator(10))
            for label in ax.get_yticklabels():
                label.set_fontsize(20)
            
            # 調整圖表邊距
            plt.subplots_adjust(right=0.85)
            
            return True
                
        except Exception as e:
            logger.error(f"繪製趨勢分析圖時發生錯誤 (股票代碼: {stock_id}, 日期: {stock_price.index[-1]}): {str(e)}")
            ax.clear()
            ax.text(0.5, 0.5, f'繪製圖表時發生錯誤:\n{str(e)}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=20, color='red')
            return False



# =======================================累積營收圖========看長期趨勢
#
# ============================

    def plot_revenue2(self, ax, stock_id):
        """
        繪製累積營收年增率分析圖，包含:
        - 累積營收柱狀圖（正負值分開顯示）
        - 正負成長區域填充
        - 移動平均線
        - 月均價
        - 趨勢指標和箭頭
        - 營收動能指標
        - 進階季節性分析
        - 綜合警示系統
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            繪圖用的axes物件
        stock_id : str
            股票代碼
            
        Returns:
        --------
        ax : matplotlib.axes.Axes
            繪製完成的axes物件
        """
        # 1. 顏色函數定義
        def get_positive_color(value):
            if value >= 30:
                return '#FF0000'  # 深紅色
            elif value >= 20:
                return '#FF4444'  # 紅色
            elif value >= 10:
                return '#FF8888'  # 淺紅色
            else:
                return '#FFCCCC'  # 極淺紅色

        def get_negative_color(value):
            if value <= -30:
                return '#004400'  # 深綠色
            elif value <= -20:
                return '#006600'  # 綠色
            elif value <= -10:
                return '#008800'  # 淺綠色
            else:
                return '#00AA00'  # 極淺綠色

        # 2. 數據驗證函數
        def validate_revenue_data(acc_rev, last_year_acc_rev):
            """驗證和處理營收數據"""
            if acc_rev is None or acc_rev.empty:
                raise ValueError("無累計營收數據")
            if last_year_acc_rev is None or last_year_acc_rev.empty:
                raise ValueError("無去年累計營收數據")
                
            # 獲取最後一個應該有數據的月份
            current_date = pd.Timestamp.now()
            base_date = current_date.replace(day=10)
            
            # 處理假日順延
            while base_date.weekday() in [5, 6]:  # 週末順延
                base_date += pd.Timedelta(days=1)
            base_date += pd.Timedelta(days=3)  # 額外緩衝3天
            
            # 確定最後可用的數據月份
            if current_date <= base_date:
                last_valid_date = (current_date - pd.DateOffset(months=1)).replace(day=1)
            else:
                last_valid_date = current_date.replace(day=1)
                
            # 過濾數據
            acc_rev = acc_rev[acc_rev.index <= last_valid_date]
            last_year_acc_rev = last_year_acc_rev[last_year_acc_rev.index <= last_valid_date]
            
            return acc_rev.dropna(), last_year_acc_rev.dropna()

        # 3. 主要繪圖邏輯
        try:
            # 3.1 數據獲取和預處理
            if not isinstance(stock_id, str):
                stock_id = str(stock_id)

            # 獲取基礎數據
            acc_rev = self.data.get('monthly_revenue:當月累計營收')[stock_id]
            last_year_acc_rev = self.data.get('monthly_revenue:去年累計營收')[stock_id]
            
            # 驗證和處理數據
            acc_rev, last_year_acc_rev = validate_revenue_data(acc_rev, last_year_acc_rev)
            
            close = self.data.get('price:收盤價')[stock_id]
            if close is None or close.empty:
                raise ValueError("無股價數據")
            # 2. 計算基本技術指標
            yoy_growth = (acc_rev / last_year_acc_rev - 1) * 100

            # 營收的移動平均線（使用原始營收數據）
            ma3 = acc_rev.rolling(window=3).mean()
            ma12 = acc_rev.rolling(window=12).mean()
            ma24 = acc_rev.rolling(window=24).mean()

            # 計算營收移動平均的年增率
            ma3_yoy = (ma3 / ma3.shift(12) - 1) * 100
            ma12_yoy = (ma12 / ma12.shift(12) - 1) * 100
            ma24_yoy = (ma24 / ma24.shift(12) - 1) * 100

            # 計算月均價
            monthly_price = close.resample('M').mean()

            # 3. 計算進階指標
            # 3.1 營收動能指標
            revenue_momentum = (ma3_yoy.iloc[-1] - ma3_yoy.iloc[-2]) / abs(ma3_yoy.iloc[-2]) * 100 \
                if abs(ma3_yoy.iloc[-2]) > 0 else 0

            # 3.2 進階季節性分析
            def analyze_advanced_seasonality(revenue_data):
                """
                進階季節性分析函數
                返回詳細的季節性指標
                """
                # 基本數據準備
                df = pd.DataFrame({
                    'revenue': revenue_data,
                    'month': revenue_data.index.month,
                    'year': revenue_data.index.year
                })

                # 計算趨勢（12個月移動平均）
                trend = revenue_data.rolling(window=12, center=True).mean()

                # 去趨勢化
                detrended = revenue_data / trend

                # 每月統計指標
                monthly_stats = df.groupby('month').agg({
                    'revenue': ['mean', 'std', 'min', 'max', 'count']
                })

                # 計算變異係數
                monthly_stats['cv'] = monthly_stats[('revenue', 'std')] / \
                    monthly_stats[('revenue', 'mean')]

                # 最近3年的月度模式
                recent_pattern = df[df['year'] >= df['year'].max() - 2]
                recent_monthly = recent_pattern.groupby('month')['revenue'].mean()

                # 當前月份分析
                current_month = revenue_data.index[-1].month
                current_revenue = revenue_data.iloc[-1]
                month_avg = monthly_stats.loc[current_month, ('revenue', 'mean')]
                month_std = monthly_stats.loc[current_month, ('revenue', 'std')]

                # 計算Z分數
                z_score = (current_revenue - month_avg) / \
                    month_std if month_std != 0 else 0

                # 計算季節性強度
                seasonal_strength = (
                    current_revenue / month_avg - 1) * 100 if month_avg != 0 else 0

                # 季節性穩定度評分（1-10分）
                cv_scores = monthly_stats['cv'].rank(pct=True) * 10
                stability_score = cv_scores[current_month]

                # 季節性趨勢
                seasonal_trend = (recent_monthly / recent_monthly.mean() - 1) * 100

                return {
                    'stats': monthly_stats,
                    'z_score': z_score,
                    'seasonal_strength': seasonal_strength,
                    'stability_score': stability_score,
                    'seasonal_trend': seasonal_trend[current_month],
                    'current_month': current_month,
                    'detrended': detrended
                }

            # 執行季節性分析
            seasonality_results = analyze_advanced_seasonality(acc_rev)
            seasonal_strength = seasonality_results['seasonal_strength']
            z_score = seasonality_results['z_score']
            stability_score = seasonality_results['stability_score']

            # 3.3 計算PR值
            recent_60m = acc_rev.tail(60)
            latest_rev = acc_rev.iloc[-1]
            pr_value = stats.percentileofscore(recent_60m.dropna(), latest_rev) \
                if not recent_60m.empty and not pd.isna(latest_rev) else 50

            # 4. 狀態判斷
            latest_yoy = yoy_growth.iloc[-1]

            # 4.1 營收強度判斷
            revenue_status = (
                "營收強勢成長" if latest_yoy >= 20 else
                "營收穩定成長" if latest_yoy >= 10 else
                "營收微幅成長" if latest_yoy >= 0 else
                "營收略為下滑" if latest_yoy >= -10 else
                "營收明顯衰退"
            )

            # 4.2 PR值強度判斷
            pr_status = (
                "極度強勢" if pr_value >= 90 else
                "表現強勢" if pr_value >= 70 else
                "表現平穩" if pr_value >= 30 else
                "表現偏弱" if pr_value >= 10 else
                "極度弱勢"
            )

            # 4.3 綜合警示判斷
            alerts = []
            # 基本面警示
            if (yoy_growth.tail(3) < 0).all():
                alerts.append("營收連續三個月下滑")
            if latest_yoy < -20:
                alerts.append("營收大幅衰退")
            # 動能警示
            if revenue_momentum < -10:
                alerts.append("營收動能明顯轉弱")
            # 季節性警示
            if z_score < -2:
                alerts.append(f"顯著低於季節性表現 (Z: {z_score:.1f})")
            if seasonal_strength < -20:
                alerts.append("季節性表現不佳")
            if stability_score < 3:
                alerts.append("季節性模式不穩定")
            # 均線警示
            if ma3_yoy.iloc[-1] < ma12_yoy.iloc[-1]:
                alerts.append("短期均線跌破長期均線")

            # 4.4 標題顏色判斷
            title_color = (
                '#FF4444'  # 紅色：強勢
                if (latest_yoy >= 10 and pr_value >= 70) or
                (revenue_momentum > 10 and seasonal_strength > 10 and z_score > 1)
                else '#00CC00'  # 綠色：弱勢
                if (latest_yoy < -10 or pr_value < 30) or
                (revenue_momentum < -10 and seasonal_strength < -10 and z_score < -1)
                else 'black'    # 黑色：中性
            )

            # 5. 繪製圖表
            ax1 = ax  # 股價
            ax2 = ax1.twinx()  # 營收

            # 設定時間範圍（最近24個月）
            last_24_months = slice(-24, None)

            # 5.1 繪製月均價
            ax1.plot(monthly_price.index[last_24_months],
                    monthly_price.values[last_24_months],
                    color='black',
                    linestyle='--',
                    linewidth=2,
                    alpha=0.8,
                    label='月均價')

            # 5.2 添加0軸線
            ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

            # 5.3 繪製營收年增率柱狀圖
            positive_mask = yoy_growth.iloc[last_24_months] >= 0
            negative_mask = yoy_growth.iloc[last_24_months] < 0
            
            
            # 繪製正值柱狀圖
            for date, value in zip(acc_rev.index[last_24_months][positive_mask], 
                                yoy_growth.values[last_24_months][positive_mask]):
                color = get_positive_color(value)  # 先獲取顏色值
                ax2.bar(date, value,
                        color=color,    
                        alpha=0.6,
                        width=20)
                print(f"Positive value: {value:.2f}, Color: {color}")

            # 繪製負值柱狀圖
            for date, value in zip(acc_rev.index[last_24_months][negative_mask],
                                yoy_growth.values[last_24_months][negative_mask]):
                color = get_negative_color(value)  # 先獲取顏色值
                ax2.bar(date, value,
                        color=color,    
                        alpha=0.6,
                        width=20)
                print(f"Negative value: {value:.2f}, Color: {color}")

                       
            # 5.4 繪製移動平均線
            ax2.plot(ma3_yoy.index[last_24_months],
                    ma3_yoy.values[last_24_months],
                    color='#FF8800',
                    linestyle='-',
                    linewidth=2,
                    label='3MA')

            ax2.plot(ma12_yoy.index[last_24_months],
                    ma12_yoy.values[last_24_months],
                    color='#0088FF',
                    linestyle='-',
                    linewidth=2,
                    label='12MA')

            ax2.plot(ma24_yoy.index[last_24_months],
                    ma24_yoy.values[last_24_months],
                    color='#884400',
                    linestyle='-',
                    linewidth=2,
                    label='24MA')

            # 5.5 添加正負成長區域填充
            ax2.fill_between(acc_rev.index[last_24_months],
                            yoy_growth.values[last_24_months],
                            0,
                            where=(yoy_growth.iloc[last_24_months] >= 0),
                            color='#FF4444',
                            alpha=0.1)

            ax2.fill_between(acc_rev.index[last_24_months],
                            yoy_growth.values[last_24_months],
                            0,
                            where=(yoy_growth.iloc[last_24_months] < 0),
                            color='#00CC00',
                            alpha=0.1)

            # 6. 設置標題和標籤
            # 6.1 獲取股票資訊
            stock_info = self.stock_topics_df[
                self.stock_topics_df['stock_no'] == int(stock_id)]

            if not stock_info.empty:
                stock_name = stock_info["stock_name"].values[0]
                topic = stock_info["topic"].values[0] if "topic" in stock_info.columns else "無"
                title = f'{stock_name}({stock_id})\n{topic}'
            else:
                title = f'股票代碼: {stock_id}'

            # 6.2 格式化營收數字
            latest_rev_formatted = format(int(latest_rev/1000), ',')

            # 6.3 設置標題文字
            title_text = (
                f'{title}\n'
                f'累積營收: {latest_rev_formatted}百萬元 | {revenue_status}\n'
                f'年增率: {latest_yoy:+.1f}% | PR值: {pr_value:.0f} | {pr_status}\n'
                f'動能: {revenue_momentum:+.1f}% | 季節性: {
                    seasonal_strength:+.1f}% (Z: {z_score:.1f}, 穩定度: {stability_score:.1f})'
            )

            # 6.4 設置帶顏色的標題
            ax2.set_title(title_text, pad=15, fontsize=30,
                        fontweight='bold', color=title_color)

            # 6.5 如果有警示信息，添加到圖表上
            if alerts:
                alert_text = '\n'.join(alerts)
                ax2.text(0.02, 0.98, alert_text,
                        transform=ax2.transAxes,
                        va='top',
                        color='red',
                        fontsize=16)

            # 7. 設置軸標籤和格式
            ax1.set_ylabel('股\n價', fontsize=16, rotation=0, labelpad=20)
            ax2.set_ylabel('年\n增\n率\n%', fontsize=16, rotation=0, labelpad=20)

            # 設置x軸格式
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

            # 8. 添加年增率標籤
            for i, (date, value) in enumerate(zip(yoy_growth.index[last_24_months][-4:],
                                                yoy_growth.values[last_24_months][-4:])):
                if not pd.isna(value):
                    y_pos = value * 1.02 if value >= 0 else value * 1.02
                    va = 'bottom' if value >= 0 else 'top'
                    ax2.text(date,
                            y_pos,
                            f'{value:.1f}%',
                            ha='center',
                            va=va,
                            fontsize=10,
                            color='#FF4444' if value >= 0 else '#00CC00')

            # 9. 添加趨勢箭頭
            last_date = yoy_growth.index[-1]
            last_value = yoy_growth.iloc[-1]
            if last_value >= 0:
                ax2.annotate('↑', (last_date, last_value * 1.1),
                            color='#FF4444',
                            fontsize=16,
                            ha='center',
                            va='bottom')
            else:
                ax2.annotate('↓', (last_date, last_value * 1.1),
                            color='#00CC00',
                            fontsize=16,
                            ha='center',
                            va='bottom')

            # 10. 設置圖例
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines1 + lines2,
                    labels1 + labels2,
                    loc='upper left',
                    bbox_to_anchor=(0.02, -0.15),
                    ncol=3,
                    fontsize=16)

            # 11. 設置網格
            ax2.grid(True, linestyle='--', alpha=0.7)

            # 12. 調整Y軸範圍
            y_values = yoy_growth.values[last_24_months]
            y_values = y_values[~np.isnan(y_values)]  # 移除NaN值
            if len(y_values) > 0:  # 確保有有效數據
                y_max = np.max(y_values)
                y_min = np.min(y_values)
                y_abs_max = max(abs(y_max), abs(y_min))
                margin = y_abs_max * 0.2
                
                # 確保計算出的範圍是有效的
                if np.isfinite(y_abs_max) and np.isfinite(margin):
                    ax2.set_ylim(-y_abs_max - margin, y_abs_max + margin)
                else:
                    # 如果計算出的範圍無效，使用預設值
                    ax2.set_ylim(-100, 100)

            # 設置股價軸的範圍
            price_values = monthly_price.values[last_24_months]
            price_values = price_values[~np.isnan(price_values)]  # 移除NaN值
            if len(price_values) > 0:  # 確保有有效數據
                price_max = np.max(price_values)
                price_min = np.min(price_values)
                price_margin = (price_max - price_min) * 0.1
                
                # 確保計算出的範圍是有效的
                if np.isfinite(price_min) and np.isfinite(price_max) and np.isfinite(price_margin):
                    ax1.set_ylim(price_min - price_margin, price_max + price_margin)
                else:
                    # 如果計算出的範圍無效，使用預設值
                    ax1.set_ylim(0, 100)




        except Exception as e:
            print(f"Error in plot_revenue2: {str(e)}")
            return ax

# =========================財務報表============================================================


    def plot_financial_analysis(self, ax, stock_id):
        """繪製財務指標分析圖"""
        try:
            if not isinstance(stock_id, str):
                stock_id = str(stock_id)

            def safe_process_series(series):
                """安全處理時間序列，移除 nan 並填充前值"""
                if series is None or series.empty:
                    return pd.Series()
                return series.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')

            def safe_divide(a, b):
                """安全除法，處理除以零和無效值的情況"""
                result = np.where(b != 0, a / b, np.nan)
                return pd.Series(result, index=a.index)

            def get_adaptive_limits(series_list):
                """根據數據分布自適應調整軸限制"""
                # 過濾掉空的序列並合併所有有效值
                valid_series = [s for s in series_list if not s.empty]
                if not valid_series:
                    return 0, 100  # 默認值

                all_values = pd.concat(valid_series)
                all_values = all_values.replace([np.inf, -np.inf], np.nan).dropna()

                if all_values.empty:
                    return 0, 100  # 如果沒有有效值，返回默認範圍

                # 計算基本統計量
                mean = all_values.mean()
                std = all_values.std()
                skew = all_values.skew()

                # 根據偏度調整縮放因子
                scale_factor = 1.5 + abs(skew) * 0.2  # 偏度越大，縮放因子越大

                lower_bound = mean - scale_factor * std
                upper_bound = mean + scale_factor * std

                # 確保包含所有重要數據點
                actual_min = all_values.min()
                actual_max = all_values.max()

                lower_bound = min(lower_bound, actual_min)
                upper_bound = max(upper_bound, actual_max)

                if lower_bound > 0:
                    lower_bound = 0

                margin = (upper_bound - lower_bound) * 0.1
                return lower_bound - margin, upper_bound + margin

            def standardize_series(series):
                """標準化時間序列數據"""
                mean = series.mean()
                std = series.std()
                return (series - mean) / std if std != 0 else series - mean

            # 獲取並安全處理基本財務數據
            net_income = safe_process_series(self.data.get(
                "fundamental_features:經常稅後淨利").get(stock_id, pd.Series()))
            total_equity = safe_process_series(self.data.get(
                "financial_statement:股東權益總額").get(stock_id, pd.Series()))
            close = safe_process_series(self.data.get(
                "price:收盤價").get(stock_id, pd.Series()))
            GPM = safe_process_series(self.data.get(
                "fundamental_features:營業毛利率").get(stock_id, pd.Series()))
            OPM = safe_process_series(self.data.get(
                "fundamental_features:營業利益率").get(stock_id, pd.Series()))
            NPM = safe_process_series(self.data.get(
                "fundamental_features:稅後淨利率").get(stock_id, pd.Series()))
            EPS = safe_process_series(self.data.get(
                'financial_statement:每股盈餘').get(stock_id, pd.Series()))
            OPGR = safe_process_series(self.data.get(
                "fundamental_features:營業利益成長率").get(stock_id, pd.Series()))

            # 獲取並安全處理營運效率指標所需數據
            NOI = safe_process_series(self.data.get(
                'financial_statement:營業外收入及支出').get(stock_id, pd.Series()))
            cash = safe_process_series(self.data.get(
                'financial_statement:現金及約當現金').get(stock_id, pd.Series()))
            liabilities = safe_process_series(self.data.get(
                'financial_statement:負債總額').get(stock_id, pd.Series()))

            # 安全計算營運效率指標
            operation_efficiency = safe_divide(NOI, (cash * liabilities))
            operation_efficiency = safe_process_series(operation_efficiency)

            if net_income.empty or total_equity.empty or close.empty or EPS.empty:
                raise ValueError("缺少關鍵財務數據")

            # 確保索引是datetime類型
            for series in [net_income, total_equity, close, GPM, OPM, NPM, EPS, OPGR, operation_efficiency]:
                if not series.empty:
                    series.index = pd.to_datetime(series.index)

            # 找出所有數據的共同時間範圍
            common_dates = sorted(set(GPM.index) & set(OPM.index) & set(NPM.index) &
                                set(OPGR.index) & set(operation_efficiency.index))
            latest_12_dates = common_dates[-12:]  # 取最近12個共同日期

            # 根據共同日期重新取得數據
            GPM = GPM[GPM.index.isin(latest_12_dates)]
            OPM = OPM[OPM.index.isin(latest_12_dates)]
            NPM = NPM[NPM.index.isin(latest_12_dates)]
            OPGR = OPGR[OPGR.index.isin(latest_12_dates)]
            operation_efficiency = operation_efficiency[operation_efficiency.index.isin(
                latest_12_dates)]

            # 安全計算 EPS 相關指標
            latest_eps = EPS[EPS.index.isin(latest_12_dates)]
            latest_eps_value = latest_eps.iloc[-1] if not latest_eps.empty else np.nan
            eps_rank = latest_eps.rank(ascending=False)[
                latest_eps.index[-1]] if not latest_eps.empty else np.nan
            eps_yoy = ((latest_eps.iloc[-1] / latest_eps.iloc[-5] - 1)
                    * 100) if len(latest_eps) >= 5 else 0
            eps_qoq = ((latest_eps.iloc[-1] / latest_eps.iloc[-2] - 1)
                    * 100) if len(latest_eps) >= 2 else 0

            def calculate_stats(series):
                """安全計算統計值"""
                if len(series) < 2:
                    return 0, 0, 0, 0, 0
                recent = series[-8:]
                mean = recent.mean()
                std = recent.std()
                latest = series.iloc[-1]
                yoy = ((latest / series.iloc[-5] - 1)
                    * 100) if len(series) >= 5 else 0
                qoq = ((latest / series.iloc[-2] - 1)
                    * 100) if len(series) >= 2 else 0
                z_score = (latest - mean) / std if std != 0 else 0
                return mean, std, yoy, qoq, z_score

            # 計算各指標的統計值
            gpm_stats = calculate_stats(GPM)
            opm_stats = calculate_stats(OPM)
            npm_stats = calculate_stats(NPM)
            opgr_stats = calculate_stats(OPGR)
            oe_stats = calculate_stats(operation_efficiency)

            # 對指標進行標準化處理
            GPM_std = standardize_series(GPM)
            OPM_std = standardize_series(OPM)
            NPM_std = standardize_series(NPM)
            OPGR_std = standardize_series(OPGR)
            OE_std = standardize_series(operation_efficiency)

            x = np.arange(len(latest_12_dates))
            width = 0.15

            # 繪製柱狀圖 (使用標準化後的數據)
            ax.bar(x - width*2, GPM_std.values, width,
                label='毛利率', color='#FF9999', alpha=0.7)
            ax.bar(x - width, OPM_std.values, width,
                label='營益率', color='#99CC99', alpha=0.7)
            ax.bar(x, NPM_std.values, width, label='淨利率',
                color='#9999FF', alpha=0.7)
            ax.bar(x + width, OPGR_std.values, width,
                label='營益成長率', color='#FFB366', alpha=0.7)
            ax.bar(x + width*2, OE_std.values, width,
                label='營運效率', color='#DDA0DD', alpha=0.7)

            # 使用自適應縮放設置y軸範圍
            series_list = [GPM_std, OPM_std, NPM_std, OPGR_std, OE_std]
            y_min, y_max = get_adaptive_limits(series_list)
            ax.set_ylim(y_min, y_max)

            # 添加趨勢線（使用標準化數據）
            def add_trendline(data, x, color, label, alpha=0.3):
                """添加趨勢線並返回斜率"""
                z = np.polyfit(range(len(data)), data, 1)
                p = np.poly1d(z)
                trend_y = p(range(len(data)))
                line, = ax.plot(x, trend_y, '--', color=color, alpha=alpha, linewidth=2,
                                label=f'{label}趨勢線')
                return z[0], line

            gpm_slope, gpm_line = add_trendline(
                GPM_std.values, x, '#FF0000', '毛利率')
            opm_slope, opm_line = add_trendline(
                OPM_std.values, x, '#00FF00', '營益率')
            npm_slope, npm_line = add_trendline(
                NPM_std.values, x, '#0000FF', '淨利率')

            # 調整y軸標籤位置
            ax.set_ylabel('獲\n利\n率\n(%)', fontsize=18, rotation=0)
            ax.yaxis.set_label_coords(-0.02, 0.5)

            def get_trend_direction(slope):
                """根據斜率返回趨勢方向符號和描述"""
                if slope > 0.1:
                    return "↗", "上升"
                elif slope < -0.1:
                    return "↘", "下降"
                else:
                    return "→", "持平"

            def get_profit_status():
                # 計算年增率
                gpm_yoy = ((GPM.iloc[-1] / GPM.iloc[-5] - 1)
                        * 100) if len(GPM) >= 5 else 0
                opm_yoy = ((OPM.iloc[-1] / OPM.iloc[-5] - 1)
                        * 100) if len(OPM) >= 5 else 0
                npm_yoy = ((NPM.iloc[-1] / NPM.iloc[-5] - 1)
                        * 100) if len(NPM) >= 5 else 0
                oe_yoy = ((operation_efficiency.iloc[-1] / operation_efficiency.iloc[-5] - 1) * 100) if len(
                    operation_efficiency) >= 5 else 0

                # 檢查當前值是否高於平均
                above_mean_count = sum([
                    GPM.iloc[-1] > gpm_stats[0],
                    OPM.iloc[-1] > opm_stats[0],
                    NPM.iloc[-1] > npm_stats[0],
                    operation_efficiency.iloc[-1] > oe_stats[0]
                ])

                # 檢查年增率是否為正
                positive_yoy_count = sum([
                    gpm_yoy > 0,
                    opm_yoy > 0,
                    npm_yoy > 0,
                    oe_yoy > 0
                ])

                # 綜合判斷
                if above_mean_count >= 3 and positive_yoy_count >= 3:
                    return "優於平均 (年增率正向)", "darkred"
                elif above_mean_count >= 3 and positive_yoy_count < 3:
                    return "優於平均 (年增率待觀察)", "darkgreen"
                elif above_mean_count < 3 and positive_yoy_count >= 3:
                    return "低於平均 (年增率改善中)", "darkblue"
                else:
                    return "低於平均 (年增率負向)", "purple"

            profit_status, title_color = get_profit_status()
            eps_trend = "創新高" if eps_rank == 1 else f"第{int(eps_rank)}高" if not np.isnan(
                eps_rank) else "數據不足"

            # 獲取三率趨勢方向
            gpm_arrow, gpm_trend = get_trend_direction(gpm_slope)
            opm_arrow, opm_trend = get_trend_direction(opm_slope)
            npm_arrow, npm_trend = get_trend_direction(npm_slope)

            # 設置標題
            ax.text(0.5, 1.45, f'{stock_id}',
                    horizontalalignment='center',
                    transform=ax.transAxes,
                    fontsize=30,
                    fontweight='bold',
                    color=title_color)

            ax.text(0.5, 1.35, f'獲利分析 - {profit_status}',
                    horizontalalignment='center',
                    transform=ax.transAxes,
                    fontsize=30,
                    color=title_color)

            eps_summary = (f'EPS: {latest_eps_value:.2f}元 ({eps_trend})') if not np.isnan(
                latest_eps_value) else "EPS: 數據不足"
            if eps_yoy != 0:
                eps_summary += f' YoY {eps_yoy:+.1f}%'
            if eps_qoq != 0:
                eps_summary += f' QoQ {eps_qoq:+.1f}%'

            ax.text(0.5, 1.23, eps_summary,
            horizontalalignment='center',
            transform=ax.transAxes,
            fontsize=30)

            # 添加三率趨勢資訊
            trend_summary = (f'趨勢走向: 毛利率{gpm_arrow}{gpm_trend} / '
                            f'營益率{opm_arrow}{opm_trend} / '
                            f'淨利率{npm_arrow}{npm_trend}')
            ax.text(0.5, 1.13, trend_summary,
                    horizontalalignment='center',
                    transform=ax.transAxes,
                    fontsize=30)

            # 設置x軸刻度和標籤 - 增加字體大小為22
            ax.set_xticks(x)
            ax.set_xticklabels([d.strftime('%Y-%m') for d in latest_12_dates], rotation=45, fontsize=20)

            # 設置圖例 - 增加字體大小
            lines, labels = ax.get_legend_handles_labels()
            ax.legend(lines, labels,
                    loc='upper left',
                    fontsize=18,
                    framealpha=0.8,
                    bbox_to_anchor=(0, 1))

            # 設置網格
            ax.grid(True, linestyle='--', alpha=0.3)

            # 標示最後數值
            last_x = len(x) - 1
            offsets = {
                'GPM': (-width*2, 0),
                'OPM': (-width, 0),
                'NPM': (0, 0),
                'OPGR': (width, 0),
                'OE': (width*2, 0)
            }
            
            for series, name, offset_key in [
                (GPM_std, '毛利率', 'GPM'),
                (OPM_std, '營益率', 'OPM'),
                (NPM_std, '淨利率', 'NPM'),
                (OPGR_std, '營益成長率', 'OPGR'),
                (OE_std, '營運效率', 'OE')
            ]:
                last_value = series.iloc[-1]
                x_offset, y_offset = offsets[offset_key]
                ax.text(last_x + x_offset, last_value + y_offset + 0.2,
                        f'{last_value:.2f}',
                        ha='center',
                        va='bottom',
                        fontsize=14)

            return True

        except Exception as e:
            import logging
            logging.error(f"繪製財務分析圖表時發生錯誤: {str(e)}")
            ax.clear()
            ax.text(0.5, 0.5,
                    f'繪製圖表時發生錯誤:\n{str(e)}',
                    ha='center',
                    va='center',
                    transform=ax.transAxes,
                    fontsize=20,
                    color='red')
            return False



# =============================研發費用率==============================

    def plot_rd_vs_pm_analysis(self, ax, stock_id):
        """繪製 研發費用率 vs. 管理費用率 + 股東權益比率 + 營收變化分析圖，並加上財務評語"""
        
        try:
            stock_id = str(stock_id)  # 確保 stock_id 為字串

            def safe_process_series(series):
                """安全處理數據，處理 NaN 及填充前值"""
                if series is None or series.empty:
                    return pd.Series(dtype='float64')
                return series.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')

            def safe_divide(a, b):
                """安全除法，處理除以零的情況"""
                result = np.where(b != 0, a / b, np.nan)
                return pd.Series(result, index=a.index)

            # 取得季報數據
            rd_ratio = safe_process_series(self.data.get('fundamental_features:研究發展費用率').get(stock_id, pd.Series()))
            pm_ratio = safe_process_series(self.data.get('fundamental_features:管理費用率').get(stock_id, pd.Series()))
            eq_ratio = safe_process_series(self.data.get('fundamental_features:淨值除資產').get(stock_id, pd.Series()))

            # 取得月營收數據
            revenue = safe_process_series(self.data.get('monthly_revenue:當月營收').get(stock_id, pd.Series()))

            # 確保數據非空
            if rd_ratio.empty or pm_ratio.empty or eq_ratio.empty or revenue.empty:
                ax.text(0.5, 0.5, f"{stock_id} 數據不足，無法繪製圖表", ha='center', va='center', fontsize=14, color='red')
                return False

            # **🟢 確保所有索引都是 `PeriodIndex` (季度)**
            rd_ratio.index = pd.PeriodIndex(rd_ratio.index, freq='Q')
            pm_ratio.index = pd.PeriodIndex(pm_ratio.index, freq='Q')
            eq_ratio.index = pd.PeriodIndex(eq_ratio.index, freq='Q')

            # 取最近 12 季的索引
            latest_12_quarters = rd_ratio.index[-12:]

            # **轉換月營收為「季營收總和」**
            revenue.index = pd.to_datetime(revenue.index)  # 轉為 datetime
            revenue_quarterly = revenue.resample('Q').sum()  # 按季度重新取樣
            revenue_quarterly.index = pd.PeriodIndex(revenue_quarterly.index, freq='Q')  # 轉為季度索引

            # **計算 營收變化**
            revenue_yoy = safe_divide(revenue_quarterly, revenue_quarterly.shift(4)) - 1  # 同比 (去年同期比)
            revenue_mom = safe_divide(revenue_quarterly, revenue_quarterly.shift(1)) - 1  # 季比 (上一季比)

            # **用 `reindex()` 來確保索引一致，避免 `loc[]` 報錯**
            rd_ratio = rd_ratio.reindex(latest_12_quarters)
            pm_ratio = pm_ratio.reindex(latest_12_quarters)
            rd_pm_ratio = safe_divide(rd_ratio, pm_ratio)
            eq_ratio = eq_ratio.reindex(latest_12_quarters)
            revenue_yoy = revenue_yoy.reindex(latest_12_quarters)
            revenue_mom = revenue_mom.reindex(latest_12_quarters)

            # **防止索引超界**
            latest_rd_pm = rd_pm_ratio.iloc[-1] if not rd_pm_ratio.empty else np.nan
            latest_eq_ratio = eq_ratio.iloc[-1] if not eq_ratio.empty else np.nan
            latest_revenue_yoy = revenue_yoy.iloc[-1] if not revenue_yoy.empty else np.nan

            # 計算最新指標值的字符串表示，用於標題顯示
            latest_rd_pm_str = f"{latest_rd_pm:.2f}" if not np.isnan(latest_rd_pm) else "N/A"
            latest_eq_ratio_str = f"{latest_eq_ratio*100:.1f}%" if not np.isnan(latest_eq_ratio) else "N/A"
            latest_revenue_yoy_str = f"{latest_revenue_yoy*100:.1f}%" if not np.isnan(latest_revenue_yoy) else "N/A"
            
            # 更細緻的評估分類
            # 1. 研發/管理費用比評估
            if latest_rd_pm > 1.5:
                rd_pm_eval = "卓越研發"
            elif latest_rd_pm > 1.0:
                rd_pm_eval = "優良研發"
            elif latest_rd_pm > 0.7:
                rd_pm_eval = "平衡研發"
            elif latest_rd_pm > 0.4:
                rd_pm_eval = "偏低研發"
            else:
                rd_pm_eval = "薄弱研發"
                
            # 2. 股東權益比率評估    
            if latest_eq_ratio > 0.6:
                eq_eval = "極穩財務"
            elif latest_eq_ratio > 0.4:
                eq_eval = "穩健財務"
            elif latest_eq_ratio > 0.3:
                eq_eval = "一般財務"
            elif latest_eq_ratio > 0.2:
                eq_eval = "偏弱財務"
            else:
                eq_eval = "風險財務"
                
            # 3. 營收年增率評估    
            if latest_revenue_yoy > 0.2:
                rev_eval = "高速成長"
            elif latest_revenue_yoy > 0.05:
                rev_eval = "穩定成長"
            elif latest_revenue_yoy > -0.05:
                rev_eval = "營收持平"
            elif latest_revenue_yoy > -0.2:
                rev_eval = "營收下滑"
            else:
                rev_eval = "營收衰退"
                
            # 綜合評分 (1-5分，5分最好)
            rd_pm_score = 5 if latest_rd_pm > 1.5 else (4 if latest_rd_pm > 1.0 else (3 if latest_rd_pm > 0.7 else (2 if latest_rd_pm > 0.4 else 1)))
            eq_score = 5 if latest_eq_ratio > 0.6 else (4 if latest_eq_ratio > 0.4 else (3 if latest_eq_ratio > 0.3 else (2 if latest_eq_ratio > 0.2 else 1)))
            rev_score = 5 if latest_revenue_yoy > 0.2 else (4 if latest_revenue_yoy > 0.05 else (3 if latest_revenue_yoy > -0.05 else (2 if latest_revenue_yoy > -0.2 else 1)))
            
            total_score = rd_pm_score + eq_score + rev_score
            
            # 更細緻的標題文字和顏色
            title_text = f"{rd_pm_eval}(研管比={latest_rd_pm_str})、{eq_eval}(權益比={latest_eq_ratio_str})、{rev_eval}({latest_revenue_yoy_str})"
            
            # 根據總分設定顏色 (台股習慣：紅色=好，綠色=壞)
            if total_score >= 13:  # 接近滿分
                title_color = "#FF0000"  # 鮮紅色
            elif total_score >= 11:
                title_color = "#FF3333"  # 紅色
            elif total_score >= 9:
                title_color = "#FF6600"  # 橘紅色
            elif total_score >= 7:
                title_color = "#FFCC00"  # 黃色
            elif total_score >= 5:
                title_color = "#CCCC00"  # 黃綠色
            else:
                title_color = "#339933"  # 綠色 (不佳)

            # **保留原有標題設定**
            ax.set_title(f"{stock_id} - {title_text}", fontsize=35, fontweight='bold', color=title_color, pad=20)
                
            # **繪製圖表**
            ax2 = ax.twinx()
            x = np.arange(len(latest_12_quarters))
            width = 0.2  # 柱狀圖寬度

            # 繪製柱狀圖
            bar1 = ax.bar(x - width, rd_ratio.values, width, label='研發費用率(%)', color='#FF9999', alpha=0.7)
            bar2 = ax.bar(x, pm_ratio.values, width, label='管理費用率(%)', color='#99CC99', alpha=0.7)
            bar3 = ax.bar(x + width, eq_ratio.values, width, label='股東權益比率', color='#9999FF', alpha=0.7)

            # 繪製線圖
            line1 = ax2.plot(x, rd_pm_ratio.values, '-o', color='#FF4500', label='研發/管理費用比', linewidth=2, markersize=5)
            line2 = ax2.plot(x, revenue_yoy.values, '--s', color='#4682B4', label='營收年增率(%)', linewidth=2, alpha=0.7, markersize=5)

            # 標記最新值
            last_idx = len(x) - 1
            ax2.annotate(f"{rd_pm_ratio.values[-1]:.2f}", 
                    (x[-1], rd_pm_ratio.values[-1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, color='#FF4500')
            
            ax2.annotate(f"{revenue_yoy.values[-1]*100:.1f}%", 
                    (x[-1], revenue_yoy.values[-1]),
                    xytext=(5, -15), textcoords='offset points',
                    fontsize=9, color='#4682B4')

            # 設置 X 軸標籤 - 增加字體大小
            ax.set_xticks(x)
            ax.set_xticklabels(latest_12_quarters.astype(str), rotation=45, fontsize=18)  # 增大字體到14

            # 添加網格
            ax.grid(True, linestyle='--', alpha=0.3)

            # 設置坐標軸標籤
            ax.set_ylabel('費\n用\n率\n與\n權\n益\n比\n率\n(%)', fontsize=20, rotation=0, labelpad=20)
            ax2.set_ylabel('比\n率\n變\n化', fontsize=18, rotation=0, labelpad=20)

            # 調整 Y 軸範圍
            ax.set_ylim(0, max(max(rd_ratio.max(), pm_ratio.max(), eq_ratio.max()) * 1.2, 0.01))
            
            # 添加圖例 - 組合兩個軸的圖例
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            
            # 創建單一合併的圖例 - 保持在 1.2 位置與標題平行
            ax.legend(handles1 + handles2, labels1 + labels2, 
                    loc='upper left', bbox_to_anchor=(0, 1.2), 
                    ncol=3, frameon=True, framealpha=0.8, fontsize=18)  # 調整圖例字體大小

            # 調整圖表布局
            plt.tight_layout()
            
            return True

        except Exception as e:
            ax.clear()
            ax.text(0.5, 0.5, f'繪製圖表時發生錯誤:\n{str(e)}', ha='center', va='center', transform=ax.transAxes, fontsize=16, color='red')
            return False








# ======================================================集保張數分析================================

    def plot_holding_ratio2(self, ax, stock_id, weeks=10):
        """
        繪製持股比例分析圖，包含統計學信賴區間評估

        Parameters:
            ax: matplotlib axes object, 繪圖用的軸物件
            stock_id: str, 股票代碼
            weeks: int, 要分析的週數，預設16週
        """
        labels = ['200-400', '400-600', '600-800', '800-1000', '1000+']

        def calculate_confidence_interval(data, confidence=0.95):
            """計算信賴區間，根據樣本大小調整信賴水平"""
            import scipy.stats as stats
            import numpy as np

            data = np.array(data)
            n = len(data)

            # 樣本量較小時，降低信賴水平
            if n < 6:
                confidence = 0.90
            elif n < 10:
                confidence = 0.92

            mean = np.mean(data)
            std_err = stats.sem(data)

            if n < 3:
                # 樣本極小時，使用範圍代替信賴區間
                return mean, np.min(data), np.max(data)
            else:
                ci = stats.t.interval(confidence, n-1, loc=mean, scale=std_err)
                return mean, ci[0], ci[1]

        def analyze_trend_significance(ratio_data, window):
            """使用統計分析簡化趨勢描述"""
            from scipy import stats
            import statsmodels.api as sm
            from statsmodels.stats.multitest import multipletests

            labels = ['200-400', '400-600', '600-800', '800-1000', '1000+']
            trend_descriptions = []
            p_values = []

            for i, ratio in enumerate(ratio_data):
                # 確保使用可用的資料窗口
                data_window = min(window, len(ratio))
                if data_window < 2:
                    trend_descriptions.append(f"{labels[i]}張: 資料不足")
                    continue

                recent_data = ratio.iloc[-data_window:].values * 100
                time_x = np.array(range(len(recent_data)))

                # 線性回歸計算斜率並檢驗顯著性
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    time_x, recent_data)
                p_values.append(p_value)

                # 計算變異係數評估穩定性
                mean = np.mean(recent_data)
                std = np.std(recent_data)
                cv = std / mean if mean > 0 else float('inf')

                # 白話文描述
                if p_value > 0.05:
                    if abs(slope) < 0.05:
                        trend = "持平"
                        description = f"{labels[i]}張: 持股比例穩定"
                    elif slope > 0:
                        description = f"{labels[i]}張: 可能緩慢增加中"
                    else:
                        description = f"{labels[i]}張: 可能緩慢減少中"
                else:
                    if slope > 0:
                        description = f"{labels[i]}張: 明顯增加趨勢"
                        if p_value < 0.01:
                            description = f"{labels[i]}張: 強勁增加趨勢"
                    else:
                        description = f"{labels[i]}張: 明顯減少趨勢"
                        if p_value < 0.01:
                            description = f"{labels[i]}張: 強勁減少趨勢"

                # 加入變動性評估
                if mean > 0.5:  # 只有持股比例超過0.5%才加入
                    if cv > 0.25:
                        description += "，波動較大"
                    elif cv < 0.05 and abs(slope) < 0.05:
                        description += "，非常穩定"

                trend_descriptions.append(description)

            # 多重檢驗校正
            if len(p_values) > 1:
                try:
                    reject, pvals_corrected, _, _ = multipletests(
                        p_values, method='fdr_bh')
                    for i in range(len(trend_descriptions)):
                        if p_values[i] <= 0.05 and not reject[i]:
                            trend_descriptions[i] += " (考慮多因素後不顯著)"
                except:
                    pass

            return trend_descriptions

        def calculate_change_ratio(current, base):
            """計算變化率"""
            if abs(base) < 0.001:
                if abs(current) < 0.001:
                    return 0
                return (current - base) * 100
            else:
                return (current - base) / base * 100

        try:
            # 1. 獲取股票名稱
            stock_info = self.stock_topics_df[self.stock_topics_df['stock_no'] == int(
                stock_id)]
            stock_name = stock_info['stock_name'].values[0] if not stock_info.empty else stock_id

            # 2. 獲取持股資料
            inv = self.data.get('inventory')
            if inv is None:
                raise ValueError("無法獲取持股資料")

            # 確保資料型別正確
            inv['date'] = pd.to_datetime(inv['date'], errors='coerce')
            inv['持有股數'] = pd.to_numeric(inv['持有股數'], errors='coerce')
            inv['持股分級'] = pd.to_numeric(inv['持股分級'], errors='coerce')
            inv['人數'] = pd.to_numeric(inv['人數'], errors='coerce')

            # 3. 計算各類持股資料
            holdings_data = []
            for level in range(11, 16):
                holders = inv[inv.持股分級 == level].groupby(['date', 'stock_id'])[
                    '持有股數'].sum().reset_index()
                holdings_data.append(holders)

            all_holders = inv[inv.持股分級 == 17].groupby(['date', 'stock_id'])[
                '持有股數'].sum().reset_index()
            total_shareholders = inv[inv.持股分級 == 17].groupby(
                ['date', 'stock_id'])['人數'].sum().reset_index()

            # 4. 轉換數據格式和計算比例
            ratios = []
            for holders in holdings_data:
                holders_pivot = holders.pivot(
                    index='date', columns='stock_id', values='持有股數')
                all_holders_pivot = all_holders.pivot(
                    index='date', columns='stock_id', values='持有股數')
                common_dates = holders_pivot.index.intersection(
                    all_holders_pivot.index)
                ratio = holders_pivot.loc[common_dates].div(
                    all_holders_pivot.loc[common_dates]).fillna(0)
                ratios.append(ratio)

            # 5. 過濾最近N週數據
            end_date = ratios[0].index[-1]
            start_date = end_date - pd.Timedelta(weeks=weeks-1)  # 根據指定週數計算
            mask = (ratios[0].index >= start_date) & (ratios[0].index <= end_date)

            # 確保有足夠的數據
            if sum(mask) < 2:
                raise ValueError("沒有足夠的數據進行分析")

            ratio_data = [ratio.loc[mask, stock_id] for ratio in ratios]

            # 趨勢分析
            if len(ratio_data[0]) >= 2:
                trend_descriptions = analyze_trend_significance(
                    ratio_data, window=min(weeks, len(ratio_data[0])))
            else:
                trend_descriptions = ["數據不足以進行趨勢分析"]

            # 處理股東人數資料
            total_shareholders_pivot = total_shareholders.pivot(
                index='date', columns='stock_id', values='人數')
            shareholders_data = total_shareholders_pivot.loc[mask, stock_id]
   
            # 獲取收盤價並繪製連續股價線
            close_data = self.data.get("price:收盤價")[stock_id]
            close_data.index = pd.to_datetime(close_data.index)
            # 過濾出在開始和結束日期之間的所有股價數據
            mask_price = (close_data.index >= ratio_data[0].index[0]) & (close_data.index <= ratio_data[0].index[-1])
            filtered_prices = close_data.loc[mask_price]

        
            # 7. 繪製圖表
            ax2 = ax.twinx()
            ax3 = ax.twinx()
            ax3.spines['right'].set_position(('outward', 60))

            # 繪製柱狀圖
            bar_width = 0.85  # 調整柱寬度使圖表更易讀
            group_spacing = 0.95
            bars = []

            for date_idx, date in enumerate(ratio_data[0].index):
                for i, ratio in enumerate(ratio_data):
                    # 計算變化率
                    if date_idx > 0:
                        change = calculate_change_ratio(
                            ratio.iloc[date_idx], ratio.iloc[date_idx-1])
                    else:
                        change = 0

                    color = '#FF4444' if change > 0 else '#00CC00' if change < 0 else '#808080'
                    x_pos = mdates.date2num(
                        date) + (i - 2) * (bar_width * group_spacing)

                    bar = ax.bar(x_pos,
                                ratio.iloc[date_idx] * 100,
                                width=bar_width,
                                alpha=0.8,
                                label=f"{labels[i]}張" if date_idx == 0 else "",
                                color=color,
                                edgecolor='black',
                                linewidth=1.5)

                    # 針對較大的值顯示數值標籤
                    threshold = 0.5  # 持股比例超過0.5%才顯示標籤
                    if ratio.iloc[date_idx] * 100 >= threshold:
                        ax.text(x_pos,
                                ratio.iloc[date_idx] * 100 + 0.2,
                                f'{ratio.iloc[date_idx]*100:.1f}%',
                                ha='center',
                                va='bottom',
                                fontsize=8,
                                rotation=0)

                    if date_idx == 0:
                        bars.append(bar)

         
            # 繪製連續的股價線
            ax2.plot(filtered_prices.index, filtered_prices.values,
                    color='black', linewidth=2, label='收盤價')
            ax3.plot(ratio_data[0].index, shareholders_data,
                    color='orange', linewidth=2, label='股東人數')

           # 8-9. 設置標題和趨勢分析
            title = f'{stock_id} {stock_name} 大戶持股分佈 (觀察近{weeks}週)'
                    
            # 將所有文字合併成一個區塊
            all_text = title + "\n\n"  # 標題後加兩個換行
                    
            # 添加趨勢分析結果
            all_text += "\n".join(trend_descriptions)
                    
            # 設置標題區域，並啟用自動換行
            ax.text(0.5, 1.1, all_text,
                    horizontalalignment='center',  # 整個文字區塊的水平位置仍居中
                    verticalalignment='bottom',
                    transform=ax.transAxes,
                    fontsize=30,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
                    multialignment='left')  # 多行文字左對齊

                            
            # 移除舊的標題設置
            ax.set_title('')


            # 10. 設置軸標籤和圖例
            ax.set_xlabel('')
            ax.set_ylabel('持\n股\n比\n例\n(%)', rotation=0, labelpad=25, fontsize=18)
            ax2.set_ylabel('股\n價', rotation=0, labelpad=20, fontsize=18)
            ax3.set_yticklabels([])
            ax3.set_ylabel('')

            # 11. 設置圖例
            legend_elements = []
            legend_labels = []

            for bar, label in zip(bars, labels):
                legend_elements.append(bar[0])
                legend_labels.append(f'{label}張')

            legend_elements.append(Line2D([0], [0], color='black', linewidth=2))
            legend_elements.append(Line2D([0], [0], color='orange', linewidth=2))
            legend_labels.extend(['收盤價', '股東人數'])

            ax.legend(legend_elements, legend_labels,
                    loc='center left',
                    bbox_to_anchor=(1.05, 0.5),
                    fontsize=18,
                    frameon=True,
                    framealpha=0.8)

            # 12. 設置x軸格式
            ax.set_xlim(
                mdates.date2num(ratio_data[0].index[0]) - 2 * bar_width,
                mdates.date2num(ratio_data[0].index[-1]) + 2 * bar_width
            )
            ax.grid(True, alpha=0.3)

            # 放大X軸日期顯示
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=20)
            
            # 添加日期網格線
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
            ax.xaxis.set_minor_locator(mdates.DayLocator())
            
            # 調整圖表邊距
            plt.tight_layout(rect=[0, 0, 1, 0.7])  # 為標題和說明文字保留空間

        except Exception as e:
            import traceback
            print(f"繪製圖表時發生錯誤: {str(e)}")
            print(traceback.format_exc())
            raise




    def plot_major_holder_distribution(self, ax, stock_id, weeks=50):
            """
            繪製大股東持股分佈和股東總人數分析圖
            """
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.lines import Line2D
            from matplotlib.colors import LinearSegmentedColormap
            import matplotlib.patches as patches

            try:
                stock_info = self.stock_topics_df[self.stock_topics_df['stock_no'] == int(stock_id)]
                stock_name = stock_info['stock_name'].values[0] if not stock_info.empty else stock_id
                inv = self.data.get('inventory')
                if inv is None:
                    raise ValueError("無法獲取持股資料")
                
                inv['date'] = pd.to_datetime(inv['date'], errors='coerce')
                inv['持有股數'] = pd.to_numeric(inv['持有股數'], errors='coerce')
                inv['持股分級'] = pd.to_numeric(inv['持股分級'], errors='coerce')
                inv['人數'] = pd.to_numeric(inv['人數'], errors='coerce')
                
                holders_400plus = inv[inv.持股分級.isin([12, 13, 14, 15])].groupby(['date', 'stock_id'])['持有股數'].sum().reset_index()
                holders_1000plus = inv[inv.持股分級 == 15].groupby(['date', 'stock_id'])['持有股數'].sum().reset_index()
                all_holders = inv[inv.持股分級 == 17].groupby(['date', 'stock_id'])['持有股數'].sum().reset_index()
                total_shareholders = inv[inv.持股分級 == 17].groupby(['date', 'stock_id'])['人數'].sum().reset_index()
                
                holders_400plus_pivot = holders_400plus.pivot(index='date', columns='stock_id', values='持有股數')
                holders_1000plus_pivot = holders_1000plus.pivot(index='date', columns='stock_id', values='持有股數')
                all_holders_pivot = all_holders.pivot(index='date', columns='stock_id', values='持有股數')
                
                common_dates = holders_400plus_pivot.index.intersection(holders_1000plus_pivot.index).intersection(all_holders_pivot.index)
                ratio_400plus = holders_400plus_pivot.loc[common_dates].div(all_holders_pivot.loc[common_dates]).fillna(0) * 100
                ratio_1000plus = holders_1000plus_pivot.loc[common_dates].div(all_holders_pivot.loc[common_dates]).fillna(0) * 100
                total_shareholders_pivot = total_shareholders.pivot(index='date', columns='stock_id', values='人數')
                
                end_date = common_dates[-1]
                start_date = end_date - pd.Timedelta(weeks=weeks-1)
                date_mask = (common_dates >= start_date) & (common_dates <= end_date)
                if sum(date_mask) < 2:
                    raise ValueError(f"沒有足夠的數據進行分析，只有{sum(date_mask)}筆資料")
                
                recent_dates = common_dates[date_mask]
                ratio_400plus_recent = ratio_400plus.loc[recent_dates, stock_id]
                ratio_1000plus_recent = ratio_1000plus.loc[recent_dates, stock_id]
                shareholders_recent = total_shareholders_pivot.loc[recent_dates, stock_id]
                
                has_price_data = False
                aligned_prices = None
                try:
                    close_data = self.data.get("price:收盤價")[stock_id]
                    if close_data is not None:
                        close_data.index = pd.to_datetime(close_data.index)
                        mask_price = (close_data.index >= recent_dates[0]) & (close_data.index <= recent_dates[-1])
                        filtered_prices = close_data.loc[mask_price]
                        aligned_prices = pd.Series(index=recent_dates)
                        for date in recent_dates:
                            closest_price_idx = close_data.index.get_indexer([date], method='nearest')[0]
                            if closest_price_idx >= 0 and closest_price_idx < len(close_data):
                                aligned_prices[date] = close_data.iloc[closest_price_idx]
                        has_price_data = True
                except:
                    has_price_data = False
                
                # 繪製圖表
                ax2 = ax.twinx()
                shareholder_changes = shareholders_recent.pct_change() * 100
                shareholder_changes.iloc[0] = 0
                price_changes = None
                if has_price_data:
                    price_changes = aligned_prices.pct_change() * 100
                    price_changes.iloc[0] = 0
                
                # 新配色：好用紅色，差用綠色，中性用灰色
                good_cmap = LinearSegmentedColormap.from_list('GoodGradient', ['#FFCCCC', '#FF9999', '#CC3333'])
                bad_cmap = LinearSegmentedColormap.from_list('BadGradient', ['#CCFFCC', '#99CC99', '#336633'])
                neutral_cmap = LinearSegmentedColormap.from_list('NeutralGradient', ['#E6E6E6', '#B3B3B3', '#666666'])
                
                bar_colors = []
                weekly_evaluation = []
                for i in range(len(recent_dates)):
                    if i == 0:
                        bar_colors.append('#CCCCCC')
                        weekly_evaluation.append('基準週')
                        continue
                    sh_change = shareholder_changes.iloc[i]
                    if has_price_data:
                        price_change = price_changes.iloc[i]
                        intensity = min(max(abs(sh_change), abs(price_change)) / 3, 1.0)
                        if price_change > 0 and sh_change < 0:
                            bar_colors.append(good_cmap(intensity))
                            weekly_evaluation.append('好：收集籌碼')
                        elif price_change < 0 and sh_change > 0:
                            bar_colors.append(bad_cmap(intensity))
                            weekly_evaluation.append('差：分散籌碼')
                        else:
                            bar_colors.append(neutral_cmap(intensity))
                            direction = '同向' if (price_change > 0 and sh_change > 0) or (price_change < 0 and sh_change < 0) else '混合'
                            weekly_evaluation.append(f'中性：{direction}')
                    else:
                        intensity = min(abs(sh_change) / 3, 1.0)
                        if sh_change < 0:
                            bar_colors.append(good_cmap(intensity))
                            weekly_evaluation.append('人數減少')
                        else:
                            bar_colors.append(bad_cmap(intensity))
                            weekly_evaluation.append('人數增加')
                
                # 柱狀圖
                bars = ax.bar(recent_dates, shareholders_recent, width=5, alpha=0.6, color=bar_colors, 
                            edgecolor='#333333', label='總股東人數')
                
                # 折線圖
                line1 = ax2.plot(recent_dates, ratio_400plus_recent, 'o-', color='#FF6666', 
                                linewidth=2.5, markersize=6, label='大股東持有率(>400張)')
                line2 = ax2.plot(recent_dates, ratio_1000plus_recent, 'o-', color='#6699CC', 
                                linewidth=2.5, markersize=6, label='大股東持有率(>1000張)')
                
                if has_price_data:
                    ax3 = ax.twinx()
                    ax3.spines['right'].set_position(('outward', 60))
                    line3 = ax3.plot(recent_dates, aligned_prices, 'o--', color='#9933CC', 
                                    linewidth=2.5, markersize=6, label='股價')
                    ax3.set_ylabel('股\n價', rotation=0, labelpad=25, fontsize=20)
                    price_min, price_max = aligned_prices.min(), aligned_prices.max()
                    price_range = price_max - price_min
                    ax3.set_ylim(price_min - price_range * 0.1, price_max + price_range * 0.1)
                
                # 趨勢分析
                trend_analysis = []
                if len(recent_dates) >= 5:
                    from scipy import stats
                    x = np.arange(len(ratio_400plus_recent))
                    slope_400, _, _, p_value_400, _ = stats.linregress(x, ratio_400plus_recent)
                    trend_400 = "大股東(>400張)持股比例整體無明顯趨勢" if p_value_400 >= 0.05 else \
                                f"大股東(>400張)持股比例整體呈{'上升' if slope_400 > 0 else '下降'}趨勢 ({slope_400:.3f}%/週)"
                    trend_analysis.append(trend_400)
                    recent_weeks = min(4, len(recent_dates)-1)
                    recent_400plus_change = ratio_400plus_recent.iloc[-1] - ratio_400plus_recent.iloc[-recent_weeks-1]
                    trend_analysis.append(f"近{recent_weeks}週大股東(>400張)持股比例變化: {recent_400plus_change:.2f}%")
                    last_week_400_change = ratio_400plus_recent.iloc[-1] - ratio_400plus_recent.iloc[-2]
                    last_week_1000_change = ratio_1000plus_recent.iloc[-1] - ratio_1000plus_recent.iloc[-2]
                    last_week_sh_change = shareholders_recent.iloc[-1] - shareholders_recent.iloc[-2]
                    last_week_sh_pct_change = last_week_sh_change / shareholders_recent.iloc[-2] * 100
                    if has_price_data:
                        last_week_price_change = aligned_prices.iloc[-1] - aligned_prices.iloc[-2]
                        last_week_price_pct_change = last_week_price_change / aligned_prices.iloc[-2] * 100
                        status = "好：股價上升+人數減少" if last_week_price_pct_change > 0 and last_week_sh_pct_change < 0 else \
                                "差：股價下降+人數增加" if last_week_price_pct_change < 0 and last_week_sh_pct_change > 0 else "中性"
                        trend_analysis.append(f"最新一週: 股價變化 {last_week_price_pct_change:.2f}%, 股東人數變化 {last_week_sh_pct_change:.2f}% - {status}")
                    else:
                        status = "人數減少" if last_week_sh_pct_change < 0 else "人數增加"
                        trend_analysis.append(f"最新一週: 股東人數變化 {last_week_sh_pct_change:.2f}% - {status}")
                
                # 標題
                title = f'{stock_id} {stock_name} 大股東持股分佈與總股東人數分析 (近{weeks}週)'
                all_text = title + "\n\n" + "\n".join(trend_analysis)
                ax.text(0.5, 1.15, all_text, horizontalalignment='center', verticalalignment='bottom',
                        transform=ax.transAxes, fontsize=30, fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'), multialignment='left')
                
                # 軸標籤
                ax.set_xlabel('')
                ax.set_ylabel('股\n東\n人\n數', rotation=0, labelpad=25, fontsize=20)
                ax2.set_ylabel('持\n股\n比\n例\n(%)', rotation=0, labelpad=25, fontsize=20)
                
                # 柱子註釋
                if len(recent_dates) <= 15:
                    shareholders_max = shareholders_recent.max()
                    for i, (rect, eval_text) in enumerate(zip(bars, weekly_evaluation)):
                        if i > 0:
                            height = rect.get_height()
                            ax.text(rect.get_x() + rect.get_width()/2., height + 0.01 * shareholders_max,
                                    eval_text, ha='center', va='bottom', rotation=90, fontsize=14, color='#333333')
                
                # 圖例
                legend_elements = [
                    patches.Patch(facecolor='#CC3333', edgecolor='#333333', alpha=0.6, label='好：股價↑人數↓'),
                    patches.Patch(facecolor='#336633', edgecolor='#333333', alpha=0.6, label='差：股價↓人數↑'),
                    patches.Patch(facecolor='#666666', edgecolor='#333333', alpha=0.6, label='中性：同向變動'),
                    Line2D([0], [0], color='#FF6666', linewidth=2.5, marker='o', label='大股東(>400張)持股'),
                    Line2D([0], [0], color='#6699CC', linewidth=2.5, marker='o', label='大股東(>1000張)持股')
                ]
                if has_price_data:
                    legend_elements.append(Line2D([0], [0], color='#9933CC', linewidth=2.5, marker='o', linestyle='--', label='股價'))
                ax.legend(legend_elements, [e.get_label() for e in legend_elements], loc='upper left', 
                        bbox_to_anchor=(1.05, 1), fontsize=14, frameon=True, framealpha=0.8)
                
                # X軸格式
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=4))
                ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=18)
                
                # Y軸範圍（修復部分）
                shareholders_min, shareholders_max = shareholders_recent.min(), shareholders_recent.max()
                shareholders_range = shareholders_max - shareholders_min
                y_min = max(0, shareholders_min * 0.95)
                y_max_initial = shareholders_min + (shareholders_range * 1.5)
                y_max = y_max_initial if y_max_initial >= shareholders_max else shareholders_max * 1.1
                ax.set_ylim(y_min, y_max)
                
                # 網格與背景
                ax.grid(True, alpha=0.3, linestyle='dotted')
                ax.set_facecolor('#F5F5F5')
                plt.tight_layout(rect=[0, 0, 0.85, 0.85])
                
            except Exception as e:
                import traceback
                print(f"繪製大股東持股分佈圖時發生錯誤: {str(e)}")
                print(traceback.format_exc())
                ax.text(0.5, 0.5, f"繪製圖表時發生錯誤:\n{str(e)}", ha='center', va='center', 
                        transform=ax.transAxes, fontsize=14)


  
    def plot_average_holdings_analysis(self, ax, stock_id, weeks=50):
        """
        專門分析平均持股和股價的關係，使用柱狀圖顯示平均持股，折線圖顯示股價
        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.lines import Line2D
        import matplotlib.patches as mpatches
        
        try:
            # 數據處理部分保持不變
            stock_info = self.stock_topics_df[self.stock_topics_df['stock_no'] == int(stock_id)]
            stock_name = stock_info['stock_name'].values[0] if not stock_info.empty else stock_id
            inv = self.data.get('inventory')
            if inv is None:
                raise ValueError("無法獲取持股資料")
            
            inv['date'] = pd.to_datetime(inv['date'], errors='coerce')
            inv['持有股數'] = pd.to_numeric(inv['持有股數'], errors='coerce')
            inv['持股分級'] = pd.to_numeric(inv['持股分級'], errors='coerce')
            inv['人數'] = pd.to_numeric(inv['人數'], errors='coerce')
            
            all_holders = inv[inv.持股分級 == 17].groupby(['date', 'stock_id'])['持有股數'].sum().reset_index()
            total_shareholders = inv[inv.持股分級 == 17].groupby(['date', 'stock_id'])['人數'].sum().reset_index()
            
            all_holders_pivot = all_holders.pivot(index='date', columns='stock_id', values='持有股數')
            total_shareholders_pivot = total_shareholders.pivot(index='date', columns='stock_id', values='人數')
            
            dates = all_holders_pivot.index.intersection(total_shareholders_pivot.index)
            end_date = dates[-1]
            start_date = end_date - pd.Timedelta(weeks=weeks-1)
            date_mask = (dates >= start_date) & (dates <= end_date)
            if sum(date_mask) < 2:
                raise ValueError(f"沒有足夠的數據進行分析，只有{sum(date_mask)}筆資料")
            
            recent_dates = dates[date_mask]
            shareholders_recent = total_shareholders_pivot.loc[recent_dates, stock_id]
            avg_holdings = all_holders_pivot.loc[recent_dates, stock_id] / shareholders_recent / 1000
            current_avg = avg_holdings.iloc[-1]
            first_avg = avg_holdings.iloc[0]
            change_pct = (current_avg - first_avg) / first_avg * 100 if first_avg != 0 else 0
            
            has_price_data = False
            try:
                close_data = self.data.get("price:收盤價")[stock_id]
                if close_data is not None:
                    close_data.index = pd.to_datetime(close_data.index)
                    mask_price = (close_data.index >= recent_dates[0]) & (close_data.index <= recent_dates[-1])
                    filtered_prices = close_data.loc[mask_price]
                    aligned_prices = pd.Series(index=recent_dates)
                    for date in recent_dates:
                        closest_price_idx = close_data.index.get_indexer([date], method='nearest')[0]
                        if closest_price_idx >= 0 and closest_price_idx < len(close_data):
                            aligned_prices[date] = close_data.iloc[closest_price_idx]
                    has_price_data = True
            except:
                has_price_data = False
            
            # 週間變化與配色
            weekly_changes = avg_holdings.pct_change() * 100
            weekly_changes.iloc[0] = 0
            max_change = max(abs(weekly_changes.iloc[1:]).max(), 0.5)
            small_threshold = max_change / 3
            medium_threshold = max_change * 2 / 3
            
            colors_increase = ['#FFCCCC', '#FF9999', '#CC3333']  # 好：紅色
            colors_decrease = ['#CCFFCC', '#99CC99', '#336633']  # 差：綠色
            
            def get_color(change_pct):
                if change_pct > 0:
                    if change_pct > medium_threshold:
                        return colors_increase[2]
                    elif change_pct > small_threshold:
                        return colors_increase[1]
                    else:
                        return colors_increase[0]
                else:
                    change_pct = abs(change_pct)
                    if change_pct > medium_threshold:
                        return colors_decrease[2]
                    elif change_pct > small_threshold:
                        return colors_decrease[1]
                    else:
                        return colors_decrease[0]
            
            bar_colors = [get_color(change) for change in weekly_changes]
            bars = ax.bar(recent_dates, avg_holdings, width=5, color=bar_colors, 
                        edgecolor='#333333', alpha=0.6, linewidth=0.5)
            
            # 週間變化標註
            for i, (bar, change) in enumerate(zip(bars, weekly_changes)):
                if i > 0:
                    change_text = f"{change:.2f}%" if abs(change) < 1 else f"{change:.1f}%"
                    text_color = 'black' if abs(change) < medium_threshold else 'white'
                    height = bar.get_height()
                    ax.annotate(change_text, xy=(bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 5 if height > avg_holdings.mean() else -15), 
                                textcoords='offset points', ha='center', 
                                va='bottom' if height > avg_holdings.mean() else 'top',
                                fontsize=12, fontweight='bold', color=text_color)
            
            # 起始與終點標註
            ax.annotate(f"{first_avg:.2f}", xy=(recent_dates[0], first_avg), xytext=(0, 5), 
                        textcoords='offset points', fontsize=14, color='#333333', fontweight='bold', ha='center')
            ax.annotate(f"{current_avg:.2f}", xy=(recent_dates[-1], current_avg), xytext=(0, 5), 
                        textcoords='offset points', fontsize=14, color='#333333', fontweight='bold', ha='center')
            
            # 股價折線
            if has_price_data:
                ax2 = ax.twinx()
                line_price = ax2.plot(recent_dates, aligned_prices, 'o-', color='#336699', 
                                    linewidth=2.5, markersize=6, label='股價')
                ax2.annotate(f"{aligned_prices.iloc[0]:.1f}", xy=(recent_dates[0], aligned_prices.iloc[0]), 
                            xytext=(10, -20), textcoords='offset points', fontsize=14, color='#336699')
                ax2.annotate(f"{aligned_prices.iloc[-1]:.1f}", xy=(recent_dates[-1], aligned_prices.iloc[-1]), 
                            xytext=(10, -20), textcoords='offset points', fontsize=14, color='#336699')
            
            # 趨勢分析
            trend_analysis = []
            if len(avg_holdings) >= 8:
                from scipy import stats
                x = np.arange(len(avg_holdings))
                slope_avg, _, _, p_value_avg, _ = stats.linregress(x, avg_holdings)
                trend_avg = "平均持股張數無明顯趨勢變化" if p_value_avg >= 0.05 else \
                            f"平均持股張數呈{'增加' if slope_avg > 0 else '減少'}趨勢 (+{slope_avg:.3f}張/週)"
                trend_analysis.append(trend_avg)
                if has_price_data:
                    corr = avg_holdings.corr(aligned_prices)
                    correlation = f"平均持股與股價{'正' if corr > 0 else '負'}相關 (相關係數: {corr:.2f})" if abs(corr) > 0.6 else \
                                f"平均持股與股價相關性不強 (相關係數: {corr:.2f})"
                    trend_analysis.append(correlation)
            
            # 標題
            title = f'{stock_id} {stock_name} 平均持股分析 (近{weeks}週)'
            subtitle = f"期初: {first_avg:.2f}張/人 → 期末: {current_avg:.2f}張/人 (變化 {change_pct:+.2f}%)"
            all_text = title + "\n" + subtitle + "\n\n" + "\n".join(trend_analysis)
            ax.text(0.5, 1.12, all_text, horizontalalignment='center', verticalalignment='bottom',
                    transform=ax.transAxes, fontsize=30, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'), multialignment='left')
            
            # 軸標籤
            ax.set_xlabel('')
            ax.set_ylabel('平\n均\n持\n股\n(張\n/\n人)', fontsize=20, rotation=0, labelpad=20)
            if has_price_data:
                ax2.set_ylabel('股\n價', fontsize=20, rotation=0, labelpad=20)
            
            # 圖例
            gradient_legend = [
                mpatches.Patch(facecolor=colors_increase[2], edgecolor='#333333', alpha=0.6, label=f'大幅增加 (+{medium_threshold:.2f}%以上)'),
                mpatches.Patch(facecolor=colors_increase[1], edgecolor='#333333', alpha=0.6, label=f'中幅增加 (+{small_threshold:.2f}~{medium_threshold:.2f}%)'),
                mpatches.Patch(facecolor=colors_increase[0], edgecolor='#333333', alpha=0.6, label=f'小幅增加 (0~+{small_threshold:.2f}%)'),
                mpatches.Patch(facecolor=colors_decrease[0], edgecolor='#333333', alpha=0.6, label=f'小幅減少 (0~-{small_threshold:.2f}%)'),
                mpatches.Patch(facecolor=colors_decrease[1], edgecolor='#333333', alpha=0.6, label=f'中幅減少 (-{small_threshold:.2f}~-{medium_threshold:.2f}%)'),
                mpatches.Patch(facecolor=colors_decrease[2], edgecolor='#333333', alpha=0.6, label=f'大幅減少 (-{medium_threshold:.2f}%以上)')
            ]
            if has_price_data:
                gradient_legend.append(Line2D([0], [0], color='#336699', linewidth=2.5, marker='o', label='股價'))
            ax.legend(handles=gradient_legend, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=14, 
                    frameon=True, framealpha=0.8, title='周間變化率 (最大變化: ±{:.2f}%)'.format(max_change))
            
            # X軸格式
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=4))
            ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=18)
            
            # Y軸範圍
            avg_min, avg_max = avg_holdings.min(), avg_holdings.max()
            avg_range = avg_max - avg_min
            if avg_range / avg_min < 0.05:
                middle = (avg_max + avg_min) / 2
                y_min = max(0, middle - (avg_max - avg_min) * 1.5)
                y_max = middle + (avg_max - avg_min) * 2
                ax.set_ylim(y_min, y_max)
            else:
                y_max = avg_max + avg_range * 0.5
                ax.set_ylim(0, y_max)
            
            # 網格與背景
            ax.grid(True, alpha=0.3, linestyle='dotted')
            ax.set_facecolor('#F5F5F5')
            plt.tight_layout(rect=[0, 0, 1, 0.85])
            
        except Exception as e:
            import traceback
            print(f"繪製平均持股分析圖時發生錯誤: {str(e)}")
            print(traceback.format_exc())
            ax.text(0.5, 0.5, f"繪製圖表時發生錯誤:\n{str(e)}", ha='center', va='center', 
                    transform=ax.transAxes, fontsize=14)



    def plot_internal_holding_changes(self, ax, stock_id, months=50):
        """
        繪製董監持股比例變化圖（近N月），加上異常董監加碼與減碼標記以及股價
        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.ticker import FuncFormatter

        try:
            stock_info = self.stock_topics_df[self.stock_topics_df['stock_no'] == int(stock_id)]
            stock_name = stock_info['stock_name'].values[0] if not stock_info.empty else stock_id

            # 取得資料
            director_ratio = self.data.get('internal_equity_changes:董監持有股數占比')[stock_id]
            director_shares = self.data.get('internal_equity_changes:董監持有股數')[stock_id]
            close = self.data.get('price:收盤價')[stock_id].resample('M').mean()

            # 整合資料
            df = pd.concat([
                director_ratio.rename('董監持股占比(%)'),
                director_shares.rename('董監持有張數'),
                close.rename('收盤價')
            ], axis=1).dropna().tail(months)

            df.index = pd.to_datetime(df.index)
            
            # 計算董監持股比例變化（與前一個月相比）
            df['占比變化'] = df['董監持股占比(%)'].diff()

            # 畫圖
            ax2 = ax.twinx()  # 用於繪製股價
            ax3 = ax.twinx()  # 用於繪製董監持有張數
            ax3.spines['right'].set_position(('outward', 60))
            
            # 根據董監持股比例變化設置柱狀圖顏色（上升淺紅色，下降淺綠色）
            colors = ['#ff9999' if change > 0 else '#99cc99' for change in df['占比變化'].fillna(0)]
            
            # 設定Y軸範圍以聚焦細微波動
            y_min = df['董監持股占比(%)'].min() - 2  # 下擴2%空間
            y_max = df['董監持股占比(%)'].max() + 2  # 上擴2%空間
            ax.set_ylim(y_min, y_max)

            # 繪製董監持股占比柱狀圖，寬度加粗至10以強化視覺效果
            ax.bar(df.index, df['董監持股占比(%)'], width=10, label='董監持股占比(%)', 
                color=colors, edgecolor='black', linewidth=1, alpha=0.8)
            
            # 繪製董監持有張數
            ax3.plot(df.index, df['董監持有張數']/1000, label='董監持有張數(千張)', 
                    color='#ff7f0e', linewidth=2.5)
            
            # 繪製收盤價折線
            ax2.plot(df.index, df['收盤價'], label='收盤價(月均)', 
                    color='#8c564b', linewidth=2.5, linestyle='-.')

            # 添加異常減碼標記（以占比變化）
            diff_series = df['占比變化'].rolling(4).mean()  # 使用3期移動平均濾除雜訊
            mean_diff = diff_series.mean()
            std_diff = diff_series.std()
            
            jump_mask_up = diff_series > (mean_diff + 1.5 * std_diff)  # 異常加碼標準
            jump_mask_down = diff_series < (mean_diff - 1.5 * std_diff)  # 異常減碼標準
            
            jump_dates_up = df.index[jump_mask_up]
            jump_dates_down = df.index[jump_mask_down]

            for date in jump_dates_up:
                ax.annotate('↑\n異\n常\n加\n碼',
                            xy=(date, df.loc[date, '董監持股占比(%)']),
                            xytext=(0, 10),
                            textcoords='offset points',
                            ha='center', color='red', fontsize=13,
                            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

            for date in jump_dates_down:
                ax.annotate('↓\n異\n常\n減\n碼',
                            xy=(date, df.loc[date, '董監持股占比(%)']),
                            xytext=(0, 10),
                            textcoords='offset points',
                            ha='center', color='blue', fontsize=13,
                            arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

            # 標題與統計說明
            corr = df['董監持股占比(%)'].corr(df['收盤價'])
            title = f'{stock_id} {stock_name} 董監持股變化圖（近{months}月）\n董監占比 vs 股價相關係數：{corr:.2f}'
            ax.set_title(title, fontsize=30, fontweight='bold', pad=20)

            # Y軸設定與格式化
            ax.set_ylabel('董\n監\n持\n股\n占\n比\n(%)', rotation=0, labelpad=25, fontsize=20)
            ax3.set_ylabel('董\n監\n持\n有\n張\n數\n(千\n張)', rotation=0, labelpad=25, fontsize=20)
            ax2.set_ylabel('收\n盤\n價', rotation=0, labelpad=20, fontsize=20)

            ax.tick_params(axis='y', labelcolor='blue')
            ax3.tick_params(axis='y', labelcolor='#ff7f0e')
            ax2.tick_params(axis='y', labelcolor='#8c564b')

            ax.yaxis.set_major_locator(plt.MultipleLocator(1))  # 每1%主刻度
            ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))  # 每0.2%次刻度
            ax.grid(which='minor', linestyle=':', alpha=0.5)  # 次網格線

            # X軸每三個月顯示一次，字體大小調整為20
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.tick_params(axis='x', rotation=45, labelsize=20)  # 日期字體大小設為20

            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_facecolor('#F9F9F9')

            # 格式化Y軸刻度顯示方式
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}%'))
            ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:,.0f}'))
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:,.0f}'))

        except Exception as e:
            import traceback
            print(f"繪圖錯誤: {str(e)}")

        








# ========================================市場近20日強弱圖======================================================================================

    def calculate_strength_index(self, close: pd.Series) -> pd.Series:
        """
        計算股票的強勢指數。

        Args:
            close (pd.Series): 股票的收盤價序列。

        Returns:
            pd.Series: 布林值序列，True 表示符合強勢條件，False 表示不符合。
        """
        condition1 = close > close.rolling(5).mean()
        condition2 = close.rolling(5).mean() > close.rolling(20).mean()
        return condition1 & condition2

 

    def load_stock_names(self, excel_file_path: str) -> Dict[str, str]:
        """
        從Excel檔案讀取股票代號和公司名稱的對應關係。

        Args:
            excel_file_path (str): Excel檔案的路徑。

        Returns:
            Dict[str, str]: 股票代號到公司名稱的字典。
        """
        try:
            df = pd.read_excel(excel_file_path, dtype=str)
            # 假設股票代號在 'stock_no' 欄位，公司名稱在 'stock_name' 欄位
            return {row['stock_no']: row['stock_name'] for index, row in df.iterrows()}
        except FileNotFoundError:
            print(f"找不到檔案: {excel_file_path}")
            return {}
        except Exception as e:
            print(f"讀取股票名稱時發生錯誤: {str(e)}")
            return {}

    def get_ind_stocks(self, industry_csv_path: str) -> Dict[str, List[str]]:
        df = pd.read_csv(industry_csv_path, dtype=str)
        return {ind: df.loc[df["ind"] == ind, "code"].to_list() for ind in df["ind"].unique()}

    def calculate_industry_returns(self, close: pd.DataFrame,
                                ind_stocks: Dict[str, List[str]],
                                period_days: int = 20) -> pd.Series:
        industry_returns = {}
        for ind, codes in ind_stocks.items():
            valid_codes = [code for code in codes if code in close.columns]
            if len(valid_codes) > 3:
                stock_returns = close[valid_codes].iloc[-period_days:].pct_change(fill_method=None).fillna(0)
                cum_returns = (1 + stock_returns).prod() - 1
                industry_returns[ind] = cum_returns.mean() * 100  # 轉換為百分比
            else:
                print(f"警告: 產業 {ind} 的有效股票代碼少於 4 個，已跳過。")
        return pd.Series(industry_returns)

    def get_stock_returns(self, close: pd.DataFrame,
                            stock_codes: List[str],
                            period_days: int = 20) -> pd.Series:
        valid_codes = [code for code in stock_codes if code in close.columns]
        if not valid_codes:
            return pd.Series()

        # 計算強勢指數
        strength_indices = {}
        for code in valid_codes:
            try:
                strength_indices[code] = self.calculate_strength_index(close[code].dropna())  # 移除 NaN 值
            except Exception as e:
                print(f"警告: 計算股票 {code} 的強勢指數時發生錯誤: {str(e)}")
                strength_indices[code] = pd.Series([False] * len(close[code].dropna()), index=close[code].dropna().index) # 預設為不符合強勢條件

        # 篩選符合強勢指數條件的股票
        filtered_codes = [code for code in valid_codes if code in strength_indices and strength_indices[code].iloc[-1]]  # 僅保留最後一天符合條件的股票

        if not filtered_codes:
            return pd.Series()

        returns = close[filtered_codes].iloc[-period_days:].pct_change(fill_method=None).fillna(0)
        cum_returns = (1 + returns).prod() - 1

        return cum_returns * 100  # 轉換為百分比


    def plot_industry_performance(self, fig, period_days=5):
        try:
            # 1️⃣ 取得產業與股票數據
            ind_stocks = self.get_ind_stocks("industry.csv")
            stock_names = self.load_stock_names("tw_stock_topics.xlsx")
            close_data = self.data.get("etl:adj_close")

            # 2️⃣ 計算不同時間窗口的產業報酬率
            industry_returns_5d = self.calculate_industry_returns(close_data, ind_stocks, period_days=5)
            industry_returns_10d = self.calculate_industry_returns(close_data, ind_stocks, period_days=10)
            industry_returns_20d = self.calculate_industry_returns(close_data, ind_stocks, period_days=20)
            industry_returns_60d = self.calculate_industry_returns(close_data, ind_stocks, period_days=60)

            # 3️⃣ 計算 5 日均價
            close_5d_avg = close_data.rolling(5).mean().iloc[-1]

            # 4️⃣ 確保有數據
            common_inds = industry_returns_5d.index.intersection(industry_returns_10d.index).intersection(
                industry_returns_20d.index).intersection(industry_returns_60d.index)

            if len(common_inds) == 0:
                print("❗ 沒有可用的產業數據")
                return

            # 5️⃣ 產業過濾條件：
            filtered_inds = []
            for ind in common_inds:
                if (industry_returns_10d[ind] > 10) and (industry_returns_20d[ind] < 30):
                    # 檢查該產業的主要股票是否符合「目前收盤價 > 5日均價」
                    stock_codes = ind_stocks.get(ind, [])
                    valid_stocks = [code for code in stock_codes if code in close_data.columns and close_data[code].iloc[-1] > close_5d_avg[code]]

                    if len(valid_stocks) > 0:
                        filtered_inds.append(ind)

            # 過濾後的產業
            industry_returns_filtered = industry_returns_5d[filtered_inds]

            # 6️⃣ 取得前 10 強 & 最弱 10 名產業
            top10_industries = industry_returns_filtered.nlargest(10)
            bottom10_industries = industry_returns_filtered.nsmallest(10)

            # 7️⃣ 設置子圖的 Grid (6 行 2 列)
            gs = fig.add_gridspec(6, 2, height_ratios=[1, 1, 1, 1, 1, 1], hspace=0.4, wspace=0.3)

            # 8️⃣ 繪製強勢產業（正報酬用紅色）
            ax_strong = fig.add_subplot(gs[0, 0])
            self.plot_industry_bar(
                ax_strong, 
                top10_industries, 
                f"近{period_days}日最強產業 (Top 1-10名)", 
                "RedGradient"  # 正報酬用紅色
            )

            # 9️⃣ 繪製弱勢產業（負報酬用綠色）
            ax_weak = fig.add_subplot(gs[0, 1])
            self.plot_industry_bar(
                ax_weak, 
                bottom10_industries, 
                f"近{period_days}日最弱產業 (Bottom 1-10名)", 
                "GreenGradient"  # 負報酬用綠色
            )

            # 🔟 定義10種不同的產業顏色
            industry_colors = [
                '#FF3333',  # 紅色
                '#FF6600',  # 橙色
                '#FF9900',  # 橙黃色
                '#FFCC00',  # 金黃色
                '#FF66FF',  # 粉紅色
                '#9933FF',  # 紫色
                '#3333FF',  # 藍色
                '#0099FF',  # 天藍色
                '#00CC99',  # 青色
                '#66CC33'   # 綠色
            ]

            # 11 繪製強勢產業內最強個股
            for i, (ind, ret) in enumerate(top10_industries.items()):
                row = (i // 2) + 1
                col = i % 2

                if row < 6 and col < 2:
                    ax_stock = fig.add_subplot(gs[row, col])
                    stock_returns = self.get_stock_returns(close_data, ind_stocks[ind], period_days=5)

                    if not stock_returns.empty:
                        top_stocks = stock_returns.nlargest(5)
                        # 使用對應的產業顏色
                        color = industry_colors[i]
                        title = f"第{i+1}強產業: {ind}\n報酬率: {ret:.1f}%"
                        self.plot_stock_bar(
                            ax_stock, 
                            top_stocks, 
                            title, 
                            color, 
                            stock_names
                        )
                    else:
                        print(f"警告: 產業 {ind} 沒有股票回報數據，已跳過。")
                else:
                    print(f"警告: 產業 {ind} (索引 {i}) 的子圖位置超出範圍，已跳過。")

        except Exception as e:
            print(f"繪製產業表現時發生錯誤: {str(e)}")



    def plot_industry_bar(self, ax, data, title, cmap_name):
        y_pos = np.arange(len(data))
        
        # 直接根據報酬率決定顏色，不需要cmap相關的代碼
        bars = ax.barh(y_pos, data.values, 
                    color=['#FF4444' if x > 0 else '#44FF44' for x in data.values], 
                    alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(data.index, fontsize=25)
        ax.set_title(title, fontsize=36, fontweight='bold', pad=20)
        ax.set_xlabel("報酬率 (%)", fontsize=25)
        ax.tick_params(axis='x', labelsize=25)
        ax.grid(True, linestyle="--", alpha=0.3)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width/2, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%', ha='center', va='center', 
                    fontsize=18, color='black')


            
    def plot_stock_bar(self, ax, data, industry_name, color, stock_names):
        y_pos = np.arange(len(data))
        bars = ax.barh(y_pos, data.values, color=color, alpha=0.8)
        
        # 合併公司名稱和股票代號
        labels = [f"{stock_names.get(code, code)} ({code})" for code in data.index]
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=20)  # 放大字體
        ax.set_title(f'{industry_name}', fontsize=26, fontweight='bold', pad=15)
        ax.set_xlabel("報酬率 (%)", fontsize=24)  # 放大字體
        ax.tick_params(axis='both', labelsize=24)  # 放大字體
        ax.grid(True, linestyle="--", alpha=0.3)
        for bar in bars:
            width = bar.get_width()
            ax.text(width/2, bar.get_y() + bar.get_height()/2,  # 調整文字位置到柱狀圖內
                    f'{width:.1f}%', ha='center', va='center', fontsize=16, color='black')  # 調整文字顏色
            

    def plot_industry_trend(self, fig):
        try:
            # 1️⃣ 取得產業與股票數據
            ind_stocks = self.get_ind_stocks("industry.csv")  # 取得產業內的股票清單
            stock_names = self.load_stock_names("tw_stock_topics.xlsx")  # 取得股票代號與公司名稱的對應關係
            close_data = self.data.get("etl:adj_close")

            # 檢查資料是否為空
            if close_data is None or close_data.empty:
                print("❗ 收盤價資料為空，無法計算產業報酬率")
                return

            # 2️⃣ 計算不同時間窗口的產業報酬率
            industry_returns_5d = self.calculate_industry_returns(close_data, ind_stocks, period_days=5)
            industry_returns_10d = self.calculate_industry_returns(close_data, ind_stocks, period_days=10)
            industry_returns_20d = self.calculate_industry_returns(close_data, ind_stocks, period_days=20)
            industry_returns_60d = self.calculate_industry_returns(close_data, ind_stocks, period_days=60)

            # 3️⃣ 計算 5 日均價
            close_5d_avg = close_data.rolling(5).mean().iloc[-1]

            # 4️⃣ 確保有數據
            common_inds = industry_returns_5d.index.intersection(industry_returns_10d.index).intersection(
                industry_returns_20d.index).intersection(industry_returns_60d.index)

            if len(common_inds) == 0:
                print("❗ 沒有可用的產業數據")
                return

            # 5️⃣ 產業過濾條件：
            filtered_inds = []
            for ind in common_inds:
                # 條件 1: 10日報酬率 > 10 且 20日報酬率 < 30
                condition1 = (industry_returns_10d[ind] > 10) and (industry_returns_20d[ind] < 30)

                if condition1:
                    # 檢查該產業的主要股票是否符合「目前收盤價 > 5日均價」
                    stock_codes = ind_stocks.get(ind, [])
                    valid_stocks = [
                        code for code in stock_codes
                        if code in close_data.columns and close_data[code].iloc[-1] > close_5d_avg[code]
                    ]

                    if len(valid_stocks) > 0:
                        filtered_inds.append(ind)

            # 過濾後的產業
            trend_df = pd.DataFrame({
                "5日報酬率": industry_returns_5d[filtered_inds],
                "20日報酬率": industry_returns_20d[filtered_inds],
                "60日報酬率": industry_returns_60d[filtered_inds]
            }).sort_values("5日報酬率", ascending=False).head(10)  # 選前 10 名

            # 6️⃣ 設定更細緻的趨勢標記
            trend_labels = []
            stock_labels = []
            for ind in trend_df.index:
                r5, r20, r60 = trend_df.loc[ind, "5日報酬率"], trend_df.loc[ind, "20日報酬率"], trend_df.loc[ind, "60日報酬率"]

                # 判斷產業趨勢 (保持不變)
                if r5 > r20 > r60:
                    trend_labels.append("持續增強")
                elif r5 > r60 > r20:
                    trend_labels.append("短期回調但長期仍強")
                elif r60 > r20 > r5:
                    trend_labels.append("短期反彈但長期仍弱")
                elif r5 < r20 < r60:
                    trend_labels.append("持續衰退")
                elif r20 > r5 > r60:
                    trend_labels.append("短期強勢但長期趨勢不明確")
                elif r20 < r5 < r60:
                    trend_labels.append("短期弱勢但長期仍向上")
                else:
                    trend_labels.append("趨勢震盪")

                # 取得該產業內的前三強個股
                top_stocks = self.get_stock_returns(close_data, ind_stocks[ind], period_days=5)
                if not top_stocks.empty:
                    # 取得前三名個股
                    top3_stocks = top_stocks.nlargest(3)
                    stock_info = []
                    for rank, (code, ret) in enumerate(top3_stocks.items(), 1):
                        name = stock_names.get(code, code)
                        stock_info.append(f"#{rank} {name}({code})")
                    stock_labels.append("\n".join(stock_info))  # 用換行符號分隔三支股票
                else:
                    stock_labels.append("無資料")

            # 7️⃣ 建立 X 軸的位置
            x_pos = np.arange(len(trend_df))

            # 8️⃣ 設定長條圖的寬度
            bar_width = 0.25

            # 9️⃣ 繪製長條圖
            ax = fig.add_subplot(111)

            bars_5d = ax.bar(x_pos - bar_width, trend_df["5日報酬率"], bar_width, color="#FF5733", label="5日", alpha=0.8,
                            edgecolor="black")
            bars_20d = ax.bar(x_pos, trend_df["20日報酬率"], bar_width, color="#337AB7", label="20日", alpha=0.8,
                            edgecolor="black")
            bars_60d = ax.bar(x_pos + bar_width, trend_df["60日報酬率"], bar_width, color="#33A85D", label="60日", alpha=0.8,
                            edgecolor="black")

            # 🔟 設定 X 軸標籤（產業 + 趨勢 + 前三強個股）
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"{ind}\n{trend}\n{stock}" for ind, trend, stock in
                                zip(trend_df.index, trend_labels, stock_labels)],
                            rotation=0, ha="center", fontsize=25, fontweight="bold")  # 字體稍微調小以適應更多文字

            # 🔹 放大標題 & Y 軸標籤
            ax.set_ylabel("報\n酬\n率\n(%)", fontsize=28, rotation=0, fontweight="bold", labelpad=50)

            # 🔹 設定圖例
            ax.legend(fontsize=28)

            # 🔹 在長條圖上標示數值
            for bars in [bars_5d, bars_20d, bars_60d]:
                for bar in bars:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=23, fontweight="bold")

            # 調整圖表大小以容納更多文字
            plt.subplots_adjust(bottom=0.25)  # 增加底部空間以容納更多文字

        except Exception as e:
            print(f"⚠️ 繪製產業趨勢變化時發生錯誤: {str(e)}")

 
# ========================================pdf==============================================================================


    def create_pdf(self):
        """創建包含所有圖表的PDF文件，每10張圖產生一個臨時PDF，最後合併"""
        import os
        from PyPDF2 import PdfMerger
        
        # 建立年月資料夾結構
        today = datetime.today()
        year_folder = str(today.year)
        month_folder = f"{today.month:02d}"
        
        # 建立基礎路徑
        base_path = "reports"
        
        # 建立完整的資料夾路徑 (只到月份)
        folder_path = os.path.join(base_path, year_folder, month_folder)
        
        # 確保資料夾存在
        os.makedirs(folder_path, exist_ok=True)
        
        today_date = today.strftime('%Y%m%d')
        final_file_name = os.path.join(folder_path, f"tomstrategy_{today_date}.pdf")
        
        # today_date = datetime.today().strftime('%Y%m%d')
        # final_file_name = f"tomstrategy_{today_date}.pdf"
        # 建立臨時檔案資料夾
        temp_folder = os.path.join(folder_path, "temp")
        os.makedirs(temp_folder, exist_ok=True)
        temp_pdf_files = []  # 儲存臨時PDF檔案名稱
        
        # 統一定義頁面尺寸和間距
        PAGE_WIDTH = 48
        PAGE_HEIGHT_STANDARD = 35
        PAGE_HEIGHT_LARGE = 45

        # 定義統一的字體大小和間距
        TITLE_SIZE = 40
        AXIS_LABEL_SIZE = 26
        TICK_LABEL_SIZE = 24
        STRATEGY_TEXT_SIZE = 24
        COUNT_TEXT_SIZE = 22
        BAR_LABEL_SIZE = 20

        # 定義間距
        TITLE_PAD = 30
        LABEL_PAD = 30
        LAYOUT_PAD = 1.0  # 減少頁面邊緣的空白，讓圖表稍微放大
        GRID_HSPACE = 0.8  # 保留原有的子圖高度間距
        GRID_WSPACE = 0.3  # 保留原有的子圖寬度間距

        # 策略說明字典
        strategy_descriptions = {
            "GVI": "ROE*BP_ratio*revenue關係分析、大戶買散戶賣",
            "小資": "低波動、強營收、強動能、小市值",
            "小資yoy": "低波動、強營收、強動能、最強YOY",
            "投信": "近20日買超金額最大",
            "主力券商": "追近60日分點動向",
            "創新高(super8888)": "營收股價持續創新高，維持高檔、低融資使用率、大戶買散戶賣",
            "super888": "低波動、強營收、強動能、小市值、低融資使用率、大戶買散戶賣",
            "super88": "短期營收創新高、低融資使用率、大戶買散戶賣",
            "創新高(super8)": "股價創新高、低融資使用率、大戶買散戶賣",
            "rev_growth": "近一年營收斜率最大",
            "五線上": "近20日連續站上5日均線PR80以上的強勢股取最小市值",
            "五線上V": "近20日連續站上5日均線PR80以上的強勢股且股價接近新高取最小市值",
            "千層蛋糕": "濾網式寫法、籌碼中的籌碼",
            "積分制K線圖": "加權平均計分制",
            "營收偷開": "月營收偷公布",
            "Q1_YTD": "第一季抓高股息ETF豆腐",
        }

        # 計算需要的頁數（每頁20個股票，每頁分成上下兩部分，每部分10個）
        total_stocks = len(self.result_df)
        stocks_per_page = 20
        num_pages = math.ceil(total_stocks / stocks_per_page)

        # 定義股票圖表頁面生成函數
        def create_stock_chart(page_num):
            start_idx = (page_num - 1) * stocks_per_page

            fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT_STANDARD),
                            facecolor='white')
            gs = GridSpec(2, 1, figure=fig,
                        height_ratios=[1, 1],
                        hspace=GRID_HSPACE)

            for i in range(2):
                ax = fig.add_subplot(gs[i])
                subset_start = start_idx + i * 10
                subset_end = min(start_idx + (i + 1) * 10, total_stocks)
                subset = self.result_df.iloc[subset_start:subset_end]

                if len(subset) == 0:  # 如果這部分沒有數據，跳過
                    continue

                bottoms = np.zeros(len(subset))

                for strategy in self.colors.keys():
                    heights = [strats.count(strategy)
                            for strats in subset['對應策略']]
                    bars = ax.bar(subset['股票名稱'] + '\n(' +
                                subset['股票代碼'].astype(str) + ')\n' +
                                subset['主題'],
                                heights, bottom=bottoms,
                                color=self.colors[strategy],
                                edgecolor='black', linewidth=1.5)
                    bottoms += heights

                    # 在柱狀圖中添加策略標籤
                    for bar, height in zip(bars, heights):
                        if height > 0:
                            ax.text(bar.get_x() + bar.get_width() / 2,
                                    bar.get_y() + height / 2,
                                    strategy,
                                    ha='center', va='center',
                                    fontsize=STRATEGY_TEXT_SIZE,
                                    color='black',
                                    fontweight='bold')

                start_num = subset_start + 1
                end_num = subset_end
                ax.set_title(f'長風破浪會有時，直掛雲帆濟滄海({start_num}-{end_num})',
                            fontsize=TITLE_SIZE, pad=TITLE_PAD)
                ax.set_ylabel('被\n選\n次\n數',
                            fontsize=AXIS_LABEL_SIZE,
                            rotation=0,
                            labelpad=LABEL_PAD)

                # 設置X軸標籤
                ax.set_xticks(range(len(subset)))
                ax.set_xticklabels(subset['股票名稱'] + '\n(' +
                                subset['股票代碼'].astype(str) + ')\n' +
                                subset['主題'],
                                rotation=45,
                                ha='right',
                                fontsize=TICK_LABEL_SIZE)

                # 調整Y軸範圍和刻度
                max_count = subset['被選次數'].max()
                ax.set_ylim(0, max_count * 1.3)
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.tick_params(axis='y', labelsize=TICK_LABEL_SIZE)

                # 添加計數標籤
                for j, count in enumerate(subset['被選次數']):
                    ax.text(j, count, str(count),
                            ha='center', va='bottom',
                            fontsize=COUNT_TEXT_SIZE,
                            fontweight='bold')

                ax.grid(True, linestyle='--', alpha=0.3)

            fig.suptitle('策略選股統計分析', y=0.98, fontsize=TITLE_SIZE+4, fontweight='bold')

            plt.tight_layout(pad=LAYOUT_PAD)
            return fig

        print("開始分段生成PDF報告...")
        
        # 第一部分：股票圖表頁面的PDF
        temp_file1 = os.path.join(temp_folder, f"temp_stock_charts_{today_date}.pdf")
        temp_pdf_files.append(temp_file1)
        print(f"正在生成股票圖表頁面... ({temp_file1})")

        
        with PdfPages(temp_file1) as pdf:
            for page in range(1, num_pages + 1):
                fig = create_stock_chart(page)
                pdf.savefig(fig)
                plt.close(fig)
        
        # 第二部分：熱門族群觀察圖和策略說明
        temp_file2 = os.path.join(temp_folder, f"temp_strategy_desc_{today_date}.pdf")
        temp_pdf_files.append(temp_file2)
        print(f"正在生成熱門族群觀察圖和策略說明... ({temp_file2})")
        
        with PdfPages(temp_file2) as pdf:
            fig3 = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT_LARGE),
                            facecolor='white')

            fig3.suptitle('策略說明與熱門族群分析', y=0.98, fontsize=TITLE_SIZE+4, fontweight='bold')

            gs3 = GridSpec(3, 2, figure=fig3,
                        height_ratios=[1.2, 1, 1],
                        hspace=GRID_HSPACE,
                        wspace=GRID_WSPACE)

            # 熱門族群觀察圖
            ax5 = fig3.add_subplot(gs3[0, :])
            topic_counts = self.result_df['主題'].value_counts()
            bar_colors = plt.cm.viridis(np.linspace(0, 1, len(topic_counts)))
            bars = topic_counts.plot(kind='bar', color=bar_colors, ax=ax5)

            ax5.set_title('熱門族群觀察',
                        fontsize=TITLE_SIZE,
                        pad=TITLE_PAD)
            ax5.set_ylabel('出\n現\n次\n數',
                        fontsize=AXIS_LABEL_SIZE,
                        rotation=0,
                        labelpad=LABEL_PAD)

            new_labels = [f"{topic}\n({', '.join(map(str, self.result_df[self.result_df['主題'] == topic]['股票代碼'].tolist()))})"
                        for topic in topic_counts.index]
            ax5.set_xticklabels(new_labels,
                                rotation=45,
                                ha='right',
                                fontsize=TICK_LABEL_SIZE)

            # 添加計數標籤
            for i, v in enumerate(topic_counts):
                ax5.text(i, v, str(v),
                        ha='center', va='bottom',
                        fontsize=COUNT_TEXT_SIZE,
                        fontweight='bold')

            ax5.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
            ax5.grid(True, linestyle='--', alpha=0.3)

            # 策略說明部分
            ax6 = fig3.add_subplot(gs3[1, 0])
            ax7 = fig3.add_subplot(gs3[1, 1])
            ax6.axis('off')
            ax7.axis('off')

            STRATEGY_DESC_SIZE = 32
            strategy_text1 = (
                "1. GVI:\n   ROE與BP_ratio關係分析、大戶買散戶賣\n\n"
                "2. 小資:\n   低波動、強營收、強動能、小市值\n\n"
                "3. 小資YOY:\n   低波動、強營收、強動能、最強YOY\n\n"
                "4. 投信買什麼:\n   近20日買超金額最大\n\n"
                "5. 券商分點:\n   追近60日分點動向\n\n"
                "6. super8888:\n   營收股價持續創新高，維持高檔\n   低融資使用率、大戶買散戶賣"
            )

            strategy_text2 = (
                "7. super888:\n   低波動、強營收、強動能、小市值\n   低融資使用率、大戶買散戶賣\n\n"
                "8. super88:\n   短期營收創新高、低融資使用率\n   大戶買散戶賣\n\n"
                "9. rev_growth:\n   近一年營收斜率最大\n\n"
                "10. 五線穿雲術:\n    近20日連續站上5日均線PR80\n    以上的強勢股\n\n"
                "11. 千層蛋糕:\n    濾網式寫法\n    籌碼中的籌碼\n\n"
            )

            # 調整文字框的參數
            text_box_props = dict(
                facecolor='white',
                alpha=0.8,
                edgecolor='gray',
                boxstyle='round,pad=1.5',
            )
            # 調整文字位置和格式
            for ax, text in [(ax6, strategy_text1), (ax7, strategy_text2)]:
                ax.text(0.02, 0.98, text,
                        ha="left", va="top",
                        fontsize=STRATEGY_DESC_SIZE,
                        fontweight='normal',
                        bbox=text_box_props,
                        transform=ax.transAxes,
                        linespacing=1.4)

            plt.tight_layout(pad=LAYOUT_PAD)
            pdf.savefig(fig3)
            plt.close(fig3)
        
        # 第三部分：產業強弱變化趨勢圖
        temp_file3 = os.path.join(temp_folder, f"temp_industry_trend_{today_date}.pdf")
        temp_pdf_files.append(temp_file3)
        print(f"正在生成產業強弱變化趨勢圖... ({temp_file3})")
        
        with PdfPages(temp_file3) as pdf:
            fig_trend = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT_STANDARD), facecolor='white')
            fig_trend.suptitle('產業強弱變化趨勢分析(近5/20/60日報酬變化)', y=0.99,
                            fontsize=TITLE_SIZE+4, fontweight='bold')
            self.plot_industry_trend(fig_trend)

            plt.tight_layout(pad=LAYOUT_PAD)
            pdf.savefig(fig_trend)
            plt.close(fig_trend)
        
            # 第四部分：5日產業強弱分析
        temp_file4 = os.path.join(temp_folder, f"temp_industry_5d_{today_date}.pdf")
        temp_pdf_files.append(temp_file4)
        print(f"正在生成5日產業強弱分析... ({temp_file4})")
        
        with PdfPages(temp_file4) as pdf:
            fig_industry = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT_STANDARD * 1.5), facecolor='white')
            fig_industry.suptitle('5日產業強弱分析', y=0.98,
                                fontsize=TITLE_SIZE+4, fontweight='bold')
            self.plot_industry_performance(fig_industry, period_days=5)  # 繪製近五日的

            plt.tight_layout(pad=LAYOUT_PAD)
            pdf.savefig(fig_industry)
            plt.close(fig_industry)
        
        # 第五部分：20日產業強弱分析
        temp_file5 = os.path.join(temp_folder, f"temp_industry_20d_{today_date}.pdf")
        temp_pdf_files.append(temp_file5)
        print(f"正在生成20日產業強弱分析... ({temp_file5})")
        
        with PdfPages(temp_file5) as pdf:
            fig_industry = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT_STANDARD * 1.5), facecolor='white')
            fig_industry.suptitle('20日產業強弱分析', y=0.98,
                                fontsize=TITLE_SIZE+4, fontweight='bold')
            self.plot_industry_performance(fig_industry, period_days=20)  # 繪製近20日的

            plt.tight_layout(pad=LAYOUT_PAD)
            pdf.savefig(fig_industry)
            plt.close(fig_industry)
        
        # 第六部分：60日產業強弱分析
        temp_file6 = os.path.join(temp_folder, f"temp_industry_60d_{today_date}.pdf")
        temp_pdf_files.append(temp_file6)
        print(f"正在生成60日產業強弱分析... ({temp_file6})")
        
        with PdfPages(temp_file6) as pdf:
            fig_industry = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT_STANDARD * 1.5), facecolor='white')
            fig_industry.suptitle('60日產業強弱分析', y=0.98,
                                fontsize=TITLE_SIZE+4, fontweight='bold')
            self.plot_industry_performance(fig_industry, period_days=60)  # 繪製近60日的

            plt.tight_layout(pad=LAYOUT_PAD)
            pdf.savefig(fig_industry)
            plt.close(fig_industry)
        
        # 第七部分：市場強度圖
        temp_file7 = os.path.join(temp_folder, f"temp_market_analysis_{today_date}.pdf")
        temp_pdf_files.append(temp_file7)
        print(f"正在生成市場強度圖... ({temp_file7})")
        
        with PdfPages(temp_file7) as pdf:
            plot_market_analysis = plt.figure(
                figsize=(PAGE_WIDTH, PAGE_HEIGHT_STANDARD * 1.5),  # 加高以容納兩個子圖
                facecolor='white')
            plot_market_analysis.suptitle('市場強度與相對價值分析', y=0.98, fontsize=TITLE_SIZE+4, fontweight='bold')

            # 使用 GridSpec 創建兩個子圖
            gs4 = GridSpec(2, 1, figure=plot_market_analysis,
                        height_ratios=[1, 1], hspace=0.3)
            ax8 = plot_market_analysis.add_subplot(gs4[0])    # 上方子圖
            ax9 = plot_market_analysis.add_subplot(gs4[1])    # 下方子圖

            # 繪製市場強度圖
            self.plot_market_analysis(ax8, ax9)

            # 設置上圖（市場強度）的標題和標籤
            ax8.set_title('市場強度',
                        fontsize=TITLE_SIZE,
                        pad=TITLE_PAD)
            ax8.set_ylabel('強\n度',
                        fontsize=AXIS_LABEL_SIZE,
                        rotation=0,
                        labelpad=LABEL_PAD)

            # 設置下圖（相對價值）的標題和標籤
            ax9.set_title('小型股對2330的超額報酬',
                        fontsize=TITLE_SIZE,
                        pad=TITLE_PAD)
            ax9.set_ylabel('價\n值',
                        fontsize=AXIS_LABEL_SIZE,
                        rotation=0,
                        labelpad=LABEL_PAD)

            # 設置兩個子圖的刻度標籤大小和網格
            for ax in [ax8, ax9]:
                ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
                ax.grid(True, linestyle='--', alpha=0.3)

            plt.tight_layout(pad=LAYOUT_PAD)
            pdf.savefig(plot_market_analysis)
            plt.close(plot_market_analysis)
        
        # 計算相關性和排序股票
        print("計算股票相關性並排序...")
        try:
            stock_correlations, pattern_correlations = self.calculate_correlation()

            # 如果有相關性數據，則根據相關性排序
            if pattern_correlations is not None and not pattern_correlations.empty:
                sorted_stocks = sorted(
                    self.high_score_stocks,
                    key=lambda s: (
                        pattern_correlations.get(s, 0) * 2 +
                        sum(corr for stock1, stock2, corr in stock_correlations
                            if s in (stock1, stock2)) * 0.5
                    ),
                    reverse=True
                )
            else:
                # 如果沒有相關性數據，直接使用 high_score_stocks
                print("注意: 無相關性數據，直接使用篩選後的股票清單")
                sorted_stocks = list(self.high_score_stocks)

        except Exception as e:
            print(f"計算相關性時發生錯誤: {e}")
            print("使用原始篩選後的股票清單")
            sorted_stocks = list(self.high_score_stocks)
        
        # 第八部分及以後：個股分析頁面，每10個股票一個PDF
        batch_size = 5
        total_batches = math.ceil(len(sorted_stocks) / batch_size)
    
        for batch_idx in range(0, len(sorted_stocks), batch_size):
            batch_end = min(batch_idx + batch_size, len(sorted_stocks))
            batch_stocks = sorted_stocks[batch_idx:batch_end]
            current_batch = batch_idx // batch_size + 1
            
            temp_file_stocks = os.path.join(temp_folder, f"temp_stocks_batch{current_batch}_{today_date}.pdf")
            temp_pdf_files.append(temp_file_stocks)
            
            print(f"正在生成個股分析 (批次 {current_batch}/{total_batches})... ({temp_file_stocks})")
            print(f"處理股票: {', '.join(batch_stocks)}")
            
            with PdfPages(temp_file_stocks) as pdf:
                for stock_id in batch_stocks:
                    fig = plt.figure(figsize=(48, 180), facecolor='white')
                    fig.suptitle(f'個股分析 - {stock_id}', y=0.99, fontsize=TITLE_SIZE+4, fontweight='bold')

                    # 獲取該股票對應的策略列表
                    strategies = self.result_df[self.result_df['股票代碼'] == int(stock_id)]['對應策略'].iloc[0]
                        
                    # 修改策略說明文字的顯示方式
                    # 計算文字框的位置
                    text_x = 0.5  # 水平置中（0.5 表示圖表寬度的中間位置）
                    text_y = 0.96  # 垂直位置保持不變

                    # 創建文字框顯示策略資訊
                    strategy_text_list = []
                    for strategy in strategies:
                        description = strategy_descriptions.get(strategy, "無說明")
                        strategy_text_list.append(f"{strategy}：{description}")

                    strategy_text = "\n".join(strategy_text_list)

                    # 設置文字框的樣式
                    text_box_props = dict(
                        facecolor='wheat',
                        alpha=0.5,
                        boxstyle='round,pad=1.0',  # 可以調整 pad 值來改變文字框的內邊距
                        edgecolor='gray'
                    )

                    # 添加文字，設置對齊方式
                    fig.text(text_x, text_y, strategy_text,
                            fontsize=30,
                            fontweight='bold',
                            ha='center',  # 整個文字框水平置中
                            va='center',
                            bbox=text_box_props,
                            multialignment='left')  # 文字框內的文字左對齊

                    gs = GridSpec(14, 3, figure=fig,  
                    hspace=1,
                    wspace=0.3)

                    # 第一行：由左至右排列前三個圖
                    ax_k = fig.add_subplot(gs[0, 0])
                    self.plot_stock(ax_k, stock_id)

                    ax_k2 = fig.add_subplot(gs[0, 1])
                    self.plot_stock2(ax_k2, stock_id)

                    ax_rci = fig.add_subplot(gs[0, 2])
                    self.plot_stock_with_rci(ax_rci, stock_id)

                    # 第二行：由左至右排列中間三個圖
                    ax_rev = fig.add_subplot(gs[1, 0])
                    self.plot_revenue(ax_rev, stock_id)

                    ax_rev2 = fig.add_subplot(gs[1, 1])
                    self.plot_revenue2(ax_rev2, stock_id)

                    ax_hold = fig.add_subplot(gs[1, 2])
                    self.plot_holding_ratio(ax_hold, stock_id)

                    # 第三行：由左至右排列三個圖
                    ax_broker = fig.add_subplot(gs[2, 0])
                    self.plot_broker_analysis(ax_broker, stock_id)

                    ax_inst = fig.add_subplot(gs[2, 1])
                    self.plot_institutional_investors(ax_inst, stock_id)

                    ax_signal = fig.add_subplot(gs[2, 2])
                    self.plot_margin_trading_focus(ax_signal, stock_id)
                    
                      # 繪製第一張圖：股價和買賣超對比
                    ax_broker2 = fig.add_subplot(gs[3, :])
                    self.plot_broker_analysis2(ax_broker2, stock_id)

                    # 繪製第二張圖：Z分數趨勢分析
                    ax_broker3 = fig.add_subplot(gs[4, :])
                    self.plot_broker_analysis3(ax_broker3, stock_id)
                    
                    ax_broker4 = fig.add_subplot(gs[5, :])
                    self.plot_broker_analysis4(ax_broker4, stock_id)
                    
                       
                    ax_plot_holding_distribution = fig.add_subplot(gs[6, :])
                    self.plot_holding_ratio2(ax_plot_holding_distribution, stock_id)
                    
                    ax_plot_major_holder_distribution= fig.add_subplot(gs[7, :])
                    self.plot_major_holder_distribution(ax_plot_major_holder_distribution,stock_id)
                    
                    ax_plot_average_holdings_analysis= fig.add_subplot(gs[8, :])
                    self.plot_average_holdings_analysis(ax_plot_average_holdings_analysis,stock_id)
                    
                    # 董監持股分析
                    ax_plot_internal_holding_changes=fig.add_subplot(gs[9, :])
                    self.plot_internal_holding_changes(ax_plot_internal_holding_changes, stock_id)
                  
                    institutional_analysis = fig.add_subplot(gs[10, :])
                    self.plot_institutional_analysis(institutional_analysis, stock_id)
                    
                    ax_trend_analysis = fig.add_subplot(gs[11, :])
                    self.plot_trend_analysis(ax_trend_analysis, stock_id)
                    
                    # 財務分析圖表橫跨整行
                    ax_financial = fig.add_subplot(gs[12, :])
                    self.plot_financial_analysis(ax_financial, stock_id)
                    
                    ax_plot_rd_vs_pm_analysis= fig.add_subplot(gs[13, :])
                    self.plot_rd_vs_pm_analysis(ax_plot_rd_vs_pm_analysis, stock_id)

                    plt.tight_layout(pad=LAYOUT_PAD)
                    pdf.savefig(fig)
                    plt.close(fig)
        
        
        # 合併所有臨時PDF檔案
        print(f"合併所有臨時PDF檔案為最終報告: {final_file_name}")
        merger = PdfMerger()
        for pdf_file in temp_pdf_files:
            try:
                merger.append(pdf_file)
            except Exception as e:
                print(f"合併PDF時出錯 ({pdf_file}): {e}")
        
        merger.write(final_file_name)
        merger.close()
        
        # 清理臨時檔案
        print("清理臨時檔案...")
        for temp_file in temp_pdf_files:
            try:
                os.remove(temp_file)
                print(f"已刪除: {temp_file}")
            except Exception as e:
                print(f"無法刪除臨時檔案 ({temp_file}): {e}")
        
        # 清理臨時資料夾
        try:
            os.rmdir(temp_folder)
            print(f"已刪除臨時資料夾: {temp_folder}")
        except Exception as e:
            print(f"無法刪除臨時資料夾: {e}")
        
        print(f"PDF報告已成功生成: {final_file_name}")






if __name__ == "__main__":
    total_start_time = time.time()  # 記錄總開始時間
    
    try:
        # 創建StockPlotter實例
        print(f"開始執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        stock_plotter = StockPlotter('select_stock.xlsx', 'tw_stock_topics.xlsx')
        
        # 處理數據
        start_time = time.time()
        print("正在處理數據...")
        stock_plotter.process_data()
        process_time = time.time() - start_time
        print(f"數據處理完成，耗時: {process_time:.2f} 秒")
        
        # 設置顏色
        start_time = time.time()
        print("正在設置圖表顏色...")
        stock_plotter.setup_colors()
        color_time = time.time() - start_time
        print(f"顏色設置完成，耗時: {color_time:.2f} 秒")
        
        # 計算相關性
        start_time = time.time()
        print("計算股票之間的相關性...")
        all_correlation_pairs, pattern_correlations = stock_plotter.calculate_correlation()
        correlation_time = time.time() - start_time
        print(f"相關性計算完成，耗時: {correlation_time:.2f} 秒")
        
        # 創建PDF報告
        start_time = time.time()
        print("正在生成PDF報告...")
        today_date = datetime.today().strftime('%Y%m%d')
        final_pdf_name = f"tomstrategy_{today_date}.pdf"
        stock_plotter.create_pdf()
        pdf_time = time.time() - start_time
        print(f"PDF報告生成完成，耗時: {pdf_time:.2f} 秒")
        
        # 計算總執行時間
        total_time = time.time() - total_start_time
        minutes, seconds = divmod(total_time, 60)
        hours, minutes = divmod(minutes, 60)
        
        print("\n執行統計:")
        print(f"總執行時間: {int(hours)}小時 {int(minutes)}分 {seconds:.2f}秒")
        print(f"- 數據處理時間: {process_time:.2f} 秒 ({process_time/total_time*100:.1f}%)")
        print(f"- 顏色設置時間: {color_time:.2f} 秒 ({color_time/total_time*100:.1f}%)")
        print(f"- 相關性計算時間: {correlation_time:.2f} 秒 ({correlation_time/total_time*100:.1f}%)")
        print(f"- PDF生成時間: {pdf_time:.2f} 秒 ({pdf_time/total_time*100:.1f}%)")
        print(f"結束執行時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"報告已保存為 {final_pdf_name}")

        # 檢查記憶體使用情況
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"記憶體使用: {memory_info.rss / 1024 / 1024:.1f} MB")

    except Exception as e:
        import traceback
        print(f"程式執行出錯：{str(e)}")
        print("詳細錯誤信息:")
        traceback.print_exc()
        
        # 即使發生錯誤也顯示已執行時間
        total_time = time.time() - total_start_time
        minutes, seconds = divmod(total_time, 60)
        hours, minutes = divmod(minutes, 60)
        print(f"執行中斷，已執行時間: {int(hours)}小時 {int(minutes)}分 {seconds:.2f}秒")



'''

請讀取 PDF 檔案中的所有數據與圖表，並依照以下步驟進行 完整的市場、產業與個股分析，最終提供投資建議。

📊 一、從 PDF 擷取市場趨勢數據，分析近期市場狀況
📌 請先掃描 PDF，擷取與「市場整體趨勢」相關的數據，並回答以下問題：
1️⃣ 近期市場報酬率如何？

過去 1 週、1 個月、3 個月的市場報酬率是多少？
是否有市場資金大幅流入或流出？
波動率（Z 值）是否顯示市場進入震盪？
2️⃣ 法人（外資、投信、自營商）對市場的態度？

近期法人對整體市場的買超 / 賣超變化？

📈 二、從 PDF 擷取產業報酬率數據，分析近期市場主流產業
📌 請從 PDF 中找出「各產業報酬率」，並分析目前市場資金的流向：
1️⃣ 哪些產業報酬率最高？

產業報酬率排名前 3 名 是哪些？（請提供數據）
過去 1 週、1 個月、3 個月，哪些產業成長最快？
這些產業的成長是否有基本面支撐？
2️⃣ 哪些產業資金正在流入？

法人（外資、投信）近期是否集中買超某些產業？
3️⃣ 市場熱度與題材分析


📉 三、從 PDF 擷取個股數據，進行完整技術與籌碼分析
📌 請根據 PDF 中的 15 張圖表，針對 [股票代號] 進行完整分析，包含技術面、籌碼面、營運狀況與風險評估。

📌 技術分析（圖表 1~3）

K 線走勢 + 均線（SMA5、SMA20、SMA60）
股價目前是否為多頭排列？（SMA5 > SMA20 > SMA60）
是否突破壓力位？支撐位在哪？
近期 K 線型態（如長紅、長上影線、跳空缺口）？
成交量變化
最近成交量變化？
是否符合「量價齊揚」？還是出現「價漲量縮」的警訊？
RCI 指標
是否過熱？（RCI 20 超過 80）
是否出現短線回檔訊號？
📌 籌碼分析（圖表 4~7）

千張大戶持股比例
近期大戶持股比例變化？
是否超過 40%？
三大法人買賣超
近期法人買超 / 賣超趨勢？
買超是否與股價同步上漲？
券商分點買賣超
是否有特定券商連續買超但股價未明顯上漲？
是否出現大戶分點倒貨跡象？
融資餘額變化
近期融資餘額是否大幅增加？是否來到新高？
📌 營運分析（圖表 8~10）

營收年增率（YOY、QOQ）
最近一年營收成長率？
近 3 個月、12 個月的營收趨勢？
是否顯示營運轉強或轉弱？
股價相對 PR 值
目前 PR 值在市場中的相對排名？
📌 風險評估（圖表 11~15）
股價波動率是否過高？
是否有 K 線異常型態（長上影線、跳空缺口）？
量能異常放大是否與股價脫鉤？
Z 值波動率是否顯示短線風險？
市場熱度是否過高，產生短線風險？
📌 請將這些數據整理為表格或圖表，並提供具體分析結論！

📌 四、綜合結論與投資建議
📌 請綜合市場趨勢、產業分析與個股數據，提供完整的投資建議，包括：
✅ 市場趨勢總結：目前市場是否適合進場？該股所在產業是否為主流？
✅ 技術面評估：該股是否具備突破行情？是否有回檔風險？
✅ 籌碼面分析：法人與主力分點是否同步進場？籌碼是否集中？
✅ 基本面支撐：營收指標是否顯示成長趨勢？
✅ 風險評估：股價是否過熱？近期是否有短線回檔風險？

📌 請提供明確的投資建議，包括：
✅ 建議進場價：______
✅ 停損價位：______
✅ 預期目標價：______

📌 若該股存在過熱或轉弱風險，請清楚標示，並說明應如何應對！

'''
 
 
# python Select_stock_finally.py




  