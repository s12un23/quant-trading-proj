"""
alphas.py - 101 Formulaic Alphas 完整实现
=============================================
Alpha101和此前跑的不错的若干自定义因子

需要行业分类(IndClass)或市值(cap)数据的因子已跳过并标注。
最终可用因子约80个 + 7个自定义学术因子 = 87个。

使用方法:
    from alpha101 import compute_all_alphas
    stocks, alpha_cols = compute_all_alphas(stocks)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def ts_op(series, d, func, ticker, min_pct=0.5):
    mp = max(1, int(d * min_pct))
    return series.groupby(ticker).transform(lambda x: getattr(x.rolling(d, min_periods=mp), func)())

def ts_mean(s, d, t):  return ts_op(s, d, 'mean', t)
def ts_std(s, d, t):   return ts_op(s, d, 'std', t)
def ts_max(s, d, t):   return ts_op(s, d, 'max', t)
def ts_min(s, d, t):   return ts_op(s, d, 'min', t)
def ts_sum(s, d, t):   return ts_op(s, d, 'sum', t)

def ts_rank(s, d, t):
    mp = max(1, d // 2)
    return s.groupby(t).transform(lambda x: x.rolling(d, min_periods=mp).apply(lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=False))

def ts_argmax(s, d, t):
    mp = max(1, d // 2)
    return s.groupby(t).transform(lambda x: x.rolling(d, min_periods=mp).apply(lambda w: d - 1 - np.argmax(w), raw=True))

def ts_argmin(s, d, t):
    mp = max(1, d // 2)
    return s.groupby(t).transform(lambda x: x.rolling(d, min_periods=mp).apply(lambda w: d - 1 - np.argmin(w), raw=True))

def ts_corr(x, y, d, t):
    mp = max(1, d // 2)
    return x.groupby(t).transform(lambda a: a.rolling(d, min_periods=mp).corr(y.loc[a.index]))

def ts_cov(x, y, d, t):
    mp = max(1, d // 2)
    return x.groupby(t).transform(lambda a: a.rolling(d, min_periods=mp).cov(y.loc[a.index]))

def delta(s, d, t):    return s.groupby(t).transform(lambda x: x.diff(d))
def delay(s, d, t):    return s.groupby(t).transform(lambda x: x.shift(d))
def cs_rank(s, dt):    return s.groupby(dt).rank(pct=True)
def cs_scale(s, dt):   return s / s.groupby(dt).transform(lambda x: x.abs().sum() + 1e-8)

def decay_linear(s, d, t):
    weights = np.arange(1, d + 1, dtype=float)
    weights = weights / weights.sum()
    mp = max(1, d // 2)
    return s.groupby(t).transform(lambda x: x.rolling(d, min_periods=mp).apply(lambda w: np.dot(w[-len(weights):], weights[-len(w):]) if len(w) >= mp else np.nan, raw=True))

def signed_power(s, a):
    return s.abs().pow(a.clip(-2, 2)) * np.sign(s)


def compute_all_alphas(df):
    T = df['Ticker']
    D = df['Date']
    O = df['Open']; H = df['High']; L = df['Low']; C = df['Close']
    V = df['Volume']; R = df['Return']; VWAP = df['VWAP']; ADV20 = df['adv20']
    
    alpha_cols = []
    computed = 0; skipped = 0
    
    def safe_add(name, series):
        nonlocal computed
        df[name] = series.replace([np.inf, -np.inf], np.nan)
        alpha_cols.append(name)
        computed += 1
    
    def skip(num, reason):
        nonlocal skipped; skipped += 1
    
    print("开始计算101 Alphas + 自定义因子...")
    dc = delta(C, 1, T)
    
    # Alpha#001
    inner = pd.Series(np.where(R < 0, ts_std(R, 20, T), C), index=df.index)
    safe_add('alpha_001', cs_rank(ts_argmax(inner, 5, T), D) - 0.5)
    # Alpha#002
    safe_add('alpha_002', -1 * ts_corr(cs_rank(delta(np.log(V+1), 2, T), D), cs_rank((C-O)/(O+1e-8), D), 6, T))
    # Alpha#003
    safe_add('alpha_003', -1 * ts_corr(cs_rank(O, D), cs_rank(V, D), 10, T))
    # Alpha#004
    safe_add('alpha_004', -1 * ts_rank(cs_rank(L, D), 9, T))
    # Alpha#005
    safe_add('alpha_005', cs_rank(O - ts_mean(VWAP, 10, T), D) * (-1 * cs_rank(C - VWAP, D).abs()))
    # Alpha#006
    safe_add('alpha_006', -1 * ts_corr(O, V, 10, T))
    # Alpha#007
    cond7 = ADV20 < V
    safe_add('alpha_007', pd.Series(np.where(cond7, -1*ts_rank(dc.abs(), 60, T)*np.sign(delta(C,7,T)), -1), index=df.index))
    # Alpha#008
    tmp8 = ts_sum(O, 5, T) * ts_sum(R, 5, T)
    safe_add('alpha_008', -1 * cs_rank(tmp8 - delay(tmp8, 10, T), D))
    # Alpha#009
    c1 = ts_min(dc, 5, T) > 0; c2 = ts_max(dc, 5, T) < 0
    safe_add('alpha_009', pd.Series(np.where(c1, dc, np.where(c2, dc, -1*dc)), index=df.index))
    # Alpha#010
    c1b = ts_min(dc, 4, T) > 0; c2b = ts_max(dc, 4, T) < 0
    safe_add('alpha_010', cs_rank(pd.Series(np.where(c1b, dc, np.where(c2b, dc, -1*dc)), index=df.index), D))
    # Alpha#011
    vc = VWAP - C
    safe_add('alpha_011', (cs_rank(ts_max(vc, 3, T), D) + cs_rank(ts_min(vc, 3, T), D)) * cs_rank(delta(V, 3, T), D))
    # Alpha#012
    safe_add('alpha_012', np.sign(delta(V, 1, T)) * (-1 * dc))
    # Alpha#013
    safe_add('alpha_013', -1 * cs_rank(ts_cov(cs_rank(C, D), cs_rank(V, D), 5, T), D))
    # Alpha#014
    safe_add('alpha_014', -1 * cs_rank(delta(R, 3, T), D) * ts_corr(O, V, 10, T))
    # Alpha#015
    safe_add('alpha_015', -1 * ts_sum(cs_rank(ts_corr(cs_rank(H, D), cs_rank(V, D), 3, T), D), 3, T))
    # Alpha#016
    safe_add('alpha_016', -1 * cs_rank(ts_cov(cs_rank(H, D), cs_rank(V, D), 5, T), D))
    # Alpha#017
    safe_add('alpha_017', -1*cs_rank(ts_rank(C,10,T),D)*cs_rank(delta(dc,1,T),D)*cs_rank(ts_rank(V/(ADV20+1),5,T),D))
    # Alpha#018
    safe_add('alpha_018', -1 * cs_rank(ts_std((C-O).abs(), 5, T) + (C-O) + ts_corr(C, O, 10, T), D))
    # Alpha#019
    safe_add('alpha_019', -1*np.sign(C - delay(C,7,T) + delta(C,7,T)) * (1 + cs_rank(1 + ts_sum(R, 250, T), D)))
    # Alpha#020
    safe_add('alpha_020', -1*cs_rank(O-delay(H,1,T),D)*cs_rank(O-delay(C,1,T),D)*cs_rank(O-delay(L,1,T),D))
    # Alpha#021
    cond21a = ts_mean(C,8,T) + ts_std(C,8,T) < ts_mean(C,2,T)
    cond21b = ts_mean(V,20,T)/V < 1
    safe_add('alpha_021', pd.Series(np.where(cond21a, -1, np.where(cond21b, 1, -1)), index=df.index))
    # Alpha#022
    safe_add('alpha_022', -1*delta(ts_corr(H, V, 5, T), 5, T) * cs_rank(ts_std(C, 20, T), D))
    # Alpha#023
    safe_add('alpha_023', pd.Series(np.where(ts_mean(H,20,T)<H, -1*delta(H,2,T), 0), index=df.index))
    # Alpha#024
    cond24 = delta(ts_mean(C,100,T),100,T) / (delay(C,100,T)+1e-8) <= 0.05
    safe_add('alpha_024', pd.Series(np.where(cond24, -1*(C-ts_min(C,100,T)), -1*delta(C,3,T)), index=df.index))
    # Alpha#025
    safe_add('alpha_025', cs_rank(-1*R*ADV20*VWAP*(H-C), D))
    # Alpha#026
    safe_add('alpha_026', -1*ts_max(ts_corr(ts_rank(V,5,T), ts_rank(H,5,T), 5, T), 3, T))
    # Alpha#027
    tmp27 = ts_sum(ts_corr(cs_rank(V,D), cs_rank(VWAP,D), 6, T), 2, T) / 2
    safe_add('alpha_027', pd.Series(np.where(cs_rank(tmp27, D) > 0.5, -1, 1), index=df.index))
    # Alpha#028
    safe_add('alpha_028', cs_scale(ts_corr(ADV20, L, 5, T) + (H+L)/2 - C, D))
    # Alpha#029
    safe_add('alpha_029', ts_mean(cs_rank(-1*R, D)*cs_rank(V/(ADV20+1), D), 5, T))
    # Alpha#030
    s30 = np.sign(C-delay(C,1,T)) + np.sign(delay(C,1,T)-delay(C,2,T)) + np.sign(delay(C,2,T)-delay(C,3,T))
    safe_add('alpha_030', (1-cs_rank(s30,D))*ts_sum(V,5,T)/(ts_sum(V,20,T)+1))
    # Alpha#031
    safe_add('alpha_031', cs_rank(decay_linear(-1*cs_rank(delta(C,10,T),D),10,T),D) + cs_rank(-1*delta(C,3,T),D))
    # Alpha#032
    safe_add('alpha_032', cs_scale(ts_mean(C,7,T)-C, D) + 20*cs_scale(ts_corr(VWAP, delay(C,5,T), 60, T), D))
    # Alpha#033
    safe_add('alpha_033', cs_rank(-1*(1 - O/(C+1e-8)), D))
    # Alpha#034
    safe_add('alpha_034', cs_rank(1-cs_rank(ts_std(R,2,T)/(ts_std(R,5,T)+1e-8),D) + 1-cs_rank(dc,D), D))
    # Alpha#035
    safe_add('alpha_035', ts_rank(V,32,T)*(1-ts_rank(C+H-L,16,T))*(1-ts_rank(R,32,T)))
    # Alpha#036
    safe_add('alpha_036', 2.21*cs_rank(ts_corr(C-O, delay(V,1,T), 15, T), D) + 0.7*cs_rank(O-C,D) + 0.73*cs_rank(ts_rank(delay(-1*R,6,T),5,T),D) + cs_rank(ts_corr(VWAP,ADV20,6,T).abs(),D) + 0.6*cs_rank(ts_mean(C,200,T)-O,D))
    # Alpha#037
    safe_add('alpha_037', cs_rank(ts_corr(delay(O-C,1,T), C, 200, T), D) + cs_rank(O-C, D))
    # Alpha#038
    safe_add('alpha_038', -1*cs_rank(ts_rank(C,10,T),D)*cs_rank(C/(O+1e-8),D))
    # Alpha#039
    safe_add('alpha_039', -1*cs_rank(delta(C,7,T)*(1-cs_rank(decay_linear(V/(ADV20+1),9,T),D)),D)*(1+cs_rank(ts_sum(R,250,T),D)))
    # Alpha#040
    safe_add('alpha_040', -1*cs_rank(ts_std(H,10,T),D)*ts_corr(H,V,10,T))
    # Alpha#041
    safe_add('alpha_041', np.sqrt(H*L+1e-8) - VWAP)
    # Alpha#042
    safe_add('alpha_042', cs_rank(VWAP-C,D)/(cs_rank(VWAP+C,D)+1e-8))
    # Alpha#043
    safe_add('alpha_043', ts_rank(V/(ADV20+1),20,T)*ts_rank(-1*delta(C,7,T),8,T))
    # Alpha#044
    safe_add('alpha_044', -1*ts_corr(H, cs_rank(V,D), 5, T))
    # Alpha#045
    safe_add('alpha_045', -1*cs_rank(ts_mean(delay(C,5,T),20,T),D)*ts_corr(C,V,2,T)*cs_rank(ts_corr(ts_sum(C,5,T),ts_sum(C,20,T),2,T),D))
    # Alpha#046
    accel = delay(C,20,T)-2*delay(C,10,T)+C
    safe_add('alpha_046', -1*cs_rank(accel/(C+1e-8),D))
    # Alpha#047-048: IndClass
    skip(47,"IndClass"); skip(48,"IndClass")
    # Alpha#049
    cond49 = (delta(delay(C,1,T),1,T)+delta(delay(C,1,T),2,T)) < 0
    safe_add('alpha_049', pd.Series(np.where(cond49, 1, np.where((delta(delay(C,1,T),1,T)+delta(delay(C,1,T),2,T))>0, -1, -1*dc)), index=df.index))
    # Alpha#050
    safe_add('alpha_050', -1*ts_max(cs_rank(ts_corr(cs_rank(V,D),cs_rank(VWAP,D),5,T),D),5,T))
    # Alpha#051
    cond51 = delay(C,1,T)-delay(VWAP,1,T) > 0
    safe_add('alpha_051', pd.Series(np.where(cond51,-1,1),index=df.index) * (delay(C,1,T)-delay(VWAP,1,T)).abs())
    # Alpha#052
    safe_add('alpha_052', -1*delta(ts_min(L,5,T),5,T)*cs_rank(ts_sum(R,60,T)-ts_sum(R,10,T),D))
    # Alpha#053
    inner53 = (C-L-(H-C))/(H-L+1e-8)
    safe_add('alpha_053', -1*delta(inner53, 9, T))
    # Alpha#054
    safe_add('alpha_054', -1*(L-C)*(O**2)/((L-H+1e-8)*(C**2+1e-8)))
    # Alpha#055
    hl_range = ts_max(H,12,T)-ts_min(L,12,T)
    safe_add('alpha_055', -1*ts_corr(cs_rank((C-ts_min(L,12,T))/(hl_range+1e-8),D), cs_rank(V,D), 6, T))
    # Alpha#056: cap
    skip(56,"cap")
    # Alpha#057
    safe_add('alpha_057', -1*(C-VWAP)/(decay_linear(cs_rank(ts_argmax(C,30,T),D),2,T)+1e-8))
    # Alpha#058-059: IndClass
    skip(58,"IndClass"); skip(59,"IndClass")
    # Alpha#060
    inner60 = ((C-L-(H-C))/(H-L+1e-8))*V
    safe_add('alpha_060', -1*(2*cs_scale(cs_rank(inner60,D),D)-cs_scale(cs_rank(ts_argmax(C,10,T),D),D)))
    # Alpha#061
    safe_add('alpha_061', pd.Series(np.where(cs_rank(VWAP-ts_min(VWAP,16,T),D)<cs_rank(ts_corr(VWAP,ADV20,18,T),D),1,0),index=df.index))
    # Alpha#062
    safe_add('alpha_062', pd.Series(np.where(cs_rank(ts_corr(VWAP,ts_sum(ADV20,22,T),10,T),D)<cs_rank(pd.Series(np.where(2*cs_rank(O,D)<cs_rank((H+L)/2,D)+cs_rank(H,D),1,0),index=df.index),D),-1,0),index=df.index))
    # Alpha#063: IndClass
    skip(63,"IndClass")
    # Alpha#064
    safe_add('alpha_064', cs_rank(ts_corr(ts_sum(O,13,T),ts_sum(ADV20,13,T),25,T),D)*cs_rank(delta(dc,1,T),D))
    # Alpha#065
    mix65 = O*0.008+VWAP*0.992
    safe_add('alpha_065', pd.Series(np.where(cs_rank(ts_corr(mix65,ts_sum(ADV20,9,T),6,T),D)<cs_rank(O-ts_min(O,14,T),D),-1,0),index=df.index))
    # Alpha#066
    safe_add('alpha_066', -1*(cs_rank(decay_linear(delta(VWAP,4,T),7,T),D)+ts_rank(decay_linear((L-VWAP)/(O-(H+L)/2+1e-8),11,T),7,T)))
    # Alpha#067: IndClass
    skip(67,"IndClass")
    # Alpha#068
    safe_add('alpha_068', pd.Series(np.where(ts_rank(ts_corr(cs_rank(H,D),cs_rank(ADV20,D),9,T),14,T)<cs_rank(delta(C*0.518+L*0.482,1,T),D),-1,0),index=df.index))
    # Alpha#069-070: IndClass
    skip(69,"IndClass"); skip(70,"IndClass")
    # Alpha#071
    safe_add('alpha_071', np.maximum(ts_rank(decay_linear(ts_corr(ts_rank(C,3,T),ts_rank(ADV20,12,T),18,T),4,T),15,T), ts_rank(decay_linear(cs_rank(L+O-2*VWAP,D)**2,16,T),4,T)))
    # Alpha#072
    safe_add('alpha_072', cs_rank(decay_linear(ts_corr((H+L)/2,ADV20,9,T),10,T),D)/(cs_rank(decay_linear(ts_corr(ts_rank(VWAP,4,T),ts_rank(V,19,T),7,T),3,T),D)+1e-8))
    # Alpha#073
    safe_add('alpha_073', -1*cs_rank(decay_linear(delta(VWAP,5,T),3,T),D)+ts_rank(decay_linear((delta(O*0.147+L*0.853,2,T)/(O*0.147+L*0.853+1e-8))*-1,3,T),16,T))
    # Alpha#074
    safe_add('alpha_074', pd.Series(np.where(cs_rank(ts_corr(C,ts_sum(ADV20,37,T),15,T),D)<cs_rank(ts_corr(cs_rank(H*0.026+VWAP*0.974,D),cs_rank(V,D),11,T),D),-1,0),index=df.index))
    # Alpha#075
    safe_add('alpha_075', pd.Series(np.where(cs_rank(ts_corr(VWAP,V,4,T),D)<cs_rank(ts_corr(cs_rank(L,D),cs_rank(ADV20,D),12,T),D),1,0),index=df.index))
    # Alpha#076-080: IndClass/sector
    for n in [76,77,78,79,80]: skip(n,"IndClass")
    # Alpha#081
    safe_add('alpha_081', pd.Series(np.where(cs_rank(ts_corr(VWAP,ts_sum(ADV20,50,T),8,T),D)<cs_rank(ts_corr(cs_rank(VWAP,D),cs_rank(V,D),5,T),D),-1,0),index=df.index))
    # Alpha#082-083: IndClass
    skip(82,"IndClass"); skip(83,"sector")
    # Alpha#084
    safe_add('alpha_084', signed_power(ts_rank(VWAP-ts_max(VWAP,15,T),21,T), delta(C,5,T)))
    # Alpha#085
    safe_add('alpha_085', cs_rank(ts_corr(H*0.877+C*0.123,ADV20,10,T),D)**cs_rank(ts_corr(ts_rank((H+L)/2,4,T),ts_rank(V,10,T),7,T),D).clip(0,2))
    # Alpha#086
    safe_add('alpha_086', pd.Series(np.where(ts_rank(ts_corr(C,ts_sum(ADV20,15,T),6,T),20,T)<cs_rank(C-VWAP,D),-1,0),index=df.index))
    # Alpha#087-091: IndClass
    for n in [87,88,89,90,91]: skip(n,"IndClass")
    # Alpha#092
    safe_add('alpha_092', np.minimum(ts_rank(decay_linear((C*0.35+VWAP*0.65-delay(C*0.35+VWAP*0.65,2,T))/(C*0.35+VWAP*0.65+1e-8)*-1,3,T),17,T), ts_rank(decay_linear(ts_corr(cs_rank(L,D),cs_rank(ADV20,D),6,T),2,T),7,T)))
    # Alpha#093: IndClass
    skip(93,"IndClass")
    # Alpha#094
    safe_add('alpha_094', -1*cs_rank(VWAP-ts_min(VWAP,12,T),D)**ts_rank(ts_corr(ts_rank(VWAP,20,T),ts_rank(ADV20,4,T),18,T),3,T).clip(0,2))
    # Alpha#095
    safe_add('alpha_095', pd.Series(np.where(cs_rank(O-ts_min(O,12,T),D)<ts_rank(cs_rank(ts_corr(ts_sum((H+L)/2,19,T),ts_sum(ADV20,19,T),13,T),D),12,T),1,0),index=df.index))
    # Alpha#096
    safe_add('alpha_096', -1*np.maximum(ts_rank(decay_linear(ts_corr(cs_rank(VWAP,D),cs_rank(V,D),4,T),4,T),8,T), ts_rank(decay_linear(ts_argmax(ts_corr(ts_rank(C,7,T),ts_rank(ADV20,4,T),3,T),12,T),14,T),13,T)))
    # Alpha#097: IndClass
    skip(97,"IndClass")
    # Alpha#098
    safe_add('alpha_098', cs_rank(decay_linear(ts_corr(VWAP,ts_sum(ADV20,26,T),5,T),7,T),D)-cs_rank(decay_linear(ts_rank(ts_argmin(ts_corr(cs_rank(O,D),cs_rank(ADV20,D),21,T),9,T),7,T),8,T),D))
    # Alpha#099
    safe_add('alpha_099', pd.Series(np.where(cs_rank(ts_corr(ts_sum((H+L)/2,20,T),ts_sum(ADV20,20,T),9,T),D)<cs_rank(ts_corr(L,V,6,T),D),-1,0),index=df.index))
    # Alpha#100
    safe_add('alpha_100', -1*cs_rank(ts_corr(cs_rank(V,D),cs_rank((H+L)/2,D),5,T),D))
    # Alpha#101
    safe_add('alpha_101', (C-O)/(H-L+0.001))
    
    # === 自定义学术因子 ===
    safe_add('alpha_cust_rev1', -1*R)
    safe_add('alpha_cust_rev10', -1*df.groupby('Ticker')['Adj Close'].pct_change(10))
    r60=df.groupby('Ticker')['Adj Close'].pct_change(60); r5=df.groupby('Ticker')['Adj Close'].pct_change(5)
    safe_add('alpha_cust_mom60', r60-r5)
    r120=df.groupby('Ticker')['Adj Close'].pct_change(120); r20=df.groupby('Ticker')['Adj Close'].pct_change(20)
    safe_add('alpha_cust_mom120', r120-r20)
    v5=df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(5).mean()); v20b=df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(20).mean())
    safe_add('alpha_cust_vratio', v5/(v20b+1))
    v60=df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(60).mean()); v120=df.groupby('Ticker')['Volume'].transform(lambda x: x.rolling(120).mean())
    safe_add('alpha_cust_vtrend', v60/(v120+1))
    vs=df.groupby('Ticker')['Return'].transform(lambda x: x.rolling(20).std()); vl=df.groupby('Ticker')['Return'].transform(lambda x: x.rolling(60).std())
    safe_add('alpha_cust_volchg', -1*vs/(vl+1e-8))
    
    print(f"\n计算完成! 成功: {computed}个, 跳过(缺IndClass/cap): {skipped}个")
    return df, alpha_cols