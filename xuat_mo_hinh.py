# -*- coding: utf-8 -*-
"""
Xuat mo hinh Gradient Boosting (BT4) sang JavaScript de chay trong trinh duyet.
Dau vao : df_clean.csv (7.436 dong) - giong het notebook bai_tap_4/model.ipynb
Dau ra  : mo_hinh.js  - chua pipeline + rung cay + so lieu danh gia
"""
import json, sys, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

SEED = 42
np.random.seed(SEED)

DATA = sys.argv[1] if len(sys.argv) > 1 else 'df_bt3.csv'
df = pd.read_csv(DATA)
print('Du lieu:', df.shape)

num_cols = ['dien_tich_m2','log_dien_tich','so_phong_ngu','so_tang','so_wc','mat_do_xay','mat_tien_m','rong_hem_m']
cat_cols = ['quan_huyen','phuong','loai_hinh']
bin_cols = ['la_hem','la_mat_tien','co_so_hong','co_noi_that','oto_vao','gan_truong_cho','chinh_chu','co_tt_phong_ngu','co_tt_mat_tien']
feature_cols = num_cols + cat_cols + bin_cols

X = df[feature_cols].copy()
y = df['log_gia_ban'].copy()
y_cash = df['gia_ban'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
y_test_cash = y_cash.loc[y_test.index].values
y_train_cash = y_cash.loc[y_train.index].values

pre = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(missing_values=-1, strategy='median')),
                      ('scaler', StandardScaler())]), num_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_cols)
], remainder='passthrough')

CFG = dict(n_estimators=400, learning_rate=0.05, max_depth=5,
           min_samples_leaf=10, subsample=0.8, max_features=0.8, random_state=SEED)
pipe = make_pipeline(pre, GradientBoostingRegressor(**CFG))
pipe.fit(X_train, y_train)

gbr = pipe.named_steps['gradientboostingregressor']
ct  = pipe.named_steps['columntransformer']

# ---- Duan smearing factor (khu chech Jensen khi doi log -> tien mat)
resid_train = y_train.values - pipe.predict(X_train)
SF = float(np.mean(np.exp(resid_train)))

pred_log  = pipe.predict(X_test)
pred_cash = np.exp(pred_log) * SF
r2   = r2_score(y_test, pred_log)
mae  = mean_absolute_error(y_test_cash, pred_cash)
ape  = np.abs(pred_cash - y_test_cash) / y_test_cash * 100
print(f'R2_log={r2:.4f}  MAE={mae:,.1f}tr  MAPE={ape.mean():.2f}%  MedAPE={np.median(ape):.2f}%')

# ---- Baseline B0 (median quan x loai hinh) va B1 (median don gia x dien tich)
tr = df.loc[X_train.index]
b0_tbl = tr.groupby(['quan_huyen','loai_hinh'])['gia_ban'].median()
b1_tbl = tr.groupby(['quan_huyen','loai_hinh'])['don_gia'].median()
glob_b0, glob_b1 = tr['gia_ban'].median(), tr['don_gia'].median()
te = df.loc[X_test.index]
b0 = np.array([b0_tbl.get((q,l), glob_b0) for q,l in zip(te.quan_huyen, te.loai_hinh)])
b1 = np.array([b1_tbl.get((q,l), glob_b1) for q,l in zip(te.quan_huyen, te.loai_hinh)]) * te.dien_tich_m2.values
mae_b0, mae_b1 = mean_absolute_error(y_test_cash,b0), mean_absolute_error(y_test_cash,b1)

# ---- Khoang tin cay: phan vi cua ty le thuc te / du bao tren tap test
ratio = y_test_cash / pred_cash
q_lo, q_hi = float(np.quantile(ratio,0.10)), float(np.quantile(ratio,0.90))

# ---- Trich pipeline sang dang JS doc duoc
imp = ct.named_transformers_['num'].named_steps['imputer']
sc  = ct.named_transformers_['num'].named_steps['scaler']
ohe = ct.named_transformers_['cat']
cats = {c: list(map(str, ohe.categories_[i])) for i, c in enumerate(cat_cols)}

def r(a, nd=6): return [round(float(v), nd) for v in a]

trees = []
for est in gbr.estimators_[:, 0]:
    t = est.tree_
    trees.append({'f': t.feature.tolist(), 't': r(t.threshold, 5),
                  'l': t.children_left.tolist(), 'r': t.children_right.tolist(),
                  'v': r(t.value[:, 0, 0], 6)})

# ---- Kiem chung: mo phong dung thuat toan JS se chay
feat_names = [c.split('__')[-1] for c in ct.get_feature_names_out()]
def build_vec(row):
    v = []
    for i, c in enumerate(num_cols):
        x = row[c]
        if x == -1 or pd.isna(x): x = imp.statistics_[i]
        v.append((x - sc.mean_[i]) / sc.scale_[i])
    for c in cat_cols:
        for lv in cats[c][1:]:
            v.append(1.0 if str(row[c]) == lv else 0.0)
    for c in bin_cols:
        v.append(float(row[c]))
    return np.array(v)

def js_predict(vec):
    p = float(gbr.init_.constant_[0, 0])
    for tr_ in trees:
        n = 0
        while tr_['l'][n] != -1:
            n = tr_['l'][n] if vec[tr_['f'][n]] <= tr_['t'][n] else tr_['r'][n]
        p += CFG['learning_rate'] * tr_['v'][n]
    return p

chk = X_test.head(50)
mine = np.array([js_predict(build_vec(row)) for _, row in chk.iterrows()])
dev = float(np.max(np.abs(mine - pipe.predict(chk))))
print('Sai lech toi da giua JS-logic va sklearn:', dev)
assert dev < 1e-6, 'Khong khop!'

# ---- Bang mat bang gia theo quan (de tra cuu + bieu do)
g = df.groupby('quan_huyen').agg(don_gia=('don_gia','median'), gia=('gia_ban','median'),
                                 dt=('dien_tich_m2','median'), n=('gia_ban','size')).reset_index()
quan_tbl = [{'quan': row.quan_huyen, 'don_gia': round(row.don_gia,1), 'gia': round(row.gia,1),
             'dt': round(row.dt,1), 'n': int(row.n)} for row in g.sort_values('don_gia',ascending=False).itertuples()]

# don gia trung vi theo quan x loai hinh (dung cho phan "so sanh thi truong")
dg_qh = {f'{q}|{l}': round(v,1) for (q,l), v in df.groupby(['quan_huyen','loai_hinh'])['don_gia'].median().items()}

# phuong theo tung quan (dropdown phu thuoc)
phuong_map = {q: sorted(set(map(str, s))) for q, s in df.groupby('quan_huyen')['phuong']}

# do quan trong dac trung, gop nhom
impv = pd.Series(gbr.feature_importances_, index=feat_names)
def grp(name):
    if name in ('dien_tich_m2','log_dien_tich','mat_do_xay'): return 'Diện tích (3 biến cùng gốc)'
    if name in cats['quan_huyen'] or name in ('quan_huyen',): return 'Quận/huyện'
    if name in cats['phuong']: return 'Phường'
    if name in cats['loai_hinh']: return 'Loại hình'
    if name in ('so_phong_ngu','so_wc','so_tang'): return 'Số phòng / tầng'
    if name in ('mat_tien_m','rong_hem_m','la_hem','la_mat_tien','oto_vao'): return 'Mặt tiền / hẻm'
    return 'Pháp lý & tiện ích'
gi = impv.groupby(impv.index.map(grp)).sum().sort_values(ascending=False)
nhom_tt = [{'ten': k, 'pct': round(v*100,1)} for k, v in gi.items()]

# ---- Danh sach tin rao mau de hien thi tren trang (lay ngau nhien, trai deu cac quan)
def lam_sach_tieu_de(t):
    t = str(t).replace('<SO>', '…').replace('  ', ' ').strip()
    return (t[:95] + '…') if len(t) > 95 else t

idx = []
for _, g in df.groupby('quan_huyen'):
    idx += list(g.sample(min(len(g), 18), random_state=SEED).index)
mau = df.loc[idx].sample(frac=1, random_state=SEED)
tin_rao = [{
    'tieu_de': lam_sach_tieu_de(t.tieu_de_sach), 'quan': t.quan_huyen,
    'phuong': t.phuong if t.phuong != 'Khong_ro' else '', 'loai_hinh': t.loai_hinh,
    'dt': round(float(t.dien_tich_m2), 1), 'pn': int(t.so_phong_ngu) if t.so_phong_ngu > 0 else 0,
    'wc': int(t.so_wc) if t.so_wc > 0 else 0, 'tang': int(t.so_tang) if t.so_tang > 0 else 0,
    'gia': round(float(t.gia_ban), 1), 'don_gia': round(float(t.don_gia), 1),
    'nguon': t.nguon_du_lieu, 'url': str(t.url)[:200]
} for t in mau.itertuples()]
print('Tin rao mau:', len(tin_rao))

out = {
  'meta': {'n_dong': int(len(df)), 'n_train': int(len(X_train)), 'n_test': int(len(X_test)),
           'seed': SEED, 'cau_hinh': {k: (v if not isinstance(v,(np.integer,np.floating)) else float(v)) for k,v in CFG.items()}},
  'chi_so': {'r2_log': round(r2,4), 'mae': round(mae,1), 'mape': round(float(ape.mean()),2),
             'medape': round(float(np.median(ape)),2),
             'mae_b0': round(mae_b0,1), 'mae_b1': round(mae_b1,1),
             'giam_mae_b0': round((1-mae/mae_b0)*100,1), 'giam_mae_b1': round((1-mae/mae_b1)*100,1)},
  'khoang_tin_cay': {'lo': round(q_lo,4), 'hi': round(q_hi,4)},
  'pipeline': {'num_cols': num_cols, 'cat_cols': cat_cols, 'bin_cols': bin_cols,
               'imputer_median': r(imp.statistics_), 'mean': r(sc.mean_), 'scale': r(sc.scale_),
               'categories': cats},
  'gbr': {'init': round(float(gbr.init_.constant_[0,0]),6), 'lr': CFG['learning_rate'],
          'smearing': round(SF,6), 'trees': trees},
  'quan_tbl': quan_tbl, 'don_gia_quan_loaihinh': dg_qh, 'phuong_theo_quan': phuong_map,
  'tam_quan_trong': nhom_tt,
  'loai_hinh': sorted(map(str, df['loai_hinh'].unique())),
  'tin_rao': tin_rao,
  'so_tin_quan_loaihinh': {f'{q}|{l}': int(n) for (q,l), n in df.groupby(['quan_huyen','loai_hinh']).size().items()},
}
js = '// Tu dong sinh boi xuat_mo_hinh.py - KHONG sua tay\nconst MO_HINH = ' + json.dumps(out, ensure_ascii=False, separators=(',',':')) + ';\n'
open('mo_hinh.js','w',encoding='utf-8').write(js)
print('Da ghi mo_hinh.js:', round(len(js)/1024/1024,2), 'MB')
