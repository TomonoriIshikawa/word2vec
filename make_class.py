from sklearn.cluster import KMeans
from gensim.models import word2vec

import pandas as pd
import numpy as np

# モデルを詠みこむ
model = word2vec.Word2Vec.load("./wiki.model")

df = pd.read_csv('○○.csv', names=('id','name','parent_id','update_time','create_time'))

# 計算したい単語に変換する
df['cluster_name'] = ["運輸", "建設", "広告", "エンタメ", "不動産", "アウトソーシング", "資源", "その他", "飲食", "医療", "生活", "製造", "機械", "小売",
                     "自動車", "レジャー", "IT", "コンサルティング", "生活", "アパレル", "機械", "商社", "マスコミ", "食品", "ゲーム", "通信", "人材",
                        "教育", "家電", "金融", "エネルギー","公共"]

# idとクラスタリングの列を取得
df = df[["id", "name", "cluster_name"]]

# 単語をベクトルに変換する
arr = np.array([model["運輸"],model["建設"],model["広告"],model["エンタメ"],model["不動産"],model["アウトソーシング"],
                model["資源"],model["その他"],model["飲食"],model["医療"],model["生活"],model["製造"],model["機械"],
                model["小売"],model["自動車"],model["レジャー"],model["IT"],model["コンサルティング"],model["生活"],model["アパレル"],
                model["機械"],model["商社"],model["マスコミ"],model["食品"],model["ゲーム"],model["通信"],model["人材"],model["教育"],
                model["家電"],model["金融"],model["エネルギー"],model["公共"]])

df1 = pd.DataFrame(data=arr)

df2 = pd.concat([df,df1],axis=1)

# 15個出クラスタリング
kmeans_model = KMeans(n_clusters=15, random_state=20).fit(df1.iloc[:,0:])

labels = kmeans_model.labels_

df['class'] = labels