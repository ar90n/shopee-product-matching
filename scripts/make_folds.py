import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold

from shopee_product_matching.constants import Paths

df = pd.read_csv(Paths.shopee_product_matching / "train.csv")

gkf = GroupKFold(n_splits=5)
df['fold'] = -1
for fold, (_, valid_idx) in enumerate(gkf.split(df, None, df.label_group)):
    df.loc[valid_idx, 'fold'] = fold

le = LabelEncoder()
df.label_group = le.fit_transform(df.label_group)
df[["posting_id", "label_group", "fold"]].to_csv("fold.csv", index=False)