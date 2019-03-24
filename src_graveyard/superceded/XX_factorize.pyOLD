
from tqdm import tqdm

indexer = {}
kept_cols = ['dataset_type','AdoptionSpeed']
factorized_cols = [col for col in df_all_final.columns if col not in kept_cols]

cat_cols = [col for col in factorized_cols ]

for col in factorized_cols:
    print(col)
    _, indexer[col] = pd.factorize(df_all_final[col].astype(str))

for col in tqdm(factorized_cols):
    print(col)
    df_all_final[col] = indexer[col].get_indexer(df_all_final[col].astype(str))

# for col in factorized_cols:
#     df_all_final[col] =

df_all_final.info()
