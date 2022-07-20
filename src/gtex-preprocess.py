import os
import random
import pandas as pd
from sklearn.model_selection import train_test_split

g_write_df = True

print("+ This will take about 10 minutes with a power laptop, but requires a lot of memory for doing a groupby median on the gene expression")
print("+ Results in annotated geneset that is compatible with other datasets, like TCGA and TARGET")
print("""+ First run: 
  get_gtex.sh
""")

# Commit from https://github.com/cognoma/genes
# use this to make a compatible geneset annotation (thanks, Biobombe!)
genes_commit = 'ad9631bb4e77e2cdc5413b0d77cb8f7e93fc5bee'

def get_gene_df():
    url = 'https://raw.githubusercontent.com/cognoma/genes/{}/data/genes.tsv'.format(genes_commit)
    gene_df = pd.read_table(url)

    # Only consider protein-coding genes
    gene_df = (
        gene_df.query("gene_type == 'protein-coding'")
    )
    return gene_df

def get_old_to_new_entrez_ids():
    # Load gene updater - old to new Entrez gene identifiers
    url = 'https://raw.githubusercontent.com/cognoma/genes/{}/data/updater.tsv'.format(genes_commit)
    updater_df = pd.read_table(url)
    old_to_new_entrez = dict(zip(updater_df.old_entrez_gene_id,
                                 updater_df.new_entrez_gene_id))
    return old_to_new_entrez


random.seed(1234)
os.makedirs("data/gtex",exist_ok=True)

attr_path = 'dist/gtex/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt'
attr_df = pd.read_table(attr_path)

###### Process the gene expression data

# This involves updating Entrez gene ids, sorting and subsetting

print("+ Read gene expression - this takes a little while")
os.makedirs(f"data/gtex",exist_ok=True)
expr_path = 'dist/gtex/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz'
expr_df = pd.read_table(expr_path, sep='\t', skiprows=2, index_col=1)


print("+ Get GTEx gene mapping")
expr_gene_ids = (
    expr_df
    .loc[:, ['Name']]
    .reset_index()
    .drop_duplicates(subset='Description')
)
print("+ Perform inner merge gene df to get ensembl to entrez mapping")
gene_df=get_gene_df()
map_df = expr_gene_ids.merge(gene_df, how='inner', left_on='Description', right_on='symbol')
print("+ Save map, expression dataframes")
if g_write_df:
    map_df.reset_index().to_feather(f"data/gtex/map.ftr") # if you run out of memory, this will load fast
else:
    print("+ ! don't write map.ftr")


# transform expression matrix
print("+ *Drop 'Name' column...")
expr_df=expr_df.drop(['Name'], axis='columns')
print("+ *Drop any rows with 'na's...")
expr_df=expr_df.dropna(axis='rows')
print("+ * Use groupby to collapse duplicate genes by median (199 genes are duplicated, some more than twice, for a total of 1608 values) ...")
expr_df=expr_df.groupby(level=0).median()
print("+ *reindex map...")
expr_df=expr_df.reindex(map_df.symbol)
symbol_to_entrez = dict(zip(map_df.symbol, map_df.entrez_gene_id))
print("+ *rename...")
expr_df=expr_df.rename(index=symbol_to_entrez)
print("+ *rename again...")
expr_df=expr_df.rename(index=get_old_to_new_entrez_ids()) # add in gene annotations
print("+ *transpose...")
expr_df = expr_df.transpose()
print("+ *sort by row...")
expr_df = expr_df.sort_index(axis='rows')
print("+ *sort by columns...")
expr_df = expr_df.sort_index(axis='columns')
print("+ rename index")
expr_df.index.rename('sample_id', inplace=True)

# change gene integer ids to strings so feather will accept column names  
expr_df.columns=expr_df.columns.astype(str)

if g_write_df:
    print("+ write expr one more time")
    expr_df.reset_index().to_feather(f"data/gtex/expr.ftr")
    file=f"data/gtex/gene_ids.txt"
    print(f"+ Write out gene ids in order ({file})")
    with open(file,"a") as f:
        for col in expr_df.columns:
            f.write(f"{col}\n")
else:
    print("++ ! Didn't write expr.fltr")
    print("++ ! Didn't write gene_ids_txt")

print("+ Write out superclass names")

print("++ Change attr tissue type names to something directory-friendly")
attr_df["SMTS"] = attr_df["SMTS"].str.strip()
attr_df["SMTS"] = attr_df["SMTS"].str.replace(' - ','-')
attr_df["SMTS"] = attr_df["SMTS"].str.replace(' \(','__').replace('\)','__')
attr_df["SMTS"] = attr_df["SMTS"].str.replace(' ','_')

class_names=set(attr_df["SMTS"])
print(f"++ Class names set: {class_names}")

if g_write_df:
    attr_df[["SAMPID","SMTS"]].to_csv(f"data/gtex/sample_id-superclass_name.tsv", sep="\t", index=False, header=False)
else:
    print("++ ! Didn't write sample_id-superclass_name.tsv")


print("+ Create dir structure for classes")
os.makedirs(f'data/gtex', exist_ok=True)
for cls in class_names:
  os.makedirs(f"data/gtex/{cls}",exist_ok=True)

print(f"+ Create a numpy for each row, write to data/gtex, separate out later")
import gzip
import numpy as np
for idx, nparray in enumerate(np.array(expr_df.iloc[:])):
    nparray=nparray.astype(np.float16)
    sample_id=expr_df.index[idx]
    cls=attr_df.loc[attr_df["SAMPID"]==sample_id, "SMTS"].iloc[0]
    if g_write_df:
        with gzip.GzipFile(f"data/gtex/{cls}/{sample_id}.npy.gz", "w") as f:
            np.save(file=f, arr=nparray)
    else:
          print(f"Didn't write {sample_id}.py.gz.")

strat = attr_df.set_index('SAMPID').reindex(expr_df.index).SMTSD
tissuetype_count_df = (
    pd.DataFrame(strat.value_counts())
    .reset_index()
    .rename({'index': 'tissuetype', 'SMTSD': 'n ='}, axis='columns')
)

file = f'data/gtex/superclass-count.tsv'
print(f"+Write tissue type counts {file}")
if g_write_df:
          tissuetype_count_df.to_csv(file, sep='\t', index=False)
else:
          print(f"Didn't write {file}")

print(f"+ tissue type counts: {tissuetype_count_df}")

print("""+ Reload each observation like such:
import gzip
import numpy as np
with gzip.GzipFile(f'data/gtex/<cls>/<sample_id>.npy.gz') as f:
    obs=np.load(f)
""")


