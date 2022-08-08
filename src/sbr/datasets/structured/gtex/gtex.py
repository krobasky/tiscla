"""gtex dataset."""
g_quick_build=False

import csv
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

SAVE_PATH="data/gtex"

_DESCRIPTION = """ Downloads v8 GTEx and cognoma annotations, filters genes that are
 not in entrez that are not named in and returns a
dataset with an array of counts and a tissue type (SMTS, not
SMTSD). To build this dataset requires every bit of 36GB RAM. 

Rows with na's are dropped, medians replace duplicates (199 genes are duplicated, some more than twice, for a total of 1608 values)

There are originally 56,203 GTEx annotations which are collapsed down to 18,963 annotions.

Dataframe of SAMPID, SMTS, sample_id, <18,963 gene counts>, tissue type, and  is written to: f"{SAVE_PATH}/expr.ftr"
Gene order is written to: f"{SAVE_PATH}/gene_ids.txt"
superclass sample count is written to: f"{SAVE_PATH}//superclass-count.tsv"

"Bone Marrow" is in the GTEx dataset but has no samples with gene counts and is not included.

Training example:
```
BATCH_SIZE=32
EPOCHS=1
DIM=1000
OPTIMIZER="adam"
LOSS="mse"
import tensorflow as tf
import tensorflow_datasets as tfds
from sbr.datasets.structured import gtex
ds, info = tfds.load("gtex", split="train", with_info = True, 
                     as_supervised=True)
ds = ds.cache().shuffle(info.splits["train"].num_examples).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# compile and fit the model
from sbr.layers import BADBlock
model = tf.keras.models.Sequential()
model.add(BADBlock(units=DIM, activation=tf.nn.relu, input_shape=info.features['features'].shape, name="BAD_1", dropout_rate=0.50))
model.add(tf.keras.layers.Dense(info.features['label'].num_classes, activation=tf.nn.softmax, name="output"))
model.summary()
model.compile(optimizer=OPTIMIZER,loss=LOSS)
model.fit(ds, epochs=EPOCHS)
```
Prediction example:
```
import numpy as np
# predict on the first batch of test dataset 'ds_test' (can use ds as demonstration):
[np.argmax(logits) for logits in model.predict(ds_test.take(1)) ]
```
Example: retrieve X, y, class_names:
```
ds, info = tfds.load("gtex", split="train", with_info = True, as_supervised=True)
l=list(iter(ds.take(info.splits['train'].num_examples)))
X, y = map(np.array, zip(*l))
class_names = info.features['label'].names
y = tf.one_hot(le.fit.transform(y)

```
"""

_CITATION = """
"""

NUM_ROWS=None
NUM_GENES = 18963
TISSUE_LIST = ['Colon', 'Heart', 'Blood', 'Vagina', 'Thyroid', 'Liver', 'Salivary_Gland', 'Pancreas', 'Cervix_Uteri', 'Prostate', 'Ovary', 'Skin', 'Pituitary', 'Small_Intestine', 'Fallopian_Tube', 'Adrenal_Gland', 'Nerve', 'Adipose_Tissue', 'Spleen', 'Stomach', 'Muscle', 'Blood_Vessel', 'Lung', 'Esophagus', 'Brain', 'Testis', 'Uterus', 'Kidney', 'Bladder', 'Breast']

if g_quick_build == True:
  NUM_ROWS=256 
  NUM_GENES=105 


COUNTS_URL = "https://raw.githubusercontent.com/krobasky/toy-classifier/main/toy-counts.csv"

COUNTS_FILE="GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
#COUNTS_FILE="toy-counts.csv"

class Gtex(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for gtex dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    return tfds.core.DatasetInfo(
      builder=self,
      description=_DESCRIPTION,
      features=tfds.features.FeaturesDict({
        'features': tfds.features.Tensor(shape=(NUM_GENES,), dtype=tf.float64),
        'label': tfds.features.ClassLabel(names=TISSUE_LIST),
      }),
      supervised_keys=('features','label'),
      homepage='https://gtexportal.org/',
      citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract(COUNTS_URL)

    genes_commit = 'ad9631bb4e77e2cdc5413b0d77cb8f7e93fc5bee'
    extracted_paths = dl_manager.download_and_extract({
      'counts': 'https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz',
      'attributes': 'https://storage.googleapis.com/gtex_analysis_v8/annotations/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt',
      'entrez_annotations': 'https://raw.githubusercontent.com/cognoma/genes/{}/data/genes.tsv'.format(genes_commit),
      'update_entrez': 'https://raw.githubusercontent.com/cognoma/genes/{}/data/updater.tsv'.format(genes_commit),
    })

    import pandas as pd
    with extracted_paths['attributes'].open() as f:
      attr_df = pd.read_table(f)

    print("+ Reading gene expression - this takes a little while")
    with extracted_paths['counts'].open() as f:
      expr_df = pd.read_table(f, sep='\t', skiprows=2, index_col=1, nrows=NUM_ROWS)

    print("+ Get GTEx gene mapping")
    expr_gene_ids = (
      expr_df
      .loc[:, ['Name']]
      .reset_index()
      .drop_duplicates(subset='Description')
    )
    
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

    print("+ Perform inner merge gene df to get new gene id mapping")
    gene_df=get_gene_df()
    map_df = expr_gene_ids.merge(gene_df, how='inner', left_on='Description', right_on='symbol')

    # transform expression matrix - # needs ALL the MEMORY!
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
    print("+ rename index") # xxx maybe don't do this?
    expr_df.index.rename('sample_id', inplace=True)


    # change gene integer ids to strings so feather will accept column names  
    expr_df.columns=expr_df.columns.astype(str)

    NUM_GENES=len(expr_df.columns)

    print("+ save expression as a feather-formatted file")
    expr_df.reset_index().to_feather(f"{SAVE_PATH}/expr.ftr")
    file=f"{SAVE_PATH}/gene_ids.txt"
    print(f"+ Write out gene ids in order ({file})")
    with open(file,"a") as f:
      for col in expr_df.columns:
        f.write(f"{col}\n")

    print("+ Write out superclass names")

    print("++ Change attr tissue type names to something directory-friendly")
    attr_df["SMTS"] = attr_df["SMTS"].str.strip()
    attr_df["SMTS"] = attr_df["SMTS"].str.replace(' - ','-')
    attr_df["SMTS"] = attr_df["SMTS"].str.replace(' \(','__').replace('\)','__')
    attr_df["SMTS"] = attr_df["SMTS"].str.replace(' ','_')

    TISSUE_LIST=set(attr_df["SMTS"])
    print(f"++ Class names set: {TISSUE_LIST}")

    strat = attr_df.set_index('SAMPID').reindex(expr_df.index).SMTS
    tissuetype_count_df = (
        pd.DataFrame(strat.value_counts())
        .reset_index()
        .rename({'index': 'tissuetype', 'SMTS': 'n ='}, axis='columns')
    )

    file = f'{SAVE_PATH}/superclass-count.tsv'
    print(f"+Write tissue type counts {file}")
    tissuetype_count_df.to_csv(file, sep='\t', index=False)
    
    print(f"+ tissue type counts: {tissuetype_count_df}")

    print("""+ Reload each observation like such:
    import gzip
    import numpy as np
    with gzip.GzipFile(f'data/gtex/<cls>/<sample_id>.npy.gz') as f:
        obs=np.load(f)
    """)

    '''
    # read in counts
    all_lines = tf.io.gfile.GFile(extracted_paths["counts"]).read().splitlines()
    records = [l for l in all_lines if l]  # get rid of empty lines
    records = records[1:] #omit header line
    '''
    
    # label the expression samples (might be able to use zip above to make this step a lot faster, but more memory will be required)
    print("+ Label the expression rows") 
    label_df=attr_df[["SAMPID",
                      "SMTS"]
                     ].merge(expr_df, 
                             how='inner', 
                             left_on="SAMPID", 
                             right_on="sample_id")
    #print("+ Done.") 


    # xxx create records list, with no header, like this:
    # id,gene1,...genen,tissue_string
    records = label_df.iloc[:, list(np.r_[0,2:(len(expr_df.columns)+2), 1])].values.tolist()

    # Specify the splits
    return [
        tfds.core.SplitGenerator(
          name=tfds.Split.TRAIN, 
          gen_kwargs=dict(records= records)), #gets passed to generate_examples as args
          # gen_kwargs=dict(records= records)), #gets passed to generate_examples as args
    ]


  def _generate_examples(self, records):
    for row in records: # - won't it be slow to iterate through records this way? xxx
      yield row[0], {
        "features": row[1:-1],
        "label": row[-1], # xxx this is a text feature, make it an integer?
      }
