"""gtex dataset."""
g_quick_build=True

import csv
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_datasets as tfds

SAVE_PATH="data/gtex"
import os
os.makedirs(SAVE_PATH, exist_ok=True)

GENES_COMMIT = 'ad9631bb4e77e2cdc5413b0d77cb8f7e93fc5bee'
ENTREZ_ANNOTATION_URL = 'https://raw.githubusercontent.com/cognoma/genes/{}/data/genes.tsv'.format(GENES_COMMIT)
CLASS_WEIGHTS={0: 0.4812292358803987, 1: 2.2457364341085273, 2: 27.590476190476192, 3: 0.6236813778256189, 4: 0.43400749063670413, 5: 0.21930355791067374, 6: 1.2623093681917212, 7: 30.49473684210526, 8: 0.7437740693196405, 9: 0.4009688581314879, 10: 64.37777777777778, 11: 0.672938443670151, 12: 6.510112359550562, 13: 2.563716814159292, 14: 1.0024221453287196, 15: 0.721544209215442, 16: 0.9360258481421648, 17: 3.218888888888889, 18: 1.7664634146341462, 19: 2.0473498233215546, 20: 2.3648979591836734, 21: 3.5765432098765433, 22: 0.3202874516307352, 23: 3.098395721925134, 24: 2.404149377593361, 25: 1.6139275766016714, 26: 1.6049861495844875, 27: 0.8872894333843798, 28: 4.080281690140845, 29: 3.7141025641025642}
SAMPLE_IDS=[]



_LICENSE = f"""
"""

_DESCRIPTION = f""" Downloads v8 GTEx and cognoma annotations, filters genes that are
 not in entrez that are not named in and returns a
dataset with an array of counts and a tissue type (SMTS, not
SMTSD). To build this dataset requires every bit of 36GB RAM. 

Rows with na's are dropped, medians replace duplicates (199 genes are duplicated, some more than twice, for a total of 1608 values)

There are originally 56,203 GTEx annotations which are collapsed down to 18,963 annotions after annotating with protein-encoding entrez gene ids.

"Bone Marrow" is in the GTEx dataset but has no samples with gene counts and is not included.

Build dataset with something like:
`tfds build --register_checksums  --overwrite sbr/datasets/structured/gtex`
or `tfds build --register_checksums  --rebuild sbr/datasets/structured/gtex`
or even ``tfds build --register_checksums sbr/datasets/structured/gtex`


Upon `tfds build`:

+ Gene order is written to: f"{SAVE_PATH}/gene_ids.txt"; gene id annotations are here: {ENTREZ_ANNOTATION_URL}
+ Superclass sample count is written to: {SAVE_PATH}//superclass-count.tsv
+ Dataframe of SAMPID, SMTS, sample_id, <18,963 gene counts>, tissue type, and  is written to: {SAVE_PATH}/expr.ftr


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
model.add(tf.keras.layers.Dense(info.features['target'].num_classes, activation=tf.nn.softmax, name="output"))
model.summary()
model.compile(optimizer=OPTIMIZER,loss=LOSS)
model.fit(ds, epochs=EPOCHS,
          class_weights = info.metadata["class_weights"])
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
class_names = info.features['target'].names
y = tf.one_hot(le.fit.transform(y)

```
TROUBLE SHOOTING:
* URLError - 
If you see the following error:
```
urllib.error.URLError: <urlopen error [Errno -3] Temporary failure in name resolution>
```
Try disconnecting from any VPN's
* Sample IDS -
Sample id's can be recovered from data/gtex/samp_ids.txt, in order as long as the dataset is not shuffled.

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

  VERSION = tfds.core.Version('1.0.2')
  RELEASE_NOTES = {
      '1.0.2': 'Initial release.',
  }

  def _dl_files(self, dl_manager):
    ####
    # Load in the data
    ####
    path = dl_manager.download_and_extract(COUNTS_URL)

    extracted_paths = dl_manager.download_and_extract({
      'counts': 'https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz',
      'attributes': 'https://storage.googleapis.com/gtex_analysis_v8/annotations/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt',
      'entrez_annotations': ENTREZ_ANNOTATION_URL,
      'update_entrez': 'https://raw.githubusercontent.com/cognoma/genes/{}/data/updater.tsv'.format(GENES_COMMIT),
    })

    print(f"+ Reading GTEx sample attributes from {extracted_paths['attributes']}...")
    with extracted_paths['attributes'].open() as f:
      attr_df = pd.read_table(f)

    print(f"+ Reading entrez annotations from {extracted_paths['entrez_annotations']}...")
    with extracted_paths['entrez_annotations'].open() as f:
      gene_df = pd.read_table(f)
    # Only consider protein-coding genes
    gene_df = (
        gene_df.query("gene_type == 'protein-coding'")
    )

    print(f"+ Reading old-to-new annotations from {extracted_paths['update_entrez']}...")
    with extracted_paths['update_entrez'].open() as f:
      updater_df = pd.read_table(f)
    # Load gene updater - old to new Entrez gene identifiers
    old_to_new_entrez_dict = dict(zip(updater_df.old_entrez_gene_id,
                                 updater_df.new_entrez_gene_id))


    print("+ Reading gene expression - this takes a little while")
    with extracted_paths['counts'].open() as f:
      expr_df = pd.read_table(f, sep='\t', skiprows=2, index_col=1, nrows=NUM_ROWS)

    return expr_df, attr_df, gene_df, old_to_new_entrez_dict 


  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    # _dl_files()
    print("+++++++++INFO CALLED") # xxx

    from tensorflow_datasets.core.dataset_info import MetadataDict
    metadata = tfds.core.MetadataDict()
    metadata['class_weights']=CLASS_WEIGHTS
    metadata['sample_ids']=SAMPLE_IDS

    info= tfds.core.DatasetInfo(
      builder=self,
      description=_DESCRIPTION,
      features=tfds.features.FeaturesDict({
        'features': tfds.features.Tensor(shape=(NUM_GENES,), dtype=tf.float64),
        'target': tfds.features.ClassLabel(names=TISSUE_LIST),
      }),
      homepage='https://gtexportal.org/',
      citation=_CITATION,
      supervised_keys=('features','target'),
      metadata=metadata,
      license=_LICENSE,
    )

    return info


  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators. Computes and saves class weights
    """

    print("+++++++++GENERATORS CALLED") # xxx
    expr_df, attr_df, gene_df, old_to_new_entrez_dict = self._dl_files(dl_manager)

    ####
    # Merge and transform the data
    ####

    print("+ Get GTEx gene mapping")
    expr_gene_ids = (
      expr_df
      .loc[:, ['Name']]
      .reset_index()
      .drop_duplicates(subset='Description')
    )
    
    print("+ Perform inner merge gene df to get new gene id mapping")
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
    expr_df=expr_df.rename(index=old_to_new_entrez_dict) # add in gene annotations
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

    ####
    # Save relevant data not delivered with tfds.load
    ####
    print("+ save expression as a feather-formatted file")
    expr_df.reset_index().to_feather(f"{SAVE_PATH}/expr.ftr")
    file=f"{SAVE_PATH}/gene_ids.txt"
    print(f"+ Write out gene ids in order ({file})")
    with open(file,"a") as f:
      for col in expr_df.columns:
        f.write(f"{col}\n")

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

    # xxx
    # normalize gene counts, compute weights, splits

    ####
    # Prepare the dataset records for generating examples
    ####

    # might be able to use zip above to make this step a lot faster, but more memory will be required:
    print("+ Label the expression rows") 
    label_df=attr_df[["SAMPID",
                      "SMTS"]
                     ].merge(expr_df, 
                             how='inner', 
                             left_on="SAMPID", 
                             right_on="sample_id")
    # Create records list, with no header, like this:
    # id,gene1,...genen,tissue_string
    records = label_df.iloc[:, list(np.r_[0,2:(len(expr_df.columns)+2), 1])].values.tolist()


    df = pd.DataFrame(records)
    num_classes = len(TISSUE_LIST)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = tf.one_hot(le.fit_transform(list(df.iloc[:,-1])), num_classes)

    # compute class weights
    # bigger number reflects fewer samples
    from sklearn.utils.class_weight import compute_class_weight
    y_integers = np.argmax(y, axis=1)
    class_weights = compute_class_weight(class_weight='balanced', 
                                         classes=np.unique(y_integers), 
                                         y=y_integers)
    CLASS_WEIGHTS = dict(enumerate(class_weights))
    print(f"++ Class weights set: {CLASS_WEIGHTS}")

    SAMPLE_IDS = list(label_df['SAMPID'])
    file=f"{SAVE_PATH}/samp_ids.txt"
    print(f"+ Write out sample ids in order ({file})")
    with open(file,"a") as f:
      for id in label_df['SAMPID']:
        f.write(f"{id}\n")

    info = self._info()  # xxx this isn't working yet, so can't use class_weights right now.
    # info.write_to_directory()

    print("+++++++++GENERATORS DONE") # xxx
    # Specify the splits
    return [
        tfds.core.SplitGenerator(
          name=tfds.Split.TRAIN, 
          gen_kwargs=dict(records= records)), #gets passed to generate_examples as args
    ]


  def _generate_examples(self, records):
    for row in records: # - won't it be slow to iterate through records this way? xxx
      yield row[0], {
        "features": row[1:-1],
        "target": row[-1], # xxx this is a text feature, make it an integer?
        #"weights": [1.,2.,3.]
      }
