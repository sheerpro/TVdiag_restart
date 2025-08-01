import pandas as pd
from drain.drain_template_extractor import *
from utils import io_util

data: dict = io_util.load('/home/fuxian/lky/TVDiag_new_onlineboutique/extractor/onlineboutique/post-data-10.pkl')
label_df = pd.read_csv('/home/fuxian/DataSet/NewDataset/ob/allgroundtruth.csv', index_col=0)

logs = []
for idx, row in tqdm(label_df.iterrows(), total=label_df.shape[0]):
    if row['data_type'] == 'test':
        continue
    chunk = data[idx]
    logs.extend(chunk['log']['message'].values.tolist())

miner = extract_templates(
    log_list=logs, 
    save_pth='drain/onlineboutique-drain.pkl')
miner = io_util.load('drain/onlineboutique-drain.pkl')

sorted_clusters = sorted(miner.drain.clusters, key=lambda it: it.size, reverse=True)
template_ids = []
template_counts = []
templates = []

for cluster in sorted_clusters:
    templates.append(cluster.get_template())
    template_ids.append(cluster.cluster_id)
    template_counts.append(cluster.size)

template_df = pd.DataFrame(data={
    'id': template_ids,
    'template': templates,
    'count': template_counts
})
template_df.to_csv('./drain/onlineboutique-template.csv', index=False)
