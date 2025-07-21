from collections import Counter
import math
import dgl
from torch.utils.data import DataLoader
from tqdm import tqdm
from core.multimodal_dataset import MultiModalDataSet
from helper import io_util
import json
import pandas as pd
import numpy as np
from process.events.fasttext_w2v import FastTextEncoder
from core.aug import *

class EventProcess():

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.embedding_dim = args.embedding_dim
        self.dataset = args.dataset

    def process(self, reconstruct=False):
        self.data_path = f"data/{self.dataset}"

        #label_path = f"data/{self.dataset}/aiops22/groundtruth-2022-05-01.csv"
        # label_path = f"data/{self.dataset}/label.csv"
        label_path= f"/home/fuxian/lky/TVDiag-main/data/gaia/label.csv"
        metric_path = f"data/{self.dataset}/raw/metric.json"
        trace_path = f"data/{self.dataset}/raw/trace.json"
        log_path = f"data/{self.dataset}/raw/log.json"
        edge_path = f"data/{self.dataset}/raw/edges.pkl"
        node_path = f"data/{self.dataset}/raw/nodes.pkl"

        self.logger.info(f"Load raw events from {self.dataset} dataset")
        self.labels = pd.read_csv(label_path)
        with open(metric_path, 'r', encoding='utf8') as fp:
            self.metrics = json.load(fp)
        with open(trace_path, 'r', encoding='utf8') as fp:
            self.traces = json.load(fp)
        with open(log_path, 'r', encoding='utf8') as fp:
            self.logs = json.load(fp)

        self.edges = io_util.load(edge_path)
        self.nodes = io_util.load(node_path)
        self.types = ['normal'] + self.labels['anomaly_type'].unique().tolist()
        #print("****************************",self.types)
        if reconstruct:
            self.build_embedding()

        return self.build_dataset()

    def build_embedding(self):
        self.logger.info(f"Build embedding for raw events")
        # metric event: (instance, host, metric_name, 'abnormal')
        # trace event: (edge, host, error_type)
        # log event: (instance, eventId)

        data_map = {'metric': self.metrics, 'trace': self.traces, 'log': self.logs}
        # data_map = {'trace': self.traces}
        
        
        for key, data in data_map.items():
            encoder = FastTextEncoder(key, self.nodes, self.types, embedding_dim=self.embedding_dim, epochs=5)
            # encoder = LDAEncoder(num_topics=self.args.embedding_dim)
            # encoder = CNNW2VEncoder(
            #     seq_hidden=self.args.seq_hidden,
            #     embedding_dim=self.args.embedding_dim)

            train_idxs = self.labels[self.labels['data_type']=='train']['index'].values.tolist()
            train_ins_labels = self.labels[self.labels['data_type']=='train']['instance'].values.tolist()
            train_type_labels = self.labels[self.labels['data_type']=='train']['anomaly_type'].values.tolist()
            docs = []
            labels = []
            for i, idx in enumerate(train_idxs):
                for node in self.nodes:
                    if key == 'trace':
                        doc=['&'.join(e) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                    else:
                        doc=['&'.join(e) for e in data[str(idx)] if node in e[0]]
                    docs.append(doc)
                    if node == train_ins_labels[i]:
                        labels.append(f'__label__{self.nodes.index(node)}{self.types.index(train_type_labels[i])}')
                    else:
                        labels.append(f'__label__{self.nodes.index(node)}0')
            encoder.fit(docs, labels)

            # build embedding
            embs = []
            for idx in self.labels['index']:
                # group by instance
                graph_embs = []
                for node in self.nodes:
                    if key == 'trace':
                        doc=['&'.join(e) for e in data[str(idx)] if (node in e[0] or node in e[1])]
                    else:
                        doc=['&'.join(e) for e in data[str(idx)] if node in e[0]]
                    
                    emb = encoder.get_sentence_embedding(doc)
                    graph_embs.append(emb)
                embs.append(graph_embs)
            io_util.save(f"data/{self.dataset}/tmp/{key}.pkl", np.array(embs))


    def build_dataset(self):
        self.logger.info(f"Build dataset for training")
        metric_embs = io_util.load(f"data/{self.dataset}/tmp/metric.pkl")
        trace_embs = io_util.load(f"data/{self.dataset}/tmp/trace.pkl")
        log_embs = io_util.load(f"data/{self.dataset}/tmp/log.pkl")

        # if trace_embs.shape[1] < 24:
        #     padding = np.zeros((trace_embs.shape[0], 24 - trace_embs.shape[1], trace_embs.shape[2]))
        #     trace_embs = np.concatenate([trace_embs, padding], axis=1)
            

        # if log_embs.shape[1] < 24:
        #     padding = np.zeros((log_embs.shape[0], 24 - log_embs.shape[1], log_embs.shape[2]))
        #     log_embs = np.concatenate([log_embs, padding], axis=1)


        label_types = ['anomaly_type', 'instance']
        label_dict = {label_type: None for label_type in label_types}
        for label_type in label_types:
            label_dict[label_type] = self.get_label(label_type, self.labels)
            print(f"Label type: {label_type}")
            print(f"Label data: {label_dict[label_type]}")

        train_index = np.where(self.labels['data_type'].values == 'train')
        test_index = np.where(self.labels['data_type'].values == 'test')

        print("Shape of metric_embs:", metric_embs.shape)
        print("Shape of trace_embs:", trace_embs.shape)
        print("Shape of log_embs:", log_embs.shape)
        
        train_metric_Xs = metric_embs[train_index]
        train_trace_Xs = trace_embs[train_index]
        train_log_Xs = log_embs[train_index]
        # train_service_labels = label_dict['service'][train_index]
        train_instance_labels = label_dict['instance'][train_index]
        train_type_labels = label_dict['anomaly_type'][train_index]


        test_metric_Xs = metric_embs[test_index]
        test_trace_Xs = trace_embs[test_index]
        test_log_Xs = log_embs[test_index]
        # test_service_labels = label_dict['service'][test_index]
        test_instance_labels = label_dict['instance'][test_index]
        test_type_labels = label_dict['anomaly_type'][test_index]


        # # 调整特征数量为 24
        # if train_metric_Xs.shape[1] < 24:
        #     padding = np.zeros((train_metric_Xs.shape[0], 24 - train_metric_Xs.shape[1], train_metric_Xs.shape[2]))
        #     train_metric_Xs = np.concatenate([train_metric_Xs, padding], axis=1)
        # if test_metric_Xs.shape[1] < 24:
        #     padding = np.zeros((test_metric_Xs.shape[0], 24 - test_metric_Xs.shape[1], test_metric_Xs.shape[2]))
        #     test_metric_Xs = np.concatenate([test_metric_Xs, padding], axis=1)

        

        unique_train_labels, train_counts = np.unique(train_type_labels, return_counts=True)
        print("Train Labels 分布:")
        for label, count in zip(unique_train_labels, train_counts):
            print(f"类别 {label}: {count} 个样本")

        unique_test_labels, test_counts = np.unique(test_type_labels, return_counts=True)
        print("\nTest Labels 分布:")


        for label, count in zip(unique_test_labels, test_counts):
            print(f"类别 {label}: {count} 个样本")


        print("Shape of train_metric_Xs:", train_metric_Xs.shape)
        print("Shape of test_metric_Xs:", test_metric_Xs.shape)
        train_data = MultiModalDataSet(train_metric_Xs, 
                                       train_trace_Xs, 
                                       train_log_Xs, 
                                       train_instance_labels,
                                       train_type_labels, 
                                       self.nodes, 
                                       self.edges)
        test_data = MultiModalDataSet(test_metric_Xs, 
                                      test_trace_Xs, 
                                      test_log_Xs, 
                                      test_instance_labels, 
                                      test_type_labels, 
                                      self.nodes, 
                                      self.edges)

        # graph augmentation
        # if self.args.aug_percent > 0:
        #     # filter samples with lower count
        #     unique_roots, root_counts = np.unique(train_instance_labels, return_counts=True)
        #     unique_types, type_counts = np.unique(train_type_labels, return_counts=True)
        #     aug_root_num = int(self.args.aug_percent * len(unique_roots))
        #     aug_type_num = int(self.args.aug_percent * len(unique_types))
        #     rare_roots = unique_roots[root_counts <= np.sort(root_counts)[:aug_root_num][-1]]
        #     rare_types = unique_types[type_counts <= np.sort(type_counts)[:aug_type_num][-1]]

        #     aug_data = []
        #     for (graph, (root, type)) in train_data:
        #         if root in rare_roots or type in rare_types:
        #             aug_graph1 = aug_drop_node(graph, root, drop_percent=0.2)
        #             aug_graph2 = aug_loss_modality(graph, drop_percent=0.2)
        #             # aug_graph3 = aug_random_walk(graph, root, drop_percent=0.2)
        #             aug_data.append((aug_graph1, (root, type)))
        #             aug_data.append((aug_graph2, (root, type)))
        #             # aug_data.append((aug_graph3, (root, type)))           
        #     train_data.data.extend(aug_data)

        aug_data = []
        for (graph, (root, type)) in train_data:
            aug_graph = aug_drop_node(graph, root, drop_percent=self.args.aug_percent)
            # aug_graph2 = aug_loss_modality(graph, drop_percent=0.2)
            # aug_graph3 = aug_random_walk(graph, root, drop_percent=0.2)
            aug_data.append((aug_graph, (root, type)))
            # aug_data.append((aug_graph3, (root, type)))           
        train_data.data.extend(aug_data)        
        return train_data, test_data

    def get_label(self, label_type, run_table):
        meta_labels = sorted(list(set(list(run_table[label_type]))))
        labels_idx = {label: idx for label, idx in zip(meta_labels, range(len(meta_labels)))}
        labels = np.array(run_table[label_type].apply(lambda label_str: labels_idx[label_str]))
        return labels
