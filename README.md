1 运行 split.py，将 raw_data 分割为tokens的形式
2 运行 create_data_corpus.py，获取 tokens、 comment_tokens、 SCP_labels、 AWP_labels、 vocabs等数据
3 运行 create_data_instances.py，得到预训练所需的实例，实例由四个 features 组成，分别是 input_ids、 SCP_labels、 AWP_labels、 code_size
4 运行 run_pretraining.py，获得预训练的模型