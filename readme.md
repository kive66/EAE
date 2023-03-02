参数继承Config类，根据不同数据集设置不同参数
数据预处理根据不同数据集不同，处理后使用bert_data_processer创建data_loader
训练继承TrainerBasic类，覆写log记录方法
model中事件分类使用bert cls+Linear（

summar(bert or token) + for(pre_answer+role_name(编码？)) + (vocabs or for(vocab))
                                                                   1 0 1 1 0  or    1
                                                                   
        1、bert( summar + role + entities)
        2、bert(summar) + bert(role + entities)
        all pos tokens
        idxs
        
        batch *(512+ vocab_len+ role) softmax