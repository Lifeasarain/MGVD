import os
import numpy as np
import json
import jsonlines
import gzip
import random

def mkdir(path):
    import os
    path = path.strip()
    # path = path.rstrip("\\")

    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' is ready now')
        return True
    else:
        print(path + ' already exists')
        return False

def Split(train, valid, test, data_fold, save_path):
    print("\nbegin to split files...")
    tem_item = []
    # mkdir(os.path.split(path)[0] + '/' + 'tem_' + os.path.split(path)[1] + '/cfg')
    emb_path = data_fold + "/function_embedded.jsonl"
    path_tem = data_fold + "/tem.jsonl"
    num_list = [x for x in range(0, 3)]

    randomtem = np.arange(len(tem_item))
    clean_tem = []
    vul_tem = []

    count = 0
    with open(emb_path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            # if count in test_num:
            with jsonlines.open(path_tem, mode='a') as writer:
                writer.write(item)
            print("\r", end="")
            print("Process progress: {}%: ".format(count / 3 * 100), end="")
            count += 1

    for step, i in enumerate(randomtem):
        item = tem_item[i]
        if item["label"] == "0":
            clean_tem.append(item)
        else:
            vul_tem.append(item)

    clean_length = len(clean_tem)
    vul_length = len(vul_tem)

    pos_tem = clean_tem

    # Oversampling train set
    vul_tem_train = vul_tem[:int(vul_length * train)]
    clean_tem_train = clean_tem[:int(clean_length * train)]
    for i in range(0, int(clean_length * train) - int(vul_length * train)):
        random_index = random.randrange(int(clean_length * train))
        item = vul_tem_train[random_index]
        vul_tem_train.append(item)
    tem_train = vul_tem_train + clean_tem_train
    np.random.shuffle(tem_train)



    tem_valid = (vul_tem[int(vul_length * train):int(vul_length * (train + valid))]) + (
        clean_tem[int(clean_length * train):int(clean_length * (train + valid))])
    np.random.shuffle(tem_valid)

    tem_test = (vul_tem[int(vul_length * (train + valid)):]) + (pos_tem[int(vul_length * (train + valid)):])
    np.random.shuffle(tem_test)

    for i in tem_train:
        with jsonlines.open(path_tem, mode='a') as writer:
            writer.write(i)
    f_in = open(path_tem, 'rb')
    f_out = gzip.open(save_path + '/train.jsonl.gz', 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()
    os.remove(path_tem)
    print("train ready")

    for i in tem_test:
        with jsonlines.open(path_tem, mode='a') as writer:
            writer.write(i)
    f_in = open(path_tem, 'rb')
    f_out = gzip.open(save_path + '/test.jsonl.gz', 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()
    os.remove(path_tem)
    print("test ready")

    for i in tem_valid:
        with jsonlines.open(path_tem, mode='a') as writer:
            writer.write(i)
    f_in = open(path_tem, 'rb')
    f_out = gzip.open(save_path + '/valid.jsonl.gz', 'wb')
    f_out.writelines(f_in)
    f_out.close()
    f_in.close()
    os.remove(path_tem)
    print("valid ready")

    print("data ready")

if __name__ == "__main__":
    dataset = "sard"
    json_path = "/data/fcq_data/vul_study_project/dataset/"+dataset+"/sgs/jsonl"
    graphdataset_path = "/data/fcq_data/vul_study_project/dataset/"+dataset+"/sgs/graph_dataset"
    Split(train=0.8, valid=0.1, test=0.1, data_fold=json_path, save_path=graphdataset_path)