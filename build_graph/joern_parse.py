import os
import pandas as pd
import numpy as np
import subprocess
import shutil
from functools import partial
from multiprocessing import Pool

def read_from_file(fold):
    """
    :param path:str
    :return DataFrame
    """
    label_list = []
    function_list = []
    for root, dirs, files in os.walk(fold):
        for file in files:
            dir = root.split("/")[-1]
            if dir == "Vul":
                label = 1
                label_list.append(label)
            elif dir == "No-Vul":
                label = 0
                label_list.append(label)
            with open(os.path.join(root, file), "r") as f:
                function = f.read()
                function_list.append(function)
    data = {"target": label_list, "func": function_list}
    return pd.DataFrame(data)


def drop(data_frame: pd.DataFrame, keys):
    for key in keys:
        del data_frame[key]


def slice_frame(data_frame: pd.DataFrame, size: int):
    data_frame_size = len(data_frame)
    return data_frame.groupby(np.arange(data_frame_size) // size)


def to_files(data_frame: pd.DataFrame, out_path, code_type):
    # path = f"{self.out_path}/{self.dataset_name}/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for idx, row in data_frame.iterrows():
        file_name = f"{idx}.c"
        with open(os.path.join(out_path, file_name), 'w') as f:
            # Change
            try:
                if code_type == "raw":
                    f.write(row.raw)
                elif code_type == "normalize":
                    f.write(row.normalize)
            except:
                print(idx)
                print(row.func_name)
                print(row)


def joern_parse(file, out_fold, file_name):
    joern_path = "/home/qiufangcheng/workspace/joern/joern-cli"
    os.chdir(joern_path)
    out_file = file_name + ".bin"
    out_path = os.path.join(out_fold, out_file)
    os.environ['file'] = str(file)
    os.environ['outPath'] = str(out_path)

    process = subprocess.Popen('sh joern-parse $file --out $outPath',
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                shell=True, close_fds=True)
    output = process.communicate()
    print(output)

    return out_file


def joern_parse_task(raw_file_fold, temp_file_fold, parse_result_fold, code_type):
    # raw = read_from_file(raw_file_fold)
    raw = pd.read_csv(os.path.join(raw_file_fold, "full_data.csv"))
    slices = slice_frame(raw, size=500)
    slices = [(s, slice.apply(lambda x: x)) for s, slice in slices]
    cpg_files = []
    for root, dirs, files in os.walk(parse_result_fold):
        for file in files:
            cpg_files.append(str(file))
    for s, slice in slices:
        to_files(slice, temp_file_fold, code_type)
        cpg_file = joern_parse(temp_file_fold, parse_result_fold, f"{s}_cpg")
        cpg_files.append(cpg_file)
        print(f"Dataset {s} to cpg.")
        shutil.rmtree(temp_file_fold)


def joern_graph_task(parse_result, raw_graph_fold):
    joern_path = "/home/qiufangcheng/workspace/joern/joern-cli"
    script_file = "/home/qiufangcheng/workspace/SGS/joern/njf.sc"
    os.chdir(joern_path)

    params = f"cpgFile={parse_result},outDir={raw_graph_fold}"
    os.environ['params'] = str(params)
    os.environ['script_file'] = str(script_file)

    process = subprocess.Popen('sh joern --script $script_file --params=$params',
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               shell=True, close_fds=True)
    output = process.communicate()
    print(output)
    name = parse_result.split('/')[-1].split('.')[0]
    try:
        workspace = joern_path+"/workspace"
        shutil.rmtree(os.path.join(workspace, name + '.bin'))
    except Exception as e:
        print(e)
        print("remove error")
        return


def process_func_multi(file_fold, save_fold, code_type, flag, temp_file_fold=None):
    if "parse" == flag:
        joern_parse_task(file_fold, temp_file_fold, save_fold, code_type)

    if "graph" == flag:
        input_path_list = []
        for root, dirs, files in os.walk(file_fold):
            for file in files:
                input_path = os.path.join(root, file)
                input_path_list.append(input_path)
                joern_graph_task(input_path, save_fold)


if __name__ == "__main__":
    dataset = "fq"
    # code_type = "raw"
    code_type = "normalize"
    raw_file_fold = "/data/fcq_data/vul_study_project/dataset/" + dataset + "/sgs/"+code_type+"/nc"
    parse_result_fold = "/data/fcq_data/vul_study_project/dataset/" + dataset + "/sgs/"+code_type+"/parse_result"
    raw_graph_fold = "/data/fcq_data/vul_study_project/dataset/" + dataset + "/sgs/"+code_type+"/raw_graphs"
    temp_fold = "/data/fcq_data/vul_study_project/dataset/" + dataset + "/sgs/"+code_type+"/temp"
    # flag = "parse"
    flag = "graph"

    if "parse" == flag:
        process_func_multi(raw_file_fold, parse_result_fold, code_type, flag, temp_fold)

    if "graph" == flag:
        process_func_multi(parse_result_fold, raw_graph_fold, code_type, flag=flag)
