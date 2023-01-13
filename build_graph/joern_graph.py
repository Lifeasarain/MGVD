import os
import re
import pandas as pd


def nodeInformation(nodeList):
    # new_inf_list = re.split('[(]\d{4,20},', nodeList)
    new_inf_list = re.split('[(]\d+,', nodeList)
    # num_list = re.findall('[(]\d{4,20},', nodeList)
    num_list = re.findall('[(]\d+,', nodeList)
    new_node_list = []
    for i in range(0, len(num_list)):
        if i == len(num_list) - 1:
            new_node = num_list[i] + new_inf_list[i + 1]
        else:
            new_node = num_list[i] + new_inf_list[i + 1][:-2]
        new_node_list.append(new_node)

    return new_node_list


def joernGraph(dataPath, outPath, source_csv, code_type):
    files = os.listdir(dataPath)
    files_num = len(files)
    print(files_num)
    count = 0
    source_data = pd.read_csv(source_csv)

    for file in files:
        count = count + 1
        print("\r", end="")
        print("Process progress: {}%: ".format(count / files_num * 100), end="")

        ast_node_relation = []
        cfg_node_relation = []
        pdg_node_relation = []

        path = dataPath + '/' + file
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = str(f.read())
                alllist = lines.split("),List(")
                # Determine whether the data is empty
                ast_node1 = []
                ast_node2 = []
                ast_relation = []
                cfg_node1 = []
                cfg_node2 = []
                cfg_relation = []
                pdg_node1 = []
                pdg_node2 = []
                pdg_realtion = []

                # if (alllist[1] != "")
                if (alllist[1] != "" and alllist[0].find(".c-<global>") < 0 and alllist[0].find(".c") > 0 and alllist[
                    0].find(".c-VAR") < 0):
                    filename = alllist[0].split("/")[-1]
                    if (filename.find(".c") < 0):
                        filename = alllist[0].split("/")[-2] + filename
                        print(filename)

                    # AST add edge
                    ast_node_relation.append(alllist[1])
                    # AST add node
                    ast_node_info = alllist[2]
                    # CFG add edge
                    cfg_node_relation.append(alllist[3])
                    # CFG add node
                    cfg_node_info = alllist[4]
                    # PDG add edge
                    pdg_node_relation.append(alllist[5])
                    # PDG add node
                    pdg_node_info = alllist[6][:-3]

                    # Regular processing
                    ast_node_relation = re.findall(r"\(\d*,\d*,\d*\)", str(ast_node_relation))
                    ast_node_info = nodeInformation(ast_node_info)
                    cfg_node_relation = re.findall(r"\(\d*,\d*,\d*\)", str(cfg_node_relation))
                    cfg_node_info = nodeInformation(cfg_node_info)
                    pdg_node_relation = re.findall(r"\(\d*,\d*,\d*\)", str(pdg_node_relation))
                    pdg_node_info = nodeInformation(pdg_node_info)

                    # Extract the contents of each column into list => batch processing
                    ast_node_relation = ' '.join(ast_node_relation)
                    ast_batch = re.findall('\d+', ast_node_relation)
                    for i in range(0, len(ast_batch), 3):
                        ast_node1.append(ast_batch[i])
                        ast_node2.append(ast_batch[i + 1])
                        ast_relation.append(ast_batch[i + 2])

                    cfg_node_relation = ' '.join(cfg_node_relation)
                    cfgBatch = re.findall('\d+', cfg_node_relation)
                    for i in range(0, len(cfgBatch), 3):
                        cfg_node1.append(cfgBatch[i])
                        cfg_node2.append(cfgBatch[i + 1])
                        cfg_relation.append(cfgBatch[i + 2])

                    pdg_node_relation = ' '.join(pdg_node_relation)
                    pdgBatch = re.findall('\d+', pdg_node_relation)
                    for i in range(0, len(pdgBatch), 3):
                        pdg_node1.append(pdgBatch[i])
                        pdg_node2.append(pdgBatch[i + 1])
                        pdg_realtion.append(pdgBatch[i + 2])

                    ast_nodes = []
                    ast_means = []
                    ast_lines = []
                    cfg_nodes = []
                    cfg_means = []
                    cfg_lines = []
                    pdg_nodes = []
                    pdg_means = []
                    pdg_lines = []

                    for i in range(0, len(ast_node_info)):
                        ast_node = re.match('[(]\d+', ast_node_info[i]).group()[1:]
                        ast_ml = ast_node_info[i][len(ast_node) + 2:-1]
                        ast_line = ast_ml.split(",")[-1]
                        ast_mean = ast_ml[:len(ast_ml) - len(ast_line) - 1]
                        ast_nodes.append(ast_node)
                        ast_means.append(ast_mean)
                        ast_lines.append(ast_line)

                    for i in range(0, len(cfg_node_info)):
                        cfg_node = re.match('[(]\d+', cfg_node_info[i]).group()[1:]
                        cfg_ml = cfg_node_info[i][len(cfg_node) + 2:-1]
                        cfg_line = cfg_ml.split(",")[-1]
                        cfg_mean = cfg_ml[:len(cfg_ml) - len(cfg_line) - 1]
                        cfg_nodes.append(cfg_node)
                        cfg_means.append(cfg_mean)
                        cfg_lines.append(cfg_line)

                    for i in range(0, len(pdg_node_info)):
                        pdg_node = re.match('[(]\d+', pdg_node_info[i]).group()[1:]
                        # pdg_mean = pdg_node_info[i][len(ast_node) + 2:-1]
                        pdg_ml = pdg_node_info[i][len(pdg_node) + 2:-1]
                        pdg_line = pdg_ml.split(",")[-1]
                        pdg_mean = pdg_ml[:len(pdg_ml) - len(pdg_line) - 1]
                        pdg_nodes.append(pdg_node)
                        pdg_means.append(pdg_mean)
                        pdg_lines.append(pdg_line)

                    # Replace node numbers
                    ast_new_node1 = []
                    ast_new_node2 = []
                    ast_new_nodes = list(range(0, len(ast_nodes)))
                    for x in ast_node1:
                        for i in range(len(ast_nodes)):
                            if x == ast_nodes[i]:
                                ast_new_node1.append(str(i))
                                break
                    for x in ast_node2:
                        for i in range(len(ast_nodes)):
                            if x == ast_nodes[i]:
                                ast_new_node2.append(str(i))
                                break

                    cfg_new_node1 = []
                    cfg_new_node2 = []
                    cfg_new_nodes = list(range(0, len(cfg_nodes)))
                    for x in cfg_node1:
                        for i in range(len(cfg_nodes)):
                            if x == cfg_nodes[i]:
                                cfg_new_node1.append(str(i))
                                break
                    for x in cfg_node2:
                        for i in range(len(cfg_nodes)):
                            if x == cfg_nodes[i]:
                                cfg_new_node2.append(str(i))
                                break

                    pdg_new_node1 = []
                    pdg_new_node2 = []
                    pdg_new_nodes = list(range(0, len(pdg_nodes)))
                    for x in pdg_node1:
                        for i in range(len(pdg_nodes)):
                            if x == pdg_nodes[i]:
                                pdg_new_node1.append(str(i))
                                break
                    for x in pdg_node2:
                        for i in range(len(pdg_nodes)):
                            if x == pdg_nodes[i]:
                                pdg_new_node2.append(str(i))
                                break

                    idx = int(filename.split(".c")[0])

                    dataTag = str(source_data.loc[idx].bug)

                    # code_type == raw
                    code = source_data.loc[idx].raw
                    if code_type == "normalize":
                        code = source_data.loc[idx].normalize

                    # write to file
                    if os.path.exists(outPath) == False:
                        os.makedirs(outPath)
                    with open(outPath + "/" + filename + ".txt", 'w', encoding='utf-8') as f2:
                        f2.write("-----Label-----")
                        f2.write("\n")
                        # if dataTag == 'clean':
                        #     f2.write("0")
                        # elif dataTag == 'bad':
                        #     f2.write("1")

                        f2.write(dataTag)

                        f2.write("\n")

                        f2.write("-----Code-----")
                        f2.write("\n")
                        f2.write(code)
                        f2.write("\n")

                        f2.write("-----AST-----")
                        f2.write("\n")
                        for x, y in zip(ast_new_node1, ast_new_node2):
                            f2.write(x + ',' + y)
                            f2.write("\n")
                        f2.write("-----AST_Node-----")
                        f2.write("\n")
                        for x, y, z in zip(ast_new_nodes, ast_means, ast_lines):
                            f2.write(str(x) + '|||' + y.replace('\n', '') + '|||' + str(z))
                            f2.write("\n")

                        f2.write("-----CFG-----")
                        f2.write("\n")
                        for x, y in zip(cfg_new_node1, cfg_new_node2):
                            f2.write(x + ',' + y)
                            f2.write("\n")
                        f2.write("-----CFG_Node-----")
                        f2.write("\n")
                        for x, y in zip(cfg_new_nodes, cfg_means):
                            f2.write(str(x) + '|||' + y.replace('\n', ''))
                            f2.write("\n")

                        f2.write("-----PDG-----")
                        f2.write("\n")
                        for x, y in zip(pdg_new_node1, pdg_new_node2):
                            f2.write(x + ',' + y)
                            f2.write("\n")
                        f2.write("-----PDG_Node-----")
                        f2.write("\n")
                        for x, y, z in zip(pdg_new_nodes, pdg_means, pdg_lines):
                            f2.write(str(x) + '|||' + y.replace('\n', '') + '|||' + str(z))
                            f2.write("\n")

                        f2.write("-----End-----")
        except Exception as e:
            print(path)
            print(e)
            print(filename)


if __name__ == '__main__':
    dataset = "fq"
    code_type = "raw"
    data_path = "/data/fcq_data/vul_study_project/dataset/" + dataset + "/sgs/"+code_type+"/raw_graphs"
    out_path = "/data/fcq_data/vul_study_project/dataset/" + dataset + "/sgs/"+code_type+"/mid_graphs"
    source_csv_path = "/data/fcq_data/vul_study_project/dataset/" + dataset + "/sgs/"+code_type+"/nc/full_data.csv"
    joernGraph(data_path, out_path, source_csv_path, code_type)

    print("finish")