import json
import os
import jsonlines
import copy
from tokenizers import Tokenizer
import sent2vec


def split():
    return 0


def filter_nodes(list, num):
    list_new = []
    if len(list) == 0:
        return list_new
    for i in range(len(list)):
        l1 = list[i][0]
        l2 = list[i][1]
        if l1 < num & l2 < num:
            list_new.append(list[i])
    return list_new


def extract_graph_info(file_path):
    problem = False

    source_code = []
    adj_ast = []
    adj_cfg = []
    adj_pdg = []
    ast_nodes = []
    cfg_nodes = []
    pdg_nodes = []
    ast_start_nodes = []
    ast_dest_nodes = []
    pdg_start_nodes = []
    pdg_dest_nodes = []

    label = ""
    label_label = False
    label_code = False
    label_ast = False
    label_ast_node = False
    label_cfg = False
    label_cfg_node = False
    label_pdg = False
    label_pdg_node = False

    with open(file_path, "r") as f:
        data = f.readlines()
        for line in data:

            if line.find("-----Label-----") >= 0:
                label_label = True
                continue
            if label_label:
                label = line.replace('\n', "")
                label_label = False

            if line.find("-----Code-----") >= 0:
                label_code = True
                continue

            if label_code:
                if line.find("-----AST-----") >= 0:
                    label_code = False
                    label_ast = True
                    continue
                else:
                    source_code.append(line)
                    continue

            if label_ast:
                if line.find("-----AST_Node-----") >= 0:
                    label_ast = False
                    label_ast_node = True
                    continue
                else:
                    edges = line.split(",")
                    adj_ast.append((int(edges[0]), int(edges[1])))
                    ast_start_nodes.append(int(edges[0]))
                    ast_dest_nodes.append(int(edges[1]))
                    continue

            if label_ast_node:
                if line.find("-----CFG-----") >= 0:
                    label_ast_node = False
                    label_cfg = True
                    continue
                else:
                    nodes = line.split('\n')[0].split('|||')
                    try:
                        ast_nodes.append({"content": nodes[1], "line_num": nodes[2]})
                    except:
                        # print(file_path)
                        problem = True
                        error_node = {"content": "<ERROR>", "line_num": "<ERROR>"}
                        ast_nodes.append(error_node)
                    continue

            if label_cfg:
                if line.find("-----CFG_Node-----") >= 0:
                    label_cfg = False
                    label_cfg_node = True
                    continue
                else:
                    edges = line.strip().split(",")
                    adj_cfg.append((int(edges[0]), int(edges[1])))
                    continue

            if label_cfg_node:
                if line.find("-----PDG-----") >= 0:
                    label_cfg_node = False
                    label_pdg = True
                    continue
                else:
                    nodes = line.split('\n')[0].split('|||')
                    cfg_nodes.append(nodes[1])
                    continue

            if label_pdg:
                if line.find("-----PDG_Node-----") >= 0:
                    label_pdg = False
                    label_pdg_node = True
                    continue
                else:
                    edges = line.strip().split(",")
                    adj_pdg.append((int(edges[0]), int(edges[1])))
                    pdg_start_nodes.append(int(edges[0]))
                    pdg_dest_nodes.append(int(edges[1]))
                    continue

            if label_pdg_node:
                if line.find("-----End-----") >= 0:
                    label_pdg_node = False
                    break
                else:
                    nodes = line.split('\n')[0].split('|||')
                    try:
                        pdg_nodes.append({"content": nodes[1], "line_num": nodes[2]})
                    except:
                        problem = True
                        error_node = {"content": "<ERROR>", "line_num": "<ERROR>"}
                        pdg_nodes.append(error_node)
                    continue

    return label, source_code, adj_ast, adj_cfg, adj_pdg, \
           ast_nodes, cfg_nodes, pdg_nodes, pdg_start_nodes, pdg_dest_nodes, \
           ast_start_nodes, ast_dest_nodes, problem


def get_line_ast(line_num, ast_nodes, ast_start_nodes, ast_dest_nodes, source_code):
    line_ast_nodes = {}
    line_ast_edges = []
    root_node = {}
    for idx, node in enumerate(ast_nodes):
        content = node["content"]
        if str(line_num) == node["line_num"] or source_code[line_num - 1].find(content) > 0:

            line_ast_nodes[idx] = content

            if content == source_code[line_num-1].strip().replace("\n","")[:-1] or content == source_code[line_num-1].strip().replace("\n",""):
                root_node[idx] = content

    for node_idx in line_ast_nodes:
        for sidx, sn in enumerate(ast_start_nodes):
            if sn == node_idx:
                to_node_idx = ast_dest_nodes[sidx]
                if to_node_idx in line_ast_nodes:
                    line_ast_edges.append((node_idx, to_node_idx))

        for didx, dn in enumerate(ast_dest_nodes):
            if dn == node_idx:
                from_node_idx = ast_start_nodes[didx]
                if from_node_idx in line_ast_nodes:
                    line_ast_edges.append((from_node_idx, node_idx))

    line_ast_edges = list(set(line_ast_edges))
    return line_ast_nodes, line_ast_edges, root_node


def get_func_pdg(pdg_nodes, pdg_start_nodes, pdg_dest_nodes, source_code):
    new_line_nodes = {}
    new_line_edges = []
    for idx, node in enumerate(pdg_nodes):
        from_node_idxs = []
        to_node_idxs = []
        from_line_nums = []
        to_line_nums = []
        content = node["content"]
        line_num = node["line_num"]
        try:
            int(line_num)
        except:
            continue
        source_content = source_code[int(line_num) - 1]
        for sidx, sn in enumerate(pdg_start_nodes):
            if sn == idx:
                to_node_idx = pdg_dest_nodes[sidx]
                to_node_num = pdg_nodes[to_node_idx]["line_num"]
                try:
                    int(to_node_num)
                except:
                    continue
                to_node_idxs.append(to_node_idx)
                to_line_nums.append(to_node_num)

        for didx, dn in enumerate(pdg_dest_nodes):
            if dn == idx:
                from_node_idx = pdg_start_nodes[didx]
                from_node_num = pdg_nodes[from_node_idx]["line_num"]
                try:
                    int(from_node_num)
                except:
                    continue
                from_node_idxs.append(from_node_idx)
                from_line_nums.append(from_node_num)

        to_line_nums = set(to_line_nums)
        from_line_nums = set(from_line_nums)

        if line_num in new_line_nodes:
            if source_content.find(content) >= 0:
                # TODO 加入点边关系
                for tl in to_line_nums:
                    new_line_edges.append((int(line_num), int(tl)))
                for fl in from_line_nums:
                    new_line_edges.append((int(fl), int(line_num)))
            else:
                continue

        if line_num not in new_line_nodes:
            if source_content.find(content) >= 0:
                new_line_nodes[int(line_num)] = source_content.strip().replace("\n", "")
                for tl in to_line_nums:
                    new_line_edges.append((int(line_num), int(tl)))
                for fl in from_line_nums:
                    new_line_edges.append((int(fl), int(line_num)))

    new_line_edges = list(set(new_line_edges))
    return new_line_nodes, new_line_edges


def find_neighbor(nodes, edges, new_line_edges):
    neighbors = []
    neighbors.extend(nodes)
    for node in neighbors:
        for edge in new_line_edges:
            if edge[0] == node and not any(e == [node, edge[1]] for e in edges):
                nodes.append(edge[1])
                edges.append([node, edge[1]])
            if edge[1] == node and not any(e == [edge[0], node] for e in edges):
                nodes.append(edge[0])
                edges.append([edge[0], node])
    nodes = list(set(nodes))
    return nodes, edges


def get_sub_graph(new_line_nodes, new_line_edges, line_num,
                  ast_nodes, ast_start_nodes, ast_dest_nodes,
                  source_code, hop=2):
    step = hop
    pdg_node_list = []
    pdg_edge_list = []
    pdg_node_list.append(line_num)
    while step >= 0:
        pdg_node_list, pdg_edge_list = find_neighbor(pdg_node_list, pdg_edge_list, new_line_edges)
        step -= 1

    line_graph_edges = []
    for nline_num in pdg_node_list:
        line_ast_nodes, line_ast_edges, ast_root_node = get_line_ast(nline_num, ast_nodes, ast_start_nodes,
                                                                     ast_dest_nodes, source_code)
        if not ast_root_node:
            continue
        ast_root_idx = list(ast_root_node.keys())[0]

        for i, edge in enumerate(pdg_edge_list):
            for j, node in enumerate(edge):
                # if node == ast_root_idx:
                if node == nline_num:
                    edge[j] = ast_root_idx
            line_graph_edges.append(edge)

        for edge in line_ast_edges:
            line_graph_edges.append([edge[0], edge[1]])

    new_ast_nodes = []
    edge_nodes1 = []
    edge_nodes2 = []
    idx_mapping = []
    for i, edge in enumerate(line_graph_edges):
        edge_nodes1.append(edge[0])
        edge_nodes2.append(edge[1])

    for orignal_node_idx, node in enumerate(ast_nodes):
        if orignal_node_idx in edge_nodes1 or orignal_node_idx in edge_nodes2:
            new_ast_nodes.append(ast_nodes[orignal_node_idx])
            idx_mapping.append(orignal_node_idx)
    new_idx_edges = []
    for i, edge in enumerate(line_graph_edges):
        new_edge = copy.deepcopy(edge)
        new_edge[0] = idx_mapping.index(edge[0])
        new_edge[1] = idx_mapping.index(edge[1])
        new_idx_edges.append(new_edge)
    return new_ast_nodes, new_idx_edges


def padding_graph(line_list, max_length):
    if len(line_list) == max_length:
        return line_list
    if len(line_list) < max_length:
        blank_nodes_content = [[0 for _ in range(100)]]
        blank_nodes_edge = []
        blank_line_dict = {"node_features": blank_nodes_content, "edges": blank_nodes_edge}
        blank_sequence = [0 for _ in range(100)]
        blank_line_info = {"sequence": blank_sequence, "raw_graph": blank_line_dict, "normalize_graph": blank_line_dict}
        num_added_line = max_length - len(line_list)
        for i in range(num_added_line):
            line_list.append(blank_line_info)

        return line_list
    else:
        return line_list[:max_length]


def get_channel_graph(represent_path, graph_root_path, data_save_path):
    problem_files = []

    tokenizer_raw_path = represent_path + "/raw_tokenizer_sard.json"
    tokenizer_normalize_path = represent_path + "/normalize_tokenizer_sard.json"

    tokenizer_raw = Tokenizer.from_file(tokenizer_raw_path)
    tokenizer_normalize = Tokenizer.from_file(tokenizer_normalize_path)
    sent2vec_model_raw = sent2vec.Sent2vecModel()
    sent2vec_model_normalize = sent2vec.Sent2vecModel()
    sent2vec_model_raw.load_model(represent_path + "/raw_model.bin")
    sent2vec_model_normalize.load_model(represent_path+"/normalize_model.bin")
    idx =0
    normalize_code_fold = os.path.join(graph_root_path, "normalize/mid_graphs")
    raw_code_fold = os.path.join(graph_root_path, "raw/mid_graphs")
    normalize_files = os.listdir(normalize_code_fold)
    raw_files = os.listdir(raw_code_fold)

    # rename mid_graphs
    for root, dirs, files in os.walk(normalize_code_fold):
        for file in files:
            os.rename(os.path.join(root, file), os.path.join(root, file.split(".c-")[0]+".txt"))
    for root, dirs, files in os.walk(raw_code_fold):
        for file in files:
            os.rename(os.path.join(root, file), os.path.join(root, file.split(".c-")[0]+".txt"))

    print('normalize files: %d' % (len(normalize_files)))
    print('raw files: %d' % (len(raw_files)))
    files_num = len(normalize_files)
    count = 0


    for normalize in normalize_files:
        function_idx = normalize.split(".txt")[0]
        count = count + 1
        print("\r", end="")
        print("Process progress: {}%: ".format(count / files_num * 100), end="")

        normalize_mid_graph = os.path.join(normalize_code_fold)
        raw_mid_graph = os.path.join(raw_code_fold)

        processed_function = []
        try:
            label, normalize_source_code, normalize_adj_ast, normalize_adj_cfg, \
            normalize_adj_pdg, normalize_ast_nodes, normalize_cfg_nodes, \
            normalize_pdg_nodes, normalize_pdg_start_nodes, normalize_pdg_dest_nodes, \
            normalize_ast_start_nodes, normalize_ast_dest_nodes, normalize_problem = extract_graph_info(os.path.join(normalize_mid_graph, normalize))
        except:
            continue
        if normalize_problem:
            problem_files.append(normalize)
        normalize_new_line_nodes, normalize_new_line_edges = get_func_pdg(normalize_pdg_nodes,
                                                                          normalize_pdg_start_nodes,
                                                                          normalize_pdg_dest_nodes,
                                                                          normalize_source_code)

        normalize_new_line_nums = sorted(normalize_new_line_nodes)

        # 处理raw code
        try:
            raw_label, raw_source_code, raw_adj_ast, raw_adj_cfg, \
            raw_adj_pdg, raw_ast_nodes, raw_cfg_nodes, \
            raw_pdg_nodes, raw_pdg_start_nodes, raw_pdg_dest_nodes, \
            raw_ast_start_nodes, raw_ast_dest_nodes, raw_problem = extract_graph_info(os.path.join(raw_mid_graph, normalize))
        except:
            print()
        if raw_problem:
            problem_files.append(normalize)
        raw_new_line_nodes, raw_new_line_edges = get_func_pdg(raw_pdg_nodes,
                                                              raw_pdg_start_nodes,
                                                              raw_pdg_dest_nodes,
                                                              raw_source_code)

        raw_new_line_nums = sorted(raw_new_line_nodes)

        if raw_new_line_nums == normalize_new_line_nums:
            # normalize code
            for line_num in normalize_new_line_nums:
                normalize_line_graph_nodes, normalize_line_graph_edges = get_sub_graph(normalize_new_line_nodes,
                                                                                   normalize_new_line_edges,
                                                                                   line_num,
                                                                                   normalize_ast_nodes,
                                                                                   normalize_ast_start_nodes,
                                                                                   normalize_ast_dest_nodes,
                                                                                   normalize_source_code,
                                                                                   hop=2)
                normalize_line_nodes_index = []
                normalize_line_nodes_content = []
                for index, node in enumerate(normalize_line_graph_nodes):
                    if node not in normalize_line_nodes_index:
                        normalize_line_nodes_index.append(node)
                        normalize_content_tokenized = tokenizer_normalize.encode(node["content"]).tokens
                        # print(" ".join(content_tokenized))
                        vector = sent2vec_model_normalize.embed_sentence(" ".join(normalize_content_tokenized))[0]
                        normalize_line_nodes_content.append(vector.tolist())
                normalize_line_dict = {"normalize_node_features": normalize_line_nodes_content, "normalize_edges": normalize_line_graph_edges}

                # raw code
                raw_line_graph_nodes, raw_line_graph_edges = get_sub_graph(raw_new_line_nodes,
                                                                       raw_new_line_edges,
                                                                       line_num,
                                                                       raw_ast_nodes,
                                                                       raw_ast_start_nodes,
                                                                       raw_ast_dest_nodes,
                                                                       raw_source_code,
                                                                       hop=2)
                raw_line_nodes_index = []
                raw_line_nodes_content = []
                for index, node in enumerate(raw_line_graph_nodes):
                    if node not in raw_line_nodes_index:
                        raw_line_nodes_index.append(node)
                        raw_content_tokenized = tokenizer_normalize.encode(node["content"]).tokens
                        vector = sent2vec_model_normalize.embed_sentence(" ".join(raw_content_tokenized))[0]
                        raw_line_nodes_content.append(vector.tolist())
                raw_line_dict = {"raw_node_features": raw_line_nodes_content, "raw_edges": raw_line_graph_edges}

                # normalize line graph
                # raw line graph
                # sequence
                line_content = raw_source_code[line_num-1]
                line_content_tokenized = tokenizer_raw.encode(line_content).tokens
                sequence = sent2vec_model_raw.embed_sentence(" ".join(line_content_tokenized))[0].tolist()

                line_info = {"sequence": sequence, "raw_graph": raw_line_dict, "abstract_graph": normalize_line_dict}
                processed_function.append(line_info)

            # If len(line_list) < 128 padding; len(line_list) > 128 cut
            if len(processed_function) > 0:
                channel_function = padding_graph(processed_function, max_length=128)
                save_path = os.path.join(data_save_path, "function_embedded.jsonl")

                with jsonlines.open(save_path, mode='a') as writer:
                    writer.write(
                        json.dumps({"function_idx": function_idx, "label": label, "channel_function": channel_function})
                    )
                idx += 1

if __name__ == '__main__':
    code_embedding_path = "/home/qiufangcheng/workspace/SGS/code_represent"
    data_save_path = "/data/fcq_data/vul_study_project/dataset/sard/sgs/jsonl_data"
    graph_root_path = "/data/fcq_data/vul_study_project/dataset/sard/sgs"
    get_channel_graph(code_embedding_path, graph_root_path, data_save_path)

