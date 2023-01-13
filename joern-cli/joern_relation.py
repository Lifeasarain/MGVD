import os
import re
import numpy as np

"""
After joern processes the c code, use this code;
0 represents AST, 1 represents CFG, 2 represents PDG
"""

def nodeInformation_old(nodeList):
    stack = []
    newNodeList = []
    nodeInf = ""
    for i in range(0, len(nodeList)):
        symbol = nodeList[i]
        if symbol in '(':
            stack.append(1)
            nodeInf = nodeInf + nodeList[i]
            continue
        if len(stack) > 0:
            nodeInf = nodeInf+nodeList[i]
        if symbol in ')':
            stack.pop()
            if len(stack) == 0:
                newNodeList.append(nodeInf)
                nodeInf = ""
                stack = []
            else:
                continue
    return newNodeList


def graphRelation(rootpath,pathdir,tag):
    files = os.listdir(rootpath)
    for file in files:
        nodeRelation = []
        nodeInformation = []
        # path = rootpath + '//' + file
        path = rootpath + '/' + file
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = str(f.read())
                alllist = lines.split("),List(")
                # Determine whether the data is empty
                node1 = []
                node2 = []
                relation = []
                if (alllist[1] != ""):
                    filename = alllist[0].split("/")[-1]
                    # add edge
                    nodeRelation.append(alllist[1])
                    nodeRelation.append(alllist[3])
                    nodeRelation.append(alllist[5])
                    # add node
                    nodeInformation.append(alllist[2])
                    nodeInformation.append(alllist[4])
                    nodeInformation.append(alllist[6])
                    # Regular processing
                    nodeRelation = re.findall(r"\(\d*,\d*,\d*\)", str(nodeRelation))
                    nodeInformation = re.findall(r"\(\d*,.*?\)", str(nodeInformation))
                    # Remove duplicate nodes
                    nodeInformation = list(set(nodeInformation))

                    # Extract the contents of each column into list => batch processing
                    nodeRelation = ' '.join(nodeRelation)
                    b = re.findall('\d+', nodeRelation)
                    for i in range(0, len(b), 3):
                        node1.append(b[i])
                        node2.append(b[i + 1])
                        relation.append(b[i + 2])
                    # relation_matrix = np.vstack([node1, node2, relation]).T

                    nodes = []
                    means = []
                    for i in nodeInformation:
                        node = re.search('\d+(?=,)', i)
                        mean = re.search('(?<=,).*', i)
                        nodes.append(node.group())
                        means.append(mean.group())
                    # feature_matrix = np.vstack([nodes,means]).T

                    # Replace node numbers
                    new_node1 = []
                    new_node2 = []
                    new_nodes = list(range(0, len(nodes)))
                    for x in node1:
                        for i in range(len(nodes)):
                            if x == nodes[i]:
                                # new_node1.append(x.replace(str(x), str(i)))
                                new_node1.append(str(i))
                                break
                    for x in node2:
                        for i in range(len(nodes)):
                            if x == nodes[i]:
                                # new_node2.append(x.replace(str(x), str(i)))
                                new_node2.append(str(i))
                                break

                    # write to file
                    if os.path.exists(pathdir) == False:
                        os.makedirs(pathdir)
                    # with open(pathdir +"\\"+ filename + ".txt", 'w', encoding='utf-8') as f1:
                    with open(pathdir + "/" + filename + ".txt", 'w', encoding='utf-8') as f1:
                        for x, y, z in zip(new_node1, new_node2, relation):
                            # for x, y, z in zip(node1, node2, relation):
                            f1.write('(' + x + ',' + y + ',' + z + ')')
                            # f1.write(x + ',' + y + ',' + z)
                            f1.write("\n")
                        f1.write("-----------------------------------")
                        f1.write("\n")
                        for x, y in zip(new_nodes, means):
                            # for x, y in zip(nodes, means):
                            f1.write('(' + str(x) + ',' + y)
                            # f1.write(str(x) + ',' + y)
                            f1.write(("\n"))
                        f1.write('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                        f1.write("\n")
                        if tag == 'good':
                            f1.write('0')
                        elif tag == 'bad':
                            f1.write('1')
        except:
            # print(path)
            break

def joernGraph_Old(dataPath, outPath, dataTag, sourceCodePath):
    files = os.listdir(dataPath)
    files_num = len(files)
    count = 1
    for file in files:
        count = count + 1
        print("\r", end="")
        print("Process progress: {}%: ".format(count/files_num * 100), end="")

        astNodeRelation = []
        astNodeInformation = []

        cfgNodeRelation = []
        cfgNodeInformation = []

        pdgNodeRelation = []
        pdgNodeInformation = []

        path = dataPath + '/' + file
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = str(f.read())
                alllist = lines.split("),List(")
                # Determine whether the data is empty

                astNode1 = []
                astNode2 = []
                astRelation = []

                cfgNode1 = []
                cfgNode2 = []
                cfgRelation = []

                pdgNode1 = []
                pdgNode2 = []
                pdgRelation = []

                if (alllist[1] != ""):
                    filename = alllist[0].split("/")[-1]
                    # AST add edge
                    astNodeRelation.append(alllist[1])
                    # astNodeRelation = alllist[1]
                    # AST add node
                    # astNodeInformation.append(alllist[2])
                    astNodeInformation = alllist[2]

                    # CFG add edge
                    cfgNodeRelation.append(alllist[3])
                    # CFG add node
                    # cfgNodeInformation.append(alllist[4])
                    cfgNodeInformation = alllist[4]

                    # PDG add edge
                    pdgNodeRelation.append(alllist[5])
                    # PDG add node
                    # pdgNodeInformation.append(alllist[6])
                    pdgNodeInformation = alllist[6][:-3]

                    # Regular processing
                    astNodeRelation = re.findall(r"\(\d*,\d*,\d*\)", str(astNodeRelation))
                    # Contains bug
                    # astNodeInformation = re.findall(r"\(\d*,.*?\)", str(astNodeInformation))
                    astNodeInformation = nodeInormation(astNodeInformation)

                    cfgNodeRelation = re.findall(r"\(\d*,\d*,\d*\)", str(cfgNodeRelation))
                    # cfgNodeInformation = re.findall(r"\(\d*,.*?\)", str(cfgNodeInformation))
                    cfgNodeInformation = nodeInormation(cfgNodeInformation)

                    pdgNodeRelation = re.findall(r"\(\d*,\d*,\d*\)", str(pdgNodeRelation))
                    # pdgNodeInformation = re.findall(r"\(\d*,.*?\)", str(pdgNodeInformation))
                    pdgNodeInformation = nodeInormation(pdgNodeInformation)

                    # Remove duplicate nodes
                    astNodeInformation = list(set(astNodeInformation))

                    cfgNodeInformation = list(set(cfgNodeInformation))

                    pdgNodeInformation = list(set(pdgNodeInformation))

                    #TODO 去掉重复节点

                    # Extract the contents of each column into list => batch processing
                    astNodeRelation = ' '.join(astNodeRelation)
                    astBatch = re.findall('\d+', astNodeRelation)
                    for i in range(0, len(astBatch), 3):
                        astNode1.append(astBatch[i])
                        astNode2.append(astBatch[i + 1])
                        astRelation.append(astBatch[i + 2])

                    cfgNodeRelation = ' '.join(cfgNodeRelation)
                    cfgBatch = re.findall('\d+', cfgNodeRelation)
                    for i in range(0, len(cfgBatch), 3):
                        cfgNode1.append(cfgBatch[i])
                        cfgNode2.append(cfgBatch[i + 1])
                        cfgRelation.append(cfgBatch[i + 2])

                    pdgNodeRelation = ' '.join(pdgNodeRelation)
                    pdgBatch = re.findall('\d+', pdgNodeRelation)
                    for i in range(0, len(pdgBatch), 3):
                        pdgNode1.append(pdgBatch[i])
                        pdgNode2.append(pdgBatch[i + 1])
                        pdgRelation.append(pdgBatch[i + 2])

                    astNodes = []
                    astMeans = []

                    cfgNodes = []
                    cfgMeans = []

                    pdgNodes = []
                    pdgMeans = []

                    for i in astNodeInformation:
                        astNode = re.search('\d+(?=,)', i)
                        astMean = re.search('(?<=,).*', i)
                        astNodes.append(astNode.group())
                        astMeans.append(astMean.group())

                    for i in cfgNodeInformation:
                        node = re.search('\d+(?=,)', i)
                        mean = re.search('(?<=,).*', i)
                        cfgNodes.append(node.group())
                        cfgMeans.append(mean.group())

                    for i in pdgNodeInformation:
                        node = re.search('\d+(?=,)', i)
                        mean = re.search('(?<=,).*', i)
                        pdgNodes.append(node.group())
                        pdgMeans.append(mean.group())

                    # Replace node numbers
                    ast_new_node1 = []
                    ast_new_node2 = []
                    ast_new_nodes = list(range(0, len(astNodes)))
                    for x in astNode1:
                        for i in range(len(astNodes)):
                            if x == astNodes[i]:
                                ast_new_node1.append(str(i))
                                break
                    for x in astNode2:
                        for i in range(len(astNodes)):
                            if x == astNodes[i]:
                                ast_new_node2.append(str(i))
                                break

                    cfg_new_node1 = []
                    cfg_new_node2 = []
                    cfg_new_nodes = list(range(0, len(cfgNodes)))
                    for x in cfgNode1:
                        for i in range(len(cfgNodes)):
                            if x == cfgNodes[i]:
                                cfg_new_node1.append(str(i))
                                break
                    for x in cfgNode2:
                        for i in range(len(cfgNodes)):
                            if x == cfgNodes[i]:
                                cfg_new_node2.append(str(i))
                                break

                    pdg_new_node1 = []
                    pdg_new_node2 = []
                    pdg_new_nodes = list(range(0, len(pdgNodes)))
                    for x in pdgNode1:
                        for i in range(len(pdgNodes)):
                            if x == pdgNodes[i]:
                                pdg_new_node1.append(str(i))
                                break
                    for x in pdgNode2:
                        for i in range(len(pdgNodes)):
                            if x == pdgNodes[i]:
                                pdg_new_node2.append(str(i))
                                break


                    # Source code
                    source = []
                    with open(sourceCodePath+"/"+filename[:16], 'r', encoding='utf-8') as f:
                        source = f.readlines()

                    # write to file
                    if os.path.exists(outPath) == False:
                        os.makedirs(outPath)
                    with open(outPath + "/" + filename + ".txt", 'w', encoding='utf-8') as f1:
                        f1.write("-----Label-----")
                        f1.write("\n")
                        if dataTag == 'clean':
                            f1.write("0")
                        elif dataTag == 'bad':
                            f1.write("1")
                        f1.write("\n")

                        f1.write("-----Code-----")
                        f1.write("\n")
                        f1.writelines(source)
                        f1.write("\n")

                        f1.write("-----AST-----")
                        f1.write("\n")
                        for x, y, z in zip(ast_new_node1, ast_new_node2, astRelation):
                            f1.write(x + ',' + y + ',' + z)
                            f1.write("\n")
                        f1.write("-----AST_Node-----")
                        f1.write("\n")
                        for x, y in zip(ast_new_nodes, astMeans):
                            f1.write(str(x) + '|||' + y)
                            f1.write("\n")

                        f1.write("-----CFG-----")
                        f1.write("\n")
                        for x, y, z in zip(cfg_new_node1, cfg_new_node2, cfgRelation):
                            f1.write(x + ',' + y + ',' + z)
                            f1.write("\n")
                        f1.write("-----CFG_Node-----")
                        f1.write("\n")
                        for x, y in zip(cfg_new_nodes, cfgMeans):
                            f1.write(str(x) + '|||' + y)
                            f1.write("\n")

                        f1.write("-----PDG-----")
                        f1.write("\n")
                        for x, y, z in zip(pdg_new_node1, pdg_new_node2, pdgRelation):
                            f1.write(x + ',' + y + ',' + z)
                            f1.write("\n")
                        f1.write("-----PDG_Node-----")
                        f1.write("\n")
                        for x, y in zip(pdg_new_nodes, pdgMeans):
                            f1.write(str(x) + '|||' + y)
                            f1.write("\n")

                        f1.write("-----End-----")
        except:
            print(path)
            break

if __name__ == '__main__':
    dataPath = r"raw_result/good"
    # dataPath = r"/Users/lifeasarain/Desktop/tmp/SE/Vulnerability/FUNDED_NISL/demo/22sevenEdges/good"
    outPath = r"result/good"
    # outPath = "/Users/lifeasarain/Desktop/tmp/SE/Vulnerability/FUNDED_NISL/demo/33joern/result/good"
    # bad or good
    dataTag = 'bad'
    graphRelation(dataPath, outPath, dataTag)
    print("ooooooooooooooover")




