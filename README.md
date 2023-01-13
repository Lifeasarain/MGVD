# Vulnerability Detection via Multiple-Graph-Based Code Representation
During software development and maintenance, vulnerability detection is an essential part of software quality assurance. Even though many program-analysis-based and machine-learning based approaches have been proposed to automatically detect vulnerabilities, they rely on explicit rules or patterns defined by security experts and suffer from either high false positives or high false negatives. Recently, an increasing number of studies leverage deep learning techniques, especially Graph Neural Network (GNN), to detect vulnerability. These approaches leverage program analysis to represent the program semantics as graphs and perform graph analysis to detect vulnerabilities. However, they suffer from two main problems: (i) They mainly convert source code into a single representation, e.g., Program Dependency Graph, which limits their ability to capture multi-dimensional vulnerability-related features. (ii) They convert each function into a single graph where each node usually denotes a statement, making it difficult to effectively capture fine-grained code features and extract vulnerability-related features from long functions. Because before being processed by GNN, each node in such graph is embedded into one feature vector and no fine-grained information is explicitly kept, and the graph constructed for a long function is large, complex, and hard for GNN to analyze. To tackle these problems, in this paper, we propose a novel vulnerability detection approach, named MGVD (MULTIPLE-GRAPH-BASED VULNERABILITY DETECTION), to detect vulnerabilities at the function level. To enrich the representations of source code, MGVD uses three different ways to represent each function and encode such representations into a 3-channel feature matrix. The feature matrix contains the structural information and the semantic information of the function. To overcome the second problem, MGVD constructs each graph representation of the input function using multiple different graphs instead of a single large graph. Each graph focuses on one statement in the function and its nodes denote the related statements and their fine-grained code elements. Finally, MGVD leverages CNN to identify whether this function is vulnerable based on such feature matrix. We conduct experiments on 3 vulnerability datasets with a total of 30,441 vulnerable functions and 134,428 non-vulnerable functions. The experimental results show that our method outperforms the state-of-the-art by 10.98% - 15.96% in terms of F1-score.



## Process Data 
In this step we normalize the source code to get raw function, and abstact variables and function names to get abstract function
	process_data/normalize.py
	line 233 set raw_csv as original data csv
	line 234 set store_path as processed data csv


## Extract AST and PDG
In this step, we use Joern to extract AST and PDG of raw functrion and abstract function
### 1. parse source code
	build_graph/joern_parse.py
	line 134 set the dataset as sard/fq/bigvul
	line 136 set the code_type as raw/normalize
	line 141 set the flag as parse

### 2. extract graph
	build_graph/joern_parse.py
	line 141 set the flag as graph

### 3. process graph
	build_graph/joern_graph 
	line 256 set the dataset as sard/fq/bigvul
	line 257 set set the code_type as raw/normalize


## Encode Source Code
In this step, we will train a Sent2Vec model to encode source code
### 1. code_embedding/tokenize.py
	build a BPE tokenizer and prepare text for Sent2Vec
	
### 2. code_embedding/sent2vec
	follow the readme.md to train a Sent2Vec model


## Build Feature Matrix
In this step, we will build the feature matrix of each function
	build_graph/build_feature_matrix.py


## Generate Dataset
In this step, we will build dataset for training, test and validation
### 1. We split the raw dataset to train, test and validation set
	dataset_util/split_data.py
	
	2. Train model
	main/main.py
	set flag as train

	3. Test model 
	main/main.py
	set flag as test













