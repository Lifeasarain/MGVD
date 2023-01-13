import os
import fasttext
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def BPE_trainer(source_code_fold, save_path):
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                         min_frequency=2)
    train_files = []
    for root, dirs, files in os.walk(source_code_fold):
        for file in files:
            train_files.append(str(root + '/' + file))
    print(len(train_files))
    print(train_files[0])
    tokenizer.train(train_files, trainer)
    tokenizer.save(save_path+"/tokenizer.json", pretty=True)


def main():
    source_code_fold = "../data/sard/normalize"
    code_represent_path = "../code_embedding"
    BPE_trainer(source_code_fold, code_represent_path)


if __name__ == "__main__":
    main()