import os
import shutil

def mv_file(img, num):
    list_ = os.listdir(img)
    if num > len(list_):
        print('长度需小于：', len(list_))
        exit()
    num_file = int(len(list_)/num) + 1
    cnt = 0
    for n in range(1,num_file+1): # 创建文件夹
        new_file = os.path.join(img + '_' + str(n))
        if os.path.exists(new_file+'_'+str(cnt)):
            print('该路径已存在，请解决冲突', new_file)
            exit()
        print('创建文件夹：', new_file)
        os.mkdir(new_file)
        list_n = list_[num*cnt:num*(cnt+1)]
        for m in list_n:
            old_path = os.path.join(img, m)
            new_path = os.path.join(new_file, m)
            shutil.copy(old_path, new_path)
        cnt = cnt + 1
    print('============task OK!===========')
if __name__ == "__main__":
    mv_file('/home/qiufangcheng/workspace/FUNDED_NISL/Edge_processing/joern-cli/data/clean', 1000) # 操作目录，单文件夹存放数量