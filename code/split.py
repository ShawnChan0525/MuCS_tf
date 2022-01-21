from concurrent.futures import process
import os
import tqdm
import re
from nltk.stem.porter import PorterStemmer


def preprocess(filename):
    filter_inputs = re.compile("[^0-9|_|^a-z|^A-Z]")
    cnt = 0
    rawdir = "data/raw_data/"
    prepdir = "data/processed_data/"
    source = open(rawdir + filename, "r", encoding="utf-8")
    dest = open(prepdir + filename, "w", encoding="utf-8")
    porter_stemmer = PorterStemmer()
    for line in source.readlines():
        line = filter_inputs.sub(" ", line.strip()).lower()  # 过滤并转化为小写
        prep_line = re.sub(" +", " ", line)  # 将多个空格合并为一个
        sent = " ".join(porter_stemmer.stem(word) for word in prep_line.split(" "))
        dest.write(sent+"\n")


def getTopFW(N):
    total = 0
    fwcnt = {}
    fwmap = {}
    source = open("data/processed_data/NL.txt", "r", encoding="utf-8")
    for line in source.readlines():
        fw_in_line = line.split(" ")[0]
        if fw_in_line not in fwcnt.keys():
            fwcnt[fw_in_line] = 1
        else:
            fwcnt[fw_in_line] += 1
    for i in range(N-1):
        max_fw = max(fwcnt, key=fwcnt.get)
        fwmap[max_fw] = str(i)
        print("%s, %s"%(max_fw, fwcnt[max_fw]))
        total += fwcnt[max_fw]
        del [fwcnt[max_fw]]
    fwmap["other"] = str(N-1)
    print(total)
    return fwmap

def getFW(N):
    fwmap = getTopFW(N)
    source = open("data/processed_data/NL.txt", "r", encoding="utf-8")
    dest = open("data/processed_data/FW_%s.txt"%N,'w',encoding="utf-8")
    for line in source.readlines():
        fw_in_line = line.split(" ")[0]
        if fw_in_line not in fwmap.keys():
            dest.write(str(N-1)) # 意思是other
        else:
            dest.write(str(fwmap[fw_in_line]))
        dest.write('\n')

def find_index(src, key):
    start_pos = 0
    pos = []
    for i in range(src.count(key)):
        if start_pos == 0:
            start_pos = src.index(key)
        else:
            start_pos = src.index(key, start_pos+1)
        pos.append(start_pos)
    return pos


def split():
    file = open('data/raw_data/code.txt', 'r', encoding='utf-8')
    data_dir = "data/processed_data/code"
    codes = file.readlines()
    # for i in range(len(codes)):
    for i in range(5):
        code = codes[i]
        pos = []
        pos.extend(find_index(code, '{'))
        pos.extend(find_index(code, ';'))
        pos.extend(find_index(code, '}'))
        # pos = sum(pos, [])  # 给pos降维
        pos.sort()  # 给pos排序
        with open(os.path.join(data_dir, str(i))+'.txt', 'w')as f:
            f.write(code[0:pos[0]+1]+'\n')
            for j in range(len(pos)-1):
                try:
                    f.write(code[pos[j]+1:pos[j+1]+1]+'\n')
                except:
                    pass

def getFilesWithoutSplit(source_file, dest_dir):
    file = open(source_file, 'r', encoding='utf-8')
    codes = file.readlines()
    for i in range(len(codes)):
    # for i in range(5):
        code = codes[i]
        with open(os.path.join(dest_dir, str(i))+'.txt', 'w')as f:
            f.write(code+'\n\n')

if __name__ == '__main__':
    # preprocess('NL.txt')
    # getFW(40)
    getFilesWithoutSplit("data/processed_data/NL.txt", "data/processed_data/comment")