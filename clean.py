import os
from tqdm import tqdm
import gc
txt_path = []
# 指定目录路径
directory = 'clean/'

# 遍历目录中的所有文件和子目录
for root, dirs, files in os.walk(directory):
    # 遍历文件列表
    for filename in files:
        # 判断文件名是否以.txt结尾
        if filename.endswith('.txt'):
            # 输出文件相对路径
            relative_path = os.path.relpath(os.path.join(root, filename), directory)
            txt_path.append(directory + relative_path)
print(len(txt_path))
text_id = []
vocab = {'[CLS]': 0, '[MASK]': 1, '[SEP]': 2, '[PAD]': 3, '[UNK]': 4}
word2id = {w: i for i, w in enumerate(vocab)}


def convert_id(file_path):
    try:
        with open(file_path, "r", encoding='GB18030', errors='ignore') as f:
            text = f.read()
    except:
        with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
            text = f.read()
    text = text.replace('\n', '[MASK]')
    text_id.append(word2id['[CLS]'])
    for word in tqdm(text, total=len(text)):
        if word not in vocab:
            word2id[word] = len(word2id) - 1
        text_id.append(word2id[word])
    text_id.append(word2id['[SEP]'])


if __name__ == '__main__':
    import os.path as path
    import numpy as np

    for i, file_path in enumerate(txt_path):
        print(i)
        convert_id(file_path)
    del file_path, convert_id, vocab
    gc.collect()
    print(len(text_id))
    filename = 'train.bin'
    fp = np.memmap(filename, dtype='uint16', mode='w+', shape=len(text_id))
    text_id = np.array(text_id, dtype='uint16')
    fp[:] = text_id[:]
    del text_id
    gc.collect()
    fp.flush()
    del fp
    gc.collect()
    with open("model/gau_model/vocab.txt", 'w') as f:
        f.write('\n'.join(list(word2id.keys())))
