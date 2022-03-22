# BPE:基于BPE 的汉语tokenization

## 运行环境
python3.8  
没有其他额外环境
## 使用和测试
### train
    >python main.py --mode "train" --train_text "data/train.txt" --save_vocab "data/vocab_save.txt" 
参数说明：

- --mode 执行模式 train/apply
- --train_text 训练数据地址
- --save_vocab 词表存储位置
- --vocab_size 最大词表大小 默认为1000
- --min_frequency 最小复现频率 默认为2
### apply
    >python main.py --mode "apply" --test_text "data/test.txt" --_filename "dvocabata/vocab.txt" --answer_filename "data/answer.txt"
参数说明：

- --mode 执行模式 train/apply
- --test_text 测试数据地址
- --vocab_filename 词表地址
- --answer_filename 答案输出位置

### 样例代码运行
备注：简易版，用于测试是否可用
    
	>python main.py --mode "train" --train_text "data/smalltrain.txt" --save_vocab "data/1000.txt"
	>python main.py --mode "apply" --test_text "data/smalltest.txt" --vocab_filename "data/1000.txt" --answer_filename "answer.txt"

### 实际训练参数
运行时间：train约1h apply约1.5h

	>python main.py --mode "train" --train_text "data/train_BPE.txt" --save_vocab "data/10000.txt --vocab_size 10000"
	>python main.py --mode "apply" --test_text "data/test_BPE.txt" --vocab_filename "data/10000result.txt"


## 模块说明
### train模块
 1. 读入数据  
主要数据：  
vocab：collections.Counter() 用于记录每一句话及其出现频率。以逗号或句号进行分词，对应英文bpe中的单词  
```
def get_vocabulary_from_txt(fobj):
    """读取训练集  
            Args:
                fobj: 文件的输入路径
            Returns:vocab 词频
    """
    vocab = Counter()
    f = open(fobj, "r", encoding="utf-8")
    data = f.readlines()
    for i, line in enumerate(data):
        # 中文分词的思路是按照句号和逗号进行分词
        for word in line.strip('\n').split(","):
            word = word.replace(" ", "")
            word = word.replace("。", "")
            if word:
                vocab[word] += 1
    return vocab
```
 2. 数据预处理  
主要数据：  
sorted_vocab：按出现频率排序的句表  
pairs：所有二元组的出现频率  
indices：所有二元组的出现位置记录，加快更新速度    
```
def get_sorted_vocab(train_data):
    """加载初始数据
        Args:
            train_data: 文件的输入路径
        Returns:sorted_vocab排序后词频dict
         pairs是所有词里二元组的频率dict
         indices是所有二元组在第几个词出现了几次的记录。
    """
    vocab = get_vocabulary_from_txt(train_data)
    v_list = []
    for (x, y) in vocab.items():
        v_list.append((tuple(x[:-1]) + (x[-1] + '</w>',), y))
    v_dict = dict(v_list)
    sorted_vocab = sorted(v_dict.items(), key=lambda x: x[1], reverse=True)
    # 获取二元组的出现频率及其下标位置
    pairs, indices = get_info(sorted_vocab)
    return sorted_vocab, pairs, indices

def get_info(vocab):
    """

    Args:
        vocab: 排好序的降序词频dict

    Returns:pairs是所有词里二元组的频率dict，indices是所有二元组在第几个词出现了几次的记录。

    """

    pairs = defaultdict(int)
    indices = defaultdict(lambda: defaultdict(int))

    for i, (word, count) in enumerate(vocab):
        fir_char = word[0]
        for j in range(1, len(word)):
            pairs[fir_char, word[j]] += count
            indices[fir_char, word[j]][i] += 1
            fir_char = word[j]
    return pairs, indices
```  

 3. 更新词表  
实现思路：  
1.找到本轮出现频率最高的二元组  
2.判断找到的二元组是否满足最低出现频率要求。若不满足，转到9，若满足，转到3  
3.在vocab中找到出现该二元组的句子，将二元组合并为一个元素  
4.更新词汇表。假设需要合并的元素为a|b->ab,若在句子中的出现形式为x|a|b|y转5，为x|a|b转6，为a|b|y转7  
5.在二元组表pairs和indices中减少a和b元素的出现次数，增加x|ab和ab|y元素出现次数，转8    
6.在二元组表pairs和indices中减少a和b元素的出现次数，增加x|ab出现次数，转8    
7.在二元组表pairs和indices中减少a和b元素的出现次数，增加ab|y元素出现次数，转8    
8.存储a|b到输出词表  
9.如果词表大小已达到上限则退出，否则转1  

主函数：  
```
for i in range(vocab_size):
    time_start = time.time()
    if pairs:
        # 找到出现频率最高的pair。
        most_frequent = max(pairs, key=lambda x: (pairs[x], x))

    # 如果最高频次低于设定下限，退出
    now_fre = pairs[most_frequent]
    if now_fre < min_frequency:
        print('已达到下限')
        break

    # 更新vocab
    output = most_frequent[0] + ' ' + most_frequent[1] + '\n'
    fp.write(output)

    # 合并vocab中的被选中二元组
    # 修改pairs和indices
    pairs, indices = megre_pair(most_frequent, vocab, pairs, indices)
```
部分细节实现：  

```
def get_pair_statistics(vocab):
    """
    Args:
        vocab: 排好序的降序词频dict
    Returns:pairs是所有词里二元组的频率dict，indices是所有二元组在第几个词出现了几次的记录。
    """

    pairs = defaultdict(int)
    indices = defaultdict(lambda: defaultdict(int))
    for i, (word, freq) in enumerate(vocab):
        first_char = word[0]
        for char in word[1:]:
            pairs[first_char, char] += freq
            indices[first_char, char][i] += 1
            first_char = char

    return pairs, indices

def megre_pair(pair, vocab, pairs, indices):
    """合并vocab中的pair二元组 并且更新pairs和indices表

    Args:
        pair: 频率最高的二元组
        vocab: 排序好的词频dict
        pairs: 二元组词频表
        indices: 二元组dict，从二元组,二元组来自第几个词->频次

    Returns: 更新后的pairs和indices

    """

    first, second = pair
    new_pair = first + second
    change_record = []

    # 找到出现的句子
    iterator = indices[pair].items()

    for j, freq in iterator:
        # 如果没出现过，就跳过
        if freq == 0:
            continue
        # 从排序好的词频dict中取出这个词
        word, freq = vocab[j]
        merge_word = []
        i = 0
        while i < len(word):
            if word[i] == first:
                if word[i + 1] == second:
                    merge_word.append(''.join(pair))
                    i = i + 2
                else:
                    merge_word.append(word[i])
                    i = i + 1
            else:
                merge_word.append(word[i])
                i = i + 1

        # 更新词汇表
        vocab[j] = (tuple(merge_word), freq)
        # 把更改是第几个词，新词，原来的词，词频添加进change_record里
        change_record.append((j, tuple(merge_word), word, freq))

    # 更新两表
    pairs[pair] = 0
    indices[pair] = defaultdict(int)

    for j, word, old_word, freq in change_record:

        i = 0
        while i < len(old_word) - 1:
            # find first symbol
            if old_word.count(first) < 1:
                break

            if old_word[i] != first or old_word[i + 1] != second:
                # 不匹配，继续搜索
                i = i + 1
                continue
            else:
                if i > 0:
                    # 存在序列a b c 已经合并 bc
                    pairs[old_word[i - 1:i + 1]] -= freq
                    indices[old_word[i - 1:i + 1]][j] -= 1
                if i < len(old_word) - 2:
                    # 存在序列a b c 已经合并 ab
                    # 需要避免出现越界 因此范围再缩小1
                    pairs[old_word[i + 1:i + 3]] -= freq
                    indices[old_word[i + 1:i + 3]][j] -= 1
                i += 2

        i = 0
        while i < len(word):
            if word.count(new_pair) < 1:
                break

            if word[i] != new_pair:
                # 不匹配，继续搜索
                i = i + 1
                continue

            # 假设形式为 a b c d 其中bc被合并 需要增加 a bc 和 bc d
            # 如果形式为 a bc bc 则需要避免重复合并
            if i > 0:
                pairs[word[i - 1:i + 1]] += freq
                indices[word[i - 1:i + 1]][j] += 1
            if i < len(word) - 1 and word[i + 1] != new_pair:
                pairs[word[i:i + 2]] += freq
                indices[word[i:i + 2]][j] += 1
            i += 1

    return pairs, indices

```
### apply模块
 1. 读取此前得到的词表
```
# 读取词表
    vocab = []
    with open(vocab_filename, "r", encoding="utf-8") as fp:
        line = fp.readline()
        line = line[:-1].replace(" ", "")
        vocab.append(line)
        while line:
            line = fp.readline()
            line = line[:-1].replace(" ", "").replace("</w>", "")
            vocab.append(line)
        vocab = [x for x in vocab if len(x) < 3 and is_all_chinese(x) and '的' not in x]
```
 2. 循环合并词表
仅仅按照训练集得到的顺序，不考虑测试集的频率合并  
已经合并的单词打标记，防止被重叠合并  
```
i_vocab = 0
while i_vocab < len(vocab):
     i_data = 0
     while i_data < len(test_data):
          if test_data[i_data:i_data + len(vocab[i_vocab])] == list(vocab[i_vocab]) and vocab[i_vocab] != '':
          		# 添加一个标记，表示这个词已经被划分了，避免删除开销
          		test_data[i_data] += '<m>'
           		i_data += len(vocab[i_vocab])
          else:
                i_data += 1

```

## 文件说明
 - answer.txt ----- test_BPE的测试结果答案
 - 10000result.txt ----- 大小为10000的词表
 - 100000result.txt ----- 大小为100000的词表
 - smallxxx.txt ----- 用于测试的几个小型训练/测试/词表