import sys
import re
from collections import defaultdict, Counter


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


def get_vocabulary_from_txt(fobj):
    """读取训练集
            Args:
                fobj: 文件的输入路径
            Returns:sorted_vocab 词频dict
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
