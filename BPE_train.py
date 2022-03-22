from utils import megre_pair, get_sorted_vocab


def bpe_train(train_text="data/smalltrain.txt", save_vocab="data/10000.txt", vocab_size=10000, min_frequency=3):
    """训练词表并存储.

    Args:
        train_text: 文件的输入路径
        save_vocab: 生成的输出路径
        vocab_size:词表的大小
        min_frequency: 最小出现词频
    """
    import time
    time_start = time.time()

    # 导入数据和初始表格
    vocab, pairs, indices = get_sorted_vocab(train_text)

    time_end = time.time()
    print('导入数据完成 用时 = %fs' % (time_end - time_start))

    with open(save_vocab, "w", encoding="utf-8") as fp:
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


            if i % 100 == 0:
                time_end = time.time()
                print("已完成： ", i)
                print("当前频次： ", now_fre)
                print('用时 = %fs' % (time_end - time_start))


if __name__ == '__main__':
    # learn_bpe("train_BPE.txt")
    bpe_train()
