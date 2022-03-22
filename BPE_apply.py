import time


def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def bpe_apply(test_text, vocab_filename, outfile="answer.txt"):
    time_begin = time.time()
    vocab = []
    with open(vocab_filename, "r", encoding="utf-8") as fp:
        line = fp.readline()
        line = line[:-1].replace(" ", "")
        vocab.append(line)
        while line:
            line = fp.readline()
            line = line[:-1].replace(" ", "").replace("</w>", "")
            # line = line[:-1].replace(" ", "")
            vocab.append(line)
        vocab = [x for x in vocab if len(x) < 3 and is_all_chinese(x) and '的' not in x]
        # vocab = sorted(vocab, key=lambda i: len(i), reverse=True)

    time_end = time.time()
    print('完成读取，用时 = %fs' % (time_end - time_begin))
    time_begin = time.time()

    with open(test_text, 'r', encoding="utf-8") as fp1, open(outfile, "w", encoding="utf-8") as fp2:
        # text = ''
        num_sen = 0
        test_data = []
        for line in fp1:
            for word in line.split(' '):
                word = word.strip(' ')
                test_data.append(word)

        print("size of vocab ", len(vocab))

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

            i_vocab += 1
            if i_vocab % 100 == 0:
                print("已完成", i_vocab)
                time_end = time.time()
                print('用时 = %fs' % (time_end - time_begin))
        print("开始写入")

        i_result = 0
        while i_result < len(test_data):
            if '<m>' in test_data[i_result]:
                fp2.write(test_data[i_result].replace('<m>', ''))
            else:
                if test_data[i_result].isdigit() or test_data[i_result] == '-' or test_data[i_result] == '.':
                    if test_data[i_result + 1].isdigit() or test_data[i_result + 1] == '.':
                        fp2.write(test_data[i_result])
                    else:
                        fp2.write(test_data[i_result] + ' ')
                else:
                    fp2.write(test_data[i_result] + ' ')
            i_result += 1


if __name__ == '__main__':
    bpe_apply("test_BPE.txt", "10000.txt")
