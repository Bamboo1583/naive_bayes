"""
项目：对侮辱性词和非侮辱性词进行分类
作者：YYB
时间:2019/4/10
版本：1.0
参考文献：《机器学习实战》
    https://cuijiahua.com/blog/2017/11/ml_4_bayes_1.html
"""
from numpy import *


def load_dataSet():
    """
    函数作用：数据集的初始化
    :return: 返回词表向量和类别标签
    """
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return posting_list, class_vec


def create_vocab_list(dataset):
    """
    函数作用：创建一个不重复词的列表
    :param dataset: 给定的所有文档
    :return: 返回一个不重复词的列表
    """
    # 创建一个空集
    vocab_set = set([])
    # 遍历所有文档中的词，并添加不重复的词
    for document in dataset:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set_of_words2vec(vocab_list, input_set):
    """
    函数作用：词汇表到向量的转换函数，词集模型(每个词只能出现一次)
    :param vocab_list: 词汇列表
    :param input_set: 某个文档
    :return: 返回的是文档向量
    """
    # 创建一个其中所含元素都为0的向量
    return_vec = [0] * len(vocab_list)
    # 遍历文档中的所有词汇
    for word in input_set:
        # 如果词汇在文档中，文档向量为1；否则为0(初始化为0)(词汇不在文档中)
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print('the word: %s is not in my Vocabulary!' % word)
    return return_vec


def bag_of_words2vec(vocab_list, input_set):
    """
        函数作用：词汇表到向量的转换函数，词袋模型(每个词可以多次出现)
        :param vocab_list: 词汇列表
        :param input_set: 某个文档
        :return: 返回的是文档向量
        """
    # 创建一个其中所含元素都为0的向量
    return_vec = [0] * len(vocab_list)
    # 遍历文档中的所有词汇
    for word in input_set:
        # 如果词汇在文档中，文档向量为1；否则为0(初始化为0)(词汇不在文档中)
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


def trainNB0(train_matrix, train_category):
    """
    函数作用：朴素贝叶斯分类器训练函数
    :param train_matrix: 文档矩阵，由0,1组成
    :param train_category: 类别标签向量，由0,1组成
    :return: p_abusive表示有侮辱性词汇(1)的文档概率，先验概率P(1)
             p0_vect表示非侮辱类的条件概率数组
             p1_vect表示侮辱类的条件概率数组
    """
    # 训练文档矩阵的长度
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    # 类别标签class=1的概率，即P(1)
    p_abusive = sum(train_category) / float(num_train_docs)
    # 初始化概率
    p0_num = ones(num_words)
    p1_num = ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0
    # 遍历所有的训练文档
    for i in range(num_train_docs):
        if train_category[i] == 1:
            # 某个词汇(class=1)在所有文档中出现的次数
            # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1),...
            p1_num += train_matrix[i]
            # 所有词汇(class=1)在所有文档中出现的次数
            p1_denom += sum(train_matrix[i])
        else:
            # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0),...
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    # 对每个词汇做除法;修改为取对数，避免部分因子非常小
    p0_vect = log(p0_num / p0_denom)
    p1_vect = log(p1_num / p1_denom)
    return p0_vect, p1_vect, p_abusive


def classifyNB(vec2classify, p0_vec, p1_vec, p_class1):
    """
    函数作用：朴素贝叶斯分类函数
    :param vec2classify: 词汇表向量矩阵,由0,1组成
    :param p0_vec: 类别0中的词汇出现概率
    :param p1_vec: 类别1中的词汇出现概率
    :param p_class1: 有侮辱性词汇(1)的文档概率
    :return: p1 > p0，返回1;否则为0
    """
    # 相应元素相乘求和，并求和
    p1 = sum(vec2classify * p1_vec) + log(p_class1)
    p0 = sum(vec2classify * p0_vec) + log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def testingNB():
    """
    函数作用：测试函数，封装所有操作
    :return: 根据文档内容，分类出相应的类别(1\0)
    """
    # 返回词汇文档和类别标签
    listoposts, listclasses = load_dataSet()
    # 不重复的词汇表
    myvocablist = create_vocab_list(listoposts)
    trainMat = []
    for postin in listoposts:
        trainMat.append(set_of_words2vec(myvocablist, postin))
    p0, p1, pa = trainNB0(array(trainMat), array(listclasses))
    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(set_of_words2vec(myvocablist, test_entry))
    print(test_entry, 'classified as: ', classifyNB(this_doc, p0, p1, pa))
    test_entry = ['stupid', 'garbage']
    this_doc = array(set_of_words2vec(myvocablist, test_entry))
    print(test_entry, 'classified as: ', classifyNB(this_doc, p0, p1, pa))


if __name__ == '__main__':
    testingNB()
