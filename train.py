# coding:utf-8

import datetime
import os

import tensorflow as tf
import typing


# 编码
class Encoder(tf.keras.Model):
    # 设置参数
    def __init__(self, vocab_size: int, embedding_dim: int, enc_units: int) -> None:
        """
        编码器的初始函数
        :param vocab_size: 词库大小
        :param embedding_dim: 词向量维度
        :param enc_units: LSTM层的神经元数量
        """
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        # 词嵌入层
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # LSTM层，GRU是简单的LSTM层
        self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True)

    def call(self, x: tf.Tensor, **kwargs) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        """
        前向传播的计算结果
        :param x: 输入的文本
        :param kwargs: 关键字参数接受额外的内容
        :return:
        """
        x = self.embedding(x)
        output, state = self.gru(x)
        return output, state  # 输出预测结果和当前状态


# 注意力机制
class BahdanauAttention(tf.keras.Model):
    # 设置参数
    def __init__(self, units: int) -> None:
        """
        注意力机制初始函数
        :param units: 神经元数据量
        """
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)  # 全连接层
        self.W2 = tf.keras.layers.Dense(units)  # 全连接层
        self.V = tf.keras.layers.Dense(1)  # 输出层

    def call(self, query: tf.Tensor, values: tf.Tensor, **kwargs) -> typing.Tuple[tf.Tensor, tf.Tensor]:
        """
        设置注意力的计算方式
        :param query: 上一层输出的特征值
        :param values: 上一层输出的计算结果
        :param kwargs: 关键字参数接受额外的内容
        :return: 输出特征向量和权重
        """
        # 维度增加一维
        hidden_with_time_axis = tf.expand_dims(query, 1)
        # 构造计算方法
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        # 计算权重
        attention_weights = tf.nn.softmax(score, axis=1)
        # 计算输出
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights  # 输出特征向量和权重


# 解码
class Decoder(tf.keras.Model):
    # 设置参数
    def __init__(self, vocab_size: int, embedding_dim: int, dec_units: int):
        """
        解码器的初始函数
        :param vocab_size: 词库大小
        :param embedding_dim: 词向量维度
        :param dec_units: LSTM层的神经元数量
        """
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        # 词嵌入层
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # 添加LSTM层
        self.gru = tf.keras.layers.GRU(self.dec_units, return_sequences=True, return_state=True)
        # 全连接层
        self.fc = tf.keras.layers.Dense(vocab_size)
        # 添加注意力机制
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x: tf.Tensor, hidden: tf.Tensor, enc_output: tf.Tensor) \
            -> typing.Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        前向传播计算
        :param x: 输入的文本
        :param hidden: 上一层输出的特征值
        :param enc_output: 上一层输出的计算结果
        :return: 输出预测结果，当前状态和权重
        """
        # 计算注意力机制层的结果
        context_vector, attention_weights = self.attention(hidden, enc_output)
        # 次嵌入层
        x = self.embedding(x)
        # 词嵌入结果和注意力机制的结果合并
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # 添加注意力机制
        output, state = self.gru(x)

        # 输出结果更新维度
        output = tf.reshape(output, (-1, output.shape[2]))
        # 输出层
        x = self.fc(output)

        return x, state, attention_weights  # 输出预测结果，当前状态和权重


class DataSet:

    def __init__(self):
        # 初始化一些特殊字符的ID
        self.const = {'_BOS': 0, '_EOS': 1, '_PAD': 2, '_UNK': 3}

    def to_ids(self, text, table):
        """
        将给定的文本转换为对应的ID序列

        :param text: 输入的文本
        :param table: 用于转换的哈希表
        :return: ID序列
        """
        tokenized = tf.strings.split(tf.reshape(text, [1]), sep=' ')  # [1]表示一维，形状为1行
        ids = table.lookup(tokenized.values) + len(self.const)  # table是一个tf.lookup.StaticHashTable, 查找对应值的索引，得到的是一个numpy.array数组
        return ids

    def add_start_end_tokens(self, tokens):
        """
        在给定的tokens序列前后分别添加起始和结束符

        :param tokens: 输入的tokens序列
        :return: 添加了起始和结束符的tokens序列
        """
        tmp = tf.concat([[self.const['_BOS']], tf.cast(tokens, tf.int32), [self.const['_EOS']]], axis=0)
        return tmp

    def get_dataset(self, src_path, table):
        """
        从给定的路径加载文本数据并转换为数据集

        :param src_path: 文本数据的路径
        :param table: 用于文本转换的哈希表
        :return: 转换后的数据集
        """
        dataset = tf.data.TextLineDataset(src_path)
        dataset = dataset.map(lambda text: self.to_ids(text, table))
        dataset = dataset.map(lambda tokens: self.add_start_end_tokens(tokens))
        return dataset

    def prepare_data(self, data_path, batch_size=15, max_length=50, shuffle_buffer_size=4):
        """
        准备数据，包括加载词典，加载数据，以及构建数据集

        :param data_path: 数据路径
        :param batch_size: 一次前向/后向传播中提供的训练数据样本数
        :param max_length: 句子的最大词长
        :param shuffle_buffer_size: 清洗数据集时将缓冲的实例数
        :return: table, train_dataset
        """
        # 加载词典
        print(f'[{datetime.datetime.now()}] 加载词典...')
        # 初始化后即不可变的通用哈希表，本质是tensorflow 内置字典，把字典变成tf支持得字典类型。
        table = tf.lookup.StaticHashTable(
            # 要使用的表初始值设定项
                initializer=tf.lookup.TextFileInitializer(
                filename=os.path.join(data_path, 'all_dict.txt'),  # 文件路径
                key_dtype=tf.string,  # 键的类型
                key_index=tf.lookup.TextFileIndex.WHOLE_LINE,  # 键的索引
                value_dtype=tf.int64,  # 值的类型
                value_index=tf.lookup.TextFileIndex.LINE_NUMBER  # 值的索引
            ),
            # 表中缺少键时使用的值。
            default_value=self.const['_UNK'] - len(self.const)
        )

        # 加载数据
        print(f'[{datetime.datetime.now()}] 加载预处理后的数据...')

        # 获取问答数据
        ask = self.get_dataset(os.path.join(data_path, 'ask.txt'), table)
        answer = self.get_dataset(os.path.join(data_path, 'answer.txt'), table)

        # 把数据和特征构造为tf数据集
        train_dataset = tf.data.Dataset.zip((ask, answer))

        # 将数据打乱，数值越大，混乱程度越大
        train_dataset = train_dataset.shuffle(shuffle_buffer_size)
        # 将数据长度变为一致，长度不足用_PAD补齐
        train_dataset = train_dataset.padded_batch(
            batch_size,  # 批次数量大小
            padded_shapes=([max_length + 2], [max_length + 2]),  # 填充的维度
            padding_values=(self.const['_PAD'], self.const['_PAD']),  # 填充的值
            drop_remainder=True,  # 比如batch_size = 3,最后一个batch只有2个样本。默认是不丢掉
        )
        return table, train_dataset


def loss_function(loss_object, real, pred, pad: int = 0):
    """
    loss_object: 损失值计算方式
    real: 真实值
    pred: 预测值
    pad: "填充"占位符的数值映射
    """
    # 计算真实值和预测值的误差
    loss_ = loss_object(real, pred)
    # 返回与输出不相等的值，并用_PAD填充
    mask = tf.math.logical_not(tf.math.equal(real, pad))
    # 数据格式转换为跟损失值一致
    mask = tf.cast(mask, dtype=loss_.dtype)
    # 返回平均误差
    return tf.reduce_mean(loss_ * mask)


def create_model(vocab_size, embedding_dim, hidden_dim):
    """
    创建模型相关的内容
    :param vocab_size: 词表大小，包含原始词和一些特殊占位符
    :param embedding_dim: 词向量的大小
    :param hidden_dim: 隐藏层的神经元个数
    :param device: 是否使用GPU
    :return:
    """
    # 获得当前主机上GPU运算设备的列表
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # 如果有可用的GPU，则使用第一个可用的GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print(f'[{datetime.datetime.now()}] 使用GPU')
    else:
        print(f'[{datetime.datetime.now()}] 使用CPU')

    print(f'[{datetime.datetime.now()}] 初始化模型...')
    # 实例化编码器
    # vocab_size = table.size().numpy() + len(CONST)
    encoder = Encoder(vocab_size, embedding_dim, hidden_dim)
    # 实例化解码器
    decoder = Decoder(vocab_size, embedding_dim, hidden_dim)
    # 设置优化器
    optimizer = tf.keras.optimizers.Adam()
    # 损失值计算方式
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    # 设置模型保存
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    return encoder, decoder, optimizer, loss_object, checkpoint


# 训练步
def train_step(x, y, encoder, decoder, optimizer, loss_object):
    """
    一个批次的训练过程
    :param x: 训练数据x，也就是原始句子
    :param y: 训练数据y，也就是预测的句子
    :param encoder: 编码器
    :param decoder: 解码器
    :param optimizer: 优化器
    :param loss_object: 损失函数
    :return: batch_loss 一个批次的loss结果
    """
    # 获取标签维度
    tgt_width, tgt_length = y.shape
    loss = 0
    # 创建梯度带，用于反向计算导数
    with tf.GradientTape() as tape:
        # 对输入的文本编码
        enc_output, enc_hidden = encoder(x)
        # 设置解码的神经元数目与编码的神经元数目相等
        dec_hidden = enc_hidden
        # 循环每个单词，预测下一个输出
        for t in range(tgt_length - 1):
            # 分别取每一列的值作为输入，赋值为新增一维
            dec_input = tf.expand_dims(y[:, t], 1)
            # 解码，预测下一个值
            predictions, dec_hidden, dec_out = decoder(dec_input, dec_hidden, enc_output)
            # 计算损失值，计算预测的下一个值与原来句子的下一个值的损失
            loss += loss_function(loss_object, y[:, t + 1], predictions)
    # 计算一次训练的平均损失值
    batch_loss = loss / tgt_length
    # 更新预测值
    variables = encoder.trainable_variables + decoder.trainable_variables
    # 反向求导
    gradients = tape.gradient(loss, variables)
    # 利用优化器更新权重
    optimizer.apply_gradients(zip(gradients, variables))
    # 返回每次迭代训练的损失值
    return batch_loss


def train(train_dataset, model, epoch, checkpoint_path='./model'):
    """
    整体训练
    :param train_dataset: 训练数据
    :param encoder: 编码器
    :param decoder: 解码器
    :param loss_object: loss函数对象
    :param optimizer: 优化器对象
    :param epoch: 迭代次数
    :param checkpoint_path: 保存模型的文件夹路径
    :return:
    """
    # encoder, decoder, checkpoint, loss_object, optimizer = model(checkpoint_path, device, CONST, embedding_dim,
    #                                                              hidden_dim)
    # table, train_dataset = prepare_data(CONST, embedding_dim, hidden_dim)
    # 设置模型保存
    # 模型参数保存的路径如果不存在则新建
    encoder, decoder, optimizer, loss_object, checkpoint = model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    print(f'[{datetime.datetime.now()}] 开始训练模型...')
    # 根据设定的训练次数去训练模型
    for ep in range(epoch):
        # 设置损失值
        total_loss = 0
        # 将每批次的数据取出，放入模型里
        for batch, (x, y) in enumerate(train_dataset):
            # 训练并计算损失值
            batch_loss = train_step(x, y, encoder, decoder, optimizer, loss_object)
            total_loss += batch_loss
        if ep % 100 == 0:
            # 每100训练次保存一次模型
            checkpoint_prefix = os.path.join(checkpoint_path, 'ckptss')
            checkpoint.save(file_prefix=checkpoint_prefix)
        print(f'[{datetime.datetime.now()}] 迭代次数: {ep + 1} 损失值: {total_loss:.4f}')
    print('模型训练完毕！')

if __name__ == '__main__':

    # epoch = 501  # 训练次数
    # embedding_dim = 256  # 词嵌入维度
    # hidden_dim = 512  # 隐层神经元个数
    # MAX_LENGTH = 50  # 句子的最大词长
    # CONST = {'_BOS': 0, '_EOS': 1, '_PAD': 2, '_UNK': 3}  # 特殊词
    from functools import partial
    checkpoint_path='./model'
    data_path = './data'  # 数据路径
    max_length = 50
    dataset = DataSet()
    table, train_dataset = dataset.prepare_data(data_path, batch_size=25, max_length=max_length, shuffle_buffer_size=4)
    loss_function = partial(loss_function, pad=dataset.const["_PAD"])
    vocab_size = table.size().numpy() + len(dataset.const)  # 词表大小

    model = create_model(vocab_size, embedding_dim=256, hidden_dim=512)

    train(train_dataset, model, epoch=20, checkpoint_path=checkpoint_path)
