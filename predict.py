from train import create_model
import jieba
import tensorflow as tf
import os

# 读取字段
max_length = 50
checkpoint_path = './model'
data_path = './data'  # 数据路径
with open(os.path.join(data_path, 'all_dict.txt'), 'r', encoding='utf-8') as f:
    all_dict = f.read().split()
const = {'_BOS': 0, '_EOS': 1, '_PAD': 2, '_UNK': 3}  # 特殊词
# 构建: 词-->id的映射字典，从4开始
word2id = {j: i + len(const) for i, j in enumerate(all_dict)}
# 把CONST的值赋值给0-3
word2id.update(const)
# 构建: id-->词的映射字典
id2word = dict(zip(word2id.values(), word2id.keys()))
# 分词保留开始和结束标记
for i in ['_EOS', '_BOS']:
    jieba.add_word(i)


# 模型预测
def predict(sentence, checkpoint_path, word2id, id2word, const, max_length = 50):
    # 导入训练参数
    vocab_size = len(word2id)
    encoder, decoder, optimizer, loss_object, checkpoint = create_model(vocab_size, embedding_dim=256, hidden_dim=512)

    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
    # 给句子添加开始和结束标记
    sentence = '_BOS' + sentence + '_EOS'
    # 添加识别不到的词，用_UNK表示
    inputs = [word2id.get(i, const['_UNK']) for i in jieba.lcut(sentence)]

    # 长度填充
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_length, padding='post', value=const["_PAD"])
    # 将数据转为tensorflow的数据类型
    inputs = tf.convert_to_tensor(inputs)
    # 空字符串，用于保留预测结果
    result = ''

    # 编码
    enc_out, enc_hidden = encoder(inputs)
    dec_hidden = enc_hidden
    # 从“开始”标记进行输入
    dec_input = tf.expand_dims([word2id['_BOS']], 0)
    # 循环预测输出结果
    for t in range(max_length):
        # 解码
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        # 预测出词语对应的id
        predicted_id = tf.argmax(predictions[0]).numpy()
        # 通过字典的映射，用id寻找词，遇到_EOS停止输出
        if id2word.get(predicted_id, '_UNK') == '_EOS':
            break
        # 未预测出来的词用_UNK替代
        result += id2word.get(predicted_id, '_UNK')
        dec_input = tf.expand_dims([predicted_id], 0)

    return result  # 返回预测结果


if __name__ == '__main__':
    print(predict('你好，在吗', checkpoint_path, word2id, id2word, const))

    # while True:
    #     input_sentence = input("输入: ")
    #     if input_sentence.lower() in ["exit", "quit"]:
    #         break
    #     output_sentence = predict(input_sentence, checkpoint_path, word2id, id2word, const)
    #     print("回复: ", output_sentence)

