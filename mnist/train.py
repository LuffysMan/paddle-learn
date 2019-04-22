# 查看个人持久化工作区文件

import os
from PIL import Image # 导入图像处理模块
import matplotlib.pyplot as plt
import numpy
import paddle # 导入paddle模块
import paddle.fluid as fluid
from paddle.utils.plot import Ploter
import matplotlib.pyplot as plt
import pickle

train_prompt = 'Train cost'
test_prompt = 'Test cost'
cost_ploter = Ploter(train_prompt, test_prompt)

BATCH_SIZE = 64
train_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
                            batch_size=BATCH_SIZE, )

test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size = BATCH_SIZE)

use_cuda = False        #是否使用cuda
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

#设置训练过程的超参：
EPOCH_NUM = 5   #训练5轮
epochs = [epoch for epoch in range(EPOCH_NUM)]
#模型保存路径
save_dirname = 'recognize_digits.inference.model'


def softmax_regression():
    """
    定义softmax分类器：
        一个以softmax为激活函数的全连接层
    Return:
        predict_image -- 分类的结果
    """
    img = fluid.layers.data(name='img', shape=[1,28,28], dtype='float32')
    predict = fluid.layers.fc(img, 10, act='softmax')
    return predict

def multilayer_percetion():
    """
    定义多层感知机分类器：
        含有两个隐藏层（全连接层）的多层感知器
        其中前两个隐藏层的激活函数采用 ReLU，输出层的激活函数用 Softmax
    Return:
        predict_image -- 分类的结果
    """
    img = fluid.layers.data(name='img', shape=[1,28,28], dtype='float32')
    hidden_layer1 = fluid.layers.fc(img, 200, act='relu')    
    hidden_layer2 = fluid.layers.fc(hidden_layer1, 200, act='relu')    
    predict = fluid.layers.fc(hidden_layer2, 10, act='softmax')
    return predict

def convolutional_neural_network(img):
    """
    定义卷积神经网络分类器：
        输入的二维图像，经过两个卷积-池化层，使用以softmax为激活函数的全连接层作为输出层

    Return:
        predict -- 分类的结果
    """
    # img = fluid.layers.data(name='img', shape=[1,28,28], dtype='float32')
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=3,
        num_filters=20, 
        pool_size=2,
        pool_stride=2,
        act='relu')
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=3,
        num_filters=50, 
        pool_size=2,
        pool_stride=2,
        act='relu')
    conv_pool_1 = fluid.layers.batch_norm(conv_pool_1)
    prediction = fluid.layers.fc(conv_pool_2, 10, act='softmax')
    return prediction

def train_program(img, label):
    """
    配置train_program

    Return:
        predict -- 分类的结果
        avg_cost -- 平均损失
        acc -- 分类的准确率
    """ 
    # label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    #predict = softmax_regression()
    #predict = multilayer_percetion()
    predict = convolutional_neural_network(img)
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    acc = fluid.layers.accuracy(input=predict, label=label)
    return predict, [avg_cost, acc]

def optimizer_program():
    return fluid.optimizer.Adam(learning_rate = 0.001)

def event_handler(pass_id, batch_id, cost):
    print("Pass %d, Batch %d, Cost %f" % (pass_id,batch_id, cost))

# 将训练过程绘图表示
def event_handler_plot(ploter_title, step, cost):
    cost_ploter.append(ploter_title, step, cost)
    cost_ploter.plot()

def train_test(train_test_program, train_test_feed, train_test_reader, executor, acc, avg_loss):
    """使用验证集对模型进行验证
    """
    # 将分类准确率存储在acc_set中
    acc_set = []
    # 将平均损失存储在avg_loss_set中
    avg_loss_set = []
    # 将测试 reader yield 出的每一个数据传入网络中进行训练
    for test_data in train_test_reader():
        acc_np, avg_loss_np = executor.run(
            program = train_test_program,
            feed = train_test_feed.feed(test_data),
            fetch_list=[acc, avg_loss]
        )   
        acc_set.append(float(acc_np))
        avg_loss_set.append(float(avg_loss_np))
        # 获得测试数据上的准确率和损失值
    acc_val_mean = numpy.array(acc_set).mean()
    avg_loss_val_mean = numpy.array(avg_loss_set).mean()
    # 返回平均损失值，平均准确率
    return avg_loss_val_mean, acc_val_mean


def forward_propagation():
    """定义前向传播过程

    """

    # 输入的原始图像数据，大小为28*28*1
    img = fluid.layers.data('img', shape=[1,28,28], dtype='float32')
    # 标签层，名称为label,对应输入图片的类别标签
    label = fluid.layers.data('label', shape=[1], dtype='int64')

    # 告知网络传入的数据分为两部分，第一部分是img值，第二部分是label值
    feeder = fluid.DataFeeder(feed_list= [img, label], place=place)
    
    # 调用train_program 获取预测值，损失值，
    prediction, [avg_loss, acc]  = train_program(img, label)

    # 选择Adam优化器
    optimizer = fluid.optimizer.Adam(learning_rate = 0.001)
    optimizer.minimize(avg_loss)

    return prediction,  [avg_loss, acc], feeder

def train():
    #定义前向传播计算图
    prediction,  [avg_loss, acc], feeder = forward_propagation()

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())        #初始化变量

    main_program = fluid.default_main_program()
    test_program = fluid.default_main_program().clone(for_test=True)

    lists = []
    trend_train = []
    trend_val = []      #{epoch_id: avg_loss_val}
    

    #喂数据进行训练和验证
    step = 0
    for epoch_id in epochs:
        for step_id, data in enumerate(train_reader()):
            metrics = exe.run(main_program,
                            feed=feeder.feed(data),
                            fetch_list=[avg_loss, acc])
            if step % 100 == 0: #每训练100次 打印一次log
                print("Pass %d, Batch %d, Cost %f" % (step, epoch_id, metrics[0]))
                # event_handler_plot(train_prompt, step, metrics[0])
                #保存avg_loss和acc用于绘制曲线
                trend_train.append([step, metrics[0]])
            step += 1

        #测试每个epoch的分类效果
        avg_loss_val, acc_val = train_test(
            train_test_program=test_program,
            train_test_reader=test_reader,
            train_test_feed = feeder,
            executor=exe,
            acc=acc,
            avg_loss=avg_loss)
        #保存验证loss
        trend_val.append([epoch_id, avg_loss_val])

        print("Test with Epoch %d, avg_loss %d, acc %s"%(epoch_id, avg_loss_val, acc_val))
        # event_handler_plot(test_prompt, step, metrics[0])
        lists.append((epoch_id, avg_loss_val, acc_val))
        #保存训练好的模型参数用于预测
        if save_dirname:
            fluid.io.save_inference_model(
                save_dirname, 
                ['img'],
                [prediction],
                exe,
                model_filename=None,
                params_filename=None
            )

    #序列化训练过程的loss到文件用于绘图
    trend = [trend_train, trend_val]
    with open('trend.dmp', mode='wb') as f:
        pickle.dump(trend, f, 3)

       

    # 选择效果最好的pass
    print('最好的轮次:', len(lists))
    best = sorted(lists, key=lambda list: float(list[1]))[0]
    print('Best pass is %s, testing Avgcost is %s' % (best[0], best[1]))
    print('The classification accuracy is %.2f%%' % (float(best[2]) * 100))

def plot_trend():
    #读取文件
    with open('trend.dmp', mode='rb') as f:
        trend = pickle.load(f)
    trend_train = trend[0]
    trend_val = trend[1]
    #绘制曲线
    figure = plt.figure('trend')
    x_label = []
    y_label = []
    for x, y in trend_train:
        x_label.append(x)
        y_label.append(y)
    sub1 = plt.subplot(211)
    sub1.scatter(x_label, y_label, s=5)
    # plt.plot(x_label, y_label, 'g')
    x_val = []
    y_val = []
    for x, y in trend_val:
        x_val.append(x)
        y_val.append(y)
    sub2 = plt.subplot(212)
    sub2.scatter(x_val, y_val, s=5)
    # plt.plot(x_label, y_label, 'b')
    plt.show()

if __name__ == "__main__":
    # train()
    plot_trend()




