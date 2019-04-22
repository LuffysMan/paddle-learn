#coding: utf-8
import numpy
import os   
import paddle.fluid as fluid

from PIL import Image

from train import save_dirname

"""
加载模型, 并进行预测
"""

def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = numpy.array(im).reshape(1, 1, 28, 28).astype(numpy.float32)
    im = im / 255.0 * 2.0 - 1.0     #为什么乘以2减去1
    return im

cur_dir = os.getcwd()
tensor_img = load_image(cur_dir + '/image/infer_2.png')



infer_exe = fluid.Executor(fluid.CPUPlace())
inference_scope = fluid.core.Scope()

# 加载数据并开始预测
with fluid.scope_guard(inference_scope):
    #获取训练好的模型
    #从指定目录中加载 推理model(inference model)
    [inference_program,                                           #推理Program
     feed_target_names,                                           #是一个str列表，它包含需要在推理 Program 中提供数据的变量的名称。 
     fetch_targets] = fluid.io.load_inference_model(save_dirname,#fetch_targets：是一个 Variable 列表，从中我们可以得到推断结果。model_save_dir：模型保存的路径
                                                    infer_exe)     #infer_exe: 运行 inference model的 executor
    results = infer_exe.run(program=inference_program,     #运行推测程序
                   feed={feed_target_names[0]: tensor_img}, #喂入要预测的img
                   fetch_list=fetch_targets)         #得到推测结果,    
print(results)