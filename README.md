首先建立神经网络类的框架，然后填充初始化函数。因为训练函数和查询函数有一部分相似，所以先填充查询函数的代码。  
由于要将训练好的模型参加kaggle上的比赛，所以最后要将结果保存再csv文件中，所以要先import csv，  
使用t=open('answer.csv','w',newline='')语句打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。  
使用newline=''，是为了消除最后得到csv文件会多空一行的问题。因为一开始并未有answer.csv的文件，所以会创建一个新的csv文件，最后用close()函数结束对csv文件的操作  
在这次作业的过程中我也发现了一些坑点，因为一开始没有看群文件中的sample_submission.csv文件，导致没注意到要有“ImageId”和“Label”两个标签，导致在对answer.csv文件进行修改，只有通过query函数查询到的结果，导致第一次在kaggle上提交csv文件时，发现网页报错。
```python
import numpy
import scipy.special
import csv
class neuralNetwork:
    #初始化神经网络
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #初始化每层的节点数及学习率
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.lr=learningrate

        #设置权重矩阵
        self.wih = numpy.random.normal(0.0,pow(self.hnodes, -0.5),(self.hnodes, self.inodes))
        #print(self.wih)
        self.who = numpy.random.normal(0.0,pow(self.onodes, -0.5),(self.onodes, self.hnodes))
        #print(self.who)
        self.activation_function = lambda x:scipy.special.expit(x)

    #训练神经网络
    def train(self, inputs_list, targets_list):

        #将输入列表转换为二维数组
        inputs=numpy.array(inputs_list, ndmin=2).T
        targets=numpy.array(targets_list, ndmin=2).T

        #计算信号从输入层到隐藏层
        hidden_inputs=numpy.dot(self.wih, inputs)

        #计算隐藏层的输出
        hidden_outputs=self.activation_function(hidden_inputs)

        #计算信号从隐藏层到输出层
        final_inputs=numpy.dot(self.who, hidden_outputs)

        #计算输出层的最终输出
        final_outputs=self.activation_function(final_inputs)

        #计算误差
        output_errors=targets - final_outputs

        #返回去计算隐藏层误差
        hidden_errors=numpy.dot(self.who.T, output_errors)

        #更新权重
        self.who+=self.lr*numpy.dot((output_errors *final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.wih+=self.lr*numpy.dot((hidden_errors *hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

    def query(self,inputs_list):
        #将输入列表转换为二维数组
        inputs = numpy.array(inputs_list, ndmin=2).T

        #计算信号从输入层到隐藏层
        hidden_inputs = numpy.dot(self.wih, inputs)

        # 计算隐藏层的输出
        hidden_outputs =self.activation_function(hidden_inputs)

        # 计算信号从隐藏层到输出层
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # 计算输出层的最终输出
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
#设置每层的节点数及学习率
input_nodes=784
hidden_nodes=101
output_nodes=11
learning_rate=0.095

n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

#将训练数据加载到列表中
training_data_file = open("C:/Users/余育洲/Desktop/DigitalRecognizerData/train.csv", 'r')
training_data_list = training_data_file.readlines()[1:]
training_data_file.close()

#设置使用训练数据集的次数
epochs = 10
for e in range(epochs):
    for record in training_data_list:

        #用逗号分隔记录
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        #创建目标输出值
        targes = numpy.zeros(output_nodes) + 0.01
        targes[int(all_values[0])] = 0.99
        n.train(inputs, targes)

#将测试数据加载到列表中
test_data_file = open("C:/Users/余育洲/Desktop/DigitalRecognizerData/test.csv", 'r')
test_data_list = test_data_file.readlines()[1:]
test_data_file.close()

jilu=[]
p=[]
for record in test_data_list:
    all_values = record.split(',')
    intputs = (numpy.asfarray(all_values[0:])/255.0*0.99)+0.01
    #查询网络
    outputs = n.query(intputs)
    label = numpy.argmax(outputs)
    jilu.append(label)
n=["ImageId","Label"]
t=open('answer.csv','w',newline='')
writer=csv.writer(t)
writer.writerow(n)
i=1
for j in jilu:
    p.append(i)
    i=i+1
    p.append(j)
    writer.writerow(p)
    p.clear()
t.close()


























```
