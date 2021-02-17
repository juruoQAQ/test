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
hidden_nodes=102
output_nodes=10
learning_rate=0.05

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
t=open('sample_submission.csv','w')
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












