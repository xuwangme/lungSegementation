"""
# @author: xuwang
# @function: 入口函数
# @date: 2018/5/12 15:13
"""
from dataPreprocess import Preprocess
from evaluate import Evaluate
from modelTraining import ModelTraining

def doDataProcess():
    preprocess = Preprocess()
    preprocess.generateImage()
    preprocess.generateMask()

def trainModel():
    mT = ModelTraining()
    for i in range(0, 5):
        seg = i / 5
        print(" ------------>>>>>>> start running seg = ", seg)
        mT.train(seg)

def doEvaluate():
    eva = Evaluate()
    eva.eval()

if __name__ == '__main__':
    # 数据预处理
    # doDataProcess()

    # 模型训练
    # trainModel()

    # 效果评估
    doEvaluate()
