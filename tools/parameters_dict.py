import pandas as pd
import numpy as np

class parametersDict(object):
    def __init__(self,parameters_dict):
        self.parameters_dict = parameters_dict
        self.length = []
        self.keys = []
        for key,values in self.parameters_dict.items():
            self.keys.append(key)
            self.length.append(len(values))
        self.nums = [1] * len(self.length)
        for i in range(len(self.length)):
            for j in range(i,len(self.length)):
                self.nums[i] *= self.length[j]
        self.para_dis = []
        print(self.length)
        print(self.nums)
                
    def getindex(self,index):
        result = []
        value = index
        for i in range(len(self.nums) - 1):
            result.append(value // self.nums[i+1])
            value = value - result[i] * self.nums[i+1]
        result.append(value) 
        result_dict = dict()
        for index,value in enumerate(result):
            result_dict[self.keys[index]] = self.parameters_dict.get(self.keys[index])[value]
        return result_dict
    
    def myiter(self):
        for i in range(0,self.nums[0]):
            self.para_dis.append(self.getindex(i))
        return self.para_dis
