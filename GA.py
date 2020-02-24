# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 08:09:32 2020

genetic algorithm (GA) 
"""

import pandas as pd
import numpy as np
import random
from keras.models import load_model

"""

"""        
def cal_binary(value):
    down = sample_down
    up = sample_up
    binary = ''
    for ii in np.arange(len(value)):
        digit_value = int((2**digit-1)*(value[ii]-down[ii])/(up[ii]-down[ii]))
        binary_piece = bin(digit_value)
        binary_piece = binary_piece[2:]
        binary_piece = (digit-len(binary_piece))*'0'+binary_piece
        binary = binary + binary_piece
    return binary

def cal_value(binary):
    down = sample_down
    up = sample_up
    value = np.zeros((1,len(down)))
#    print(value.size)
    for ii in np.arange(value.size):
        binary_piece = binary[(ii*digit):(ii*digit+digit)]
        value[0,ii] = int(binary_piece,2)/int(2**digit-1)*(up[ii]-down[ii])+down[ii]
    return value[0,:].tolist()

def gen_sample(sample_number):     
#    all_samples = []
    value = np.zeros([sample_number, len(sample_down)])
    for ii in np.arange(sample_number):
        for jj in np.arange(len(sample_down)):
            value[ii,jj] = random.uniform(sample_down[jj], sample_up[jj])
    all_samples = value.tolist()
    return all_samples

def choose_sample(sample_all):
    num = random.choice(np.arange(len(sample_all)))
    return sample_all[num]

def variation_sample(one_sample):
    binary = cal_binary(one_sample)
    for ii in np.arange(variation_length):   
        num = random.choice(np.arange(len(binary)))
        if binary[num] == '0':
            binary = binary[0:num]+'1'+binary[(num+1):]
        elif binary[num] == '1':
            binary = binary[0:num]+'0'+binary[(num+1):]
    one_sample = cal_value(binary)
    return one_sample

def cross_sample(one_sample, two_sample):
    binary_one = cal_binary(one_sample)
    binary_two = cal_binary(two_sample)
    cross_length_max = 10
    cross_length = random.choice(np.arange(cross_length_max))
    cross_start = random.choice(np.arange(len(binary_one) - cross_length))
    change_two = binary_two[cross_start:(cross_start+cross_length)]
    binary_one = binary_one[0:cross_start] + change_two + binary_one[(cross_start+cross_length):]
#    print(binary_one)
    one_sample = cal_value(binary_one)
    return one_sample

def cal_output(sample):
    input_up = np.array([20, 500, 973])
    output_up = np.array([20, 2, 5000])
    anode_veosity = anode_velocity*np.ones([sample.shape[0],1])
    Xtest = pd.DataFrame(np.append(anode_veosity, sample, axis=1))
    for ii in np.arange(3):
        Xtest.iloc[:, ii] = Xtest.iloc[:, ii]/input_up[ii]
    Ypredicted = model.predict(Xtest)
    Ypredicted = pd.DataFrame(Ypredicted)
    for ii in np.arange(3):
        Ypredicted.iloc[:, ii] = Ypredicted.iloc[:, ii]*output_up[ii]
    return np.array(Ypredicted)

def one_change(all_sample):
    new_sample = choose_sample(all_sample)

    if random.random() <= cross_probability:
        sample_cross = choose_sample(all_sample)
        new_sample = cross_sample(new_sample, sample_cross)
        
    if random.random() <= variation_probability:
        new_sample = variation_sample(new_sample)
        
    return new_sample
    
def enlarge_sample(all_sample, mag):
    enlarge_sample = all_sample.copy()
    for ii in np.arange(mag*len(all_sample)):
        new_sample = one_change(all_sample)
        enlarge_sample.append(new_sample)
    new_sample = gen_sample(len(all_sample))
    enlarge_sample = enlarge_sample + new_sample
    return enlarge_sample

def sort_sample(sample):
    sample_input = np.array(sample)
    sample_output = cal_output(sample_input)
    sample_all = pd.DataFrame(np.append(sample_input, sample_output, axis=1))
    sample_all.columns = ['cathode velocity',
                        'cathode tem',
                        'y tem gradient',
                        'Q',
                        'current density']
    hh = sample_all
    hh1 = hh[abs(hh['y tem gradient'])<tem_gradient_max]
    hh2 = hh1.sort_values(by='current density', ascending=False)
    print('此次循环，最终Y的数值是：', hh2.iloc[0,4], '对应变量值是', hh2.iloc[0,0:4])
    print()
    return np.array(hh2.iloc[0:sample_num, 0:2])
    
global variation_probability, variation_length, cross_probability
variation_probability = 0.2
variation_length = 5
cross_probability = 0.8

"""
可调节参数
"""
global sample_down, sample_up, digit, sample_num
sample_num = 100
sample_down = [50, 773] #变量1和变量2的下限
sample_up = [500, 973] #变量1和变量2的上限
digit = 10

global model
model = load_model('E:\\ANN\\model3.h5')

global anode_velocity, tem_gradient_max
anode_velocity = 16
tem_gradient_max = 2

circle_num = 100
enlarge_num = 4
"""
程序开始运行
"""
sample = gen_sample(sample_num)
sample1 = sample
for ii in np.arange(circle_num):
    print('这是第', ii+1, '次循环')
    sample2 = enlarge_sample(sample1, enlarge_num)
    sample3 = sort_sample(sample2)
    sample1 = list(sample3)


