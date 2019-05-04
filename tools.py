# coding: utf-8
"""
@author Liuchen
2018
"""
import logging
logger = logging.getLogger('main.tools')


class Parameters:
    '''
    超参数对像，用于存储与管理各种参数
    '''

    def __init__(self, **args):
        self.__dict__.update(args)  # 将参数加入到self中

    def __add__(self, hps):
        '''
        重载+，使两个超参对像可以相加
        '''
        if not isinstance(hps, Parameters):
            raise Exception(f'{type(self)} and {type(hps)} cannot be added together！！！ --- by LIC ')
        param_dict = dict()
        param_dict.update(self.__dict__)
        param_dict.update(hps.__dict__)
        return Parameters(** param_dict)

    def to_str(self):
        '''
        输出参数为字符串
        '''
        params = sorted(self.__dict__.items(), key=lambda item: item[0])
        output = ''
        for param, value in params:
            output += f'{param:18}  {value}\n'
        return output

    def __str__(self):
        return self.to_str()

    def to_dict(self):
        '''
        将全部超参数输出为字典
        '''
        return self.__dict__

    def get(self, attr_name):
        '''
        获取参数值，若不存在返回None
        '''
        return self.__dict__.get(attr_name)

    def set(self, key, value):
        '''
        添加或更新一个参数
        '''
        self.__dict__[key] = value

    def default(self, default_params):
        '''
        设置默认参数，仅添加缺失的参数，不改变已有参数
        '''
        for key, value in default_params.items():
            if self.__dict__.get(key) is None:
                self.__dict__[key] = value

    def update(self, params):
        '''
        仅更新已有参数，不添加新参数
        '''
        param_set = params
        if isinstance(params, Parameters):
            param_set = params.to_dict()
        for key, value in param_set.items():
            if self.__dict__.get(key) is not None:
                self.__dict__[key] = value

    def extend(self, params):
        '''
        添加新参数，同时更新已有参数
        '''
        if isinstance(params, Parameters):
            params = params.to_dict()
        self.__dict__.update(params)
