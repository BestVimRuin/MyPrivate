# -*- coding: utf-8 -*-

"""

使用sqlalchemy连接数据库，建立数据库模型，及建表

"""

from __future__ import division
from sqlalchemy import Text,Column,Integer,String, create_engine, Float, Text, Column, Integer, String, Sequence, Boolean,ForeignKey
from sqlalchemy.dialects.mysql import DOUBLE,VARCHAR
from sqlalchemy.ext.declarative import declarative_base  
from sqlalchemy.orm import sessionmaker,relationship



DB_CONNECT = 'mysql+pymysql://admin:admin@168.61.13.128/pianduan?charset=utf8'
engine = create_engine(DB_CONNECT)
DB_Session = sessionmaker(bind=engine)
session = DB_Session()
BaseModel = declarative_base()


class Shape(BaseModel):

    """

    建立Shape表模型，存放形态的基本信息

    参数：
    - **BaseModel(sqlalchemy.ext.declarativedeclarative_base())** : sqlalchemy的子方法

    :返回

    """

    __tablename__ = 'shape'
    id = Column(Integer, Sequence('id'), primary_key=True)
    num = Column(Integer)
    score = Column(Integer)
    dm = Column(Boolean)
    value = Column(Text)
    backward_length = Column(Integer)
    forward_length = Column(Integer)
    fragment = relationship('Fragment', backref='shape')
    zhibiao = relationship('Zhibiao',backref='shape')
    stat = relationship('Stat',backref='shape')
    conver = relationship('Conver',backref='conver')


class Zhibiao(BaseModel):

    """

    建立Zhibiao表模型，存放形态制表信息

    参数：
    - **BaseModel(sqlalchemy.ext.declarativedeclarative_base())** : sqlalchemy的子方法

    :返回
    
    """

    __tablename__ = 'zhibiao'
    id = Column(Integer,Sequence('id'),primary_key=True)
    sid = Column(Integer,ForeignKey('shape.id'))
    sym_new1 = Column(Float)
    start_time = Column(Float)
    start_time_std = Column(Float)
    convMin = Column(Float)
    convMax = Column(Float)
    uparea = Column(Float)
    downarea = Column(Float)
    up_down = Column(Float)
    maxamp = Column(Float)
    minamp = Column(Float)
    orient = Column(Float)
    index_zd = Column(Float)
    index_num = Column(Float)
    index_time = Column(Float)
    total = Column(Float)


class Fragment(BaseModel):

    """

    建立Fragment表模型，存放形态制表信息

    参数：
    - **BaseModel(sqlalchemy.ext.declarativedeclarative_base())** : sqlalchemy的子方法

    :返回
    
    """

    __tablename__ = 'fragment'
    id = Column(Integer, Sequence('id'), primary_key=True)
    sid = Column(Integer,ForeignKey('shape.id'))
    corr = Column(Float)
    stock_name = Column(VARCHAR(50))
    start_time = Column(Integer)
    end_time = Column(Integer)
    price_origin = Column(Text)
    price_standard = Column(Text)
    price_forward = Column(Text)
    zhangci = Column(Integer)
    dieci = Column(Integer)
    location_max = Column(Integer)
    location_min = Column(Integer)
    max_amp = Column(Float)
    min_amp = Column(Float)
    zhang_area = Column(Float)
    die_area = Column(Float)
    each_amp = Column(Text)
    each_conver_amp = Column(Text)


class Stat(BaseModel):

    """

    建立Stat表模型，存放形态制表信息

    参数：
    - **BaseModel(sqlalchemy.ext.declarativedeclarative_base())** : sqlalchemy的子方法

    :返回
    
    """

    __tablename__ = 'stat'
    id = Column(Integer, Sequence('id'), primary_key=True)
    sid = Column(Integer,ForeignKey('shape.id'))
    num = Column(Integer)
    corr = Column(Float)
    count = Column(Integer)
    zhangci = Column(Float)
    dieci = Column(Float)
    zhangarea = Column(Float)
    diearea = Column(Float)
    ave_maxzhang = Column(Float)
    ave_maxdie = Column(Float)
    maxday = Column(Integer)


class Conver(BaseModel):

    """

    建立Conver表模型，存放形态制表信息

    参数：
    - **BaseModel(sqlalchemy.ext.declarativedeclarative_base())** : sqlalchemy的子方法
    
    """

    __tablename__ = 'conver'
    id = Column(Integer, Sequence('id'), primary_key=True)
    sid = Column(Integer,ForeignKey('shape.id'))
    name = Column(Text)
    mean = Column(Float)
    median = Column(Float)

class New(BaseModel):

    """

    建立new表模型，2018.3.5日新的需求

    参数：
    - **BaseModel(sqlalchemy.ext.declarativedeclarative_base())** : sqlalchemy的子方法
    
    """

    __tablename__ = 'new'
    id = Column(Integer, Sequence('id'), primary_key=True)
    sid = Column(Integer,ForeignKey('shape.id'))
    num = Column(Integer)
    upChance = Column(Float)
    highpoint = Column(Float)
    lowpoint = Column(Float)
    maxampButton = Column(Float)
    maxampTop = Column(Float)
    minampButton = Column(Float)
    minampTop = Column(Float)

def init_db():

    """

    创建所有的表

    """

    BaseModel.metadata.create_all(engine)


def drop_db():

    """

    删除所有的表

    """

    BaseModel.metadata.drop_all(engine)

if __name__ == '__main__':
    # drop_db()
    init_db()