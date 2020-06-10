# _*_coding: utf-8 _*_

import tensorflow.compat.v1 as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.disable_v2_behavior()

def Protein_check(protein):
    if protein == 1:
        protein_1 = 1
        protein_2 = 0
        protein_3 = 0
    elif protein ==2:
        protein_1 = 0
        protein_2 = 1
        protein_3 = 0
    else:
        protein_1 = 0
        protein_2 = 0
        protein_3 = 1

    list =[protein_1,protein_2,protein_3]

    return  list

def X_ray_check(x_ray):
    if x_ray == 1:
        x_ray1 = 1
        x_ray2 = 0
        x_ray3 = 0
        x_ray4 = 0
    elif x_ray == 2:
        x_ray1 = 0
        x_ray2 = 1
        x_ray3 = 0
        x_ray4 = 0
    elif x_ray == 3:
        x_ray1 = 0
        x_ray2 = 0
        x_ray3 = 1
        x_ray4 = 0
    else:
        x_ray1 = 0
        x_ray2 = 0
        x_ray3 = 0
        x_ray4 = 1

    list =[x_ray1,x_ray2,x_ray3,x_ray4]

    return list

def Colonoscopy_load(ch_l):
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, 25])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([25, 1]), name= 'DN')
    b = tf.Variable(tf.random_normal([1]), name = 'bias')

    hypothesis = tf.matmul(X,W) + b

    # 저장된 모델을 불러오는 객체를 초기화

    model = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(model)
        saver = tf.train.Saver()
        save_path = "./model/DN/DN_learn.cpkt"
        saver.restore(sess, save_path)
        data = ((ch_l['sex'], ch_l['age'], ch_l['alchol'], ch_l['cigarette'], ch_l['exercise'], ch_l['length'], ch_l['weight'],
         ch_l['obesity'], ch_l['bmi'], ch_l['waist'], ch_l['ratio'], ch_l['left'], ch_l['right'], ch_l['max'],ch_l['min'],
         ch_l['protein'], ch_l['color'], ch_l['sugar'], ch_l['tchol'], ch_l['hdl'], ch_l['ldl'], ch_l['crea'],
         ch_l['ast'],ch_l['alt'], ch_l['gam_gpt']),)
        arr = np.array(data, dtype=np.float32)

        x_data = arr[0:24]
        dict = sess.run(hypothesis, feed_dict={X: x_data})

        if dict[0] <0.5:
            result = 0
        else:
            result= int(np.round(dict[0]))
    return result

def Gastroscopy_load(ch_l):
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, 25])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([25, 1]), name='WN')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.matmul(X, W) + b

    # 저장된 모델을 불러오는 객체를 초기화

    model = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(model)
        saver_wn = tf.train.Saver()
        save_path = "./model/WN/WN_learn.cpkt"
        saver_wn.restore(sess, save_path)
        data = ((ch_l['sex'], ch_l['age'], ch_l['alchol'], ch_l['cigarette'], ch_l['exercise'], ch_l['length'], ch_l['weight'],
         ch_l['obesity'], ch_l['bmi'], ch_l['waist'], ch_l['ratio'], ch_l['left'], ch_l['right'], ch_l['max'],ch_l['min'],
         ch_l['protein'], ch_l['color'], ch_l['sugar'], ch_l['tchol'], ch_l['hdl'], ch_l['ldl'], ch_l['crea'],ch_l['ast'],
         ch_l['alt'], ch_l['gam_gpt']),)
        arr = np.array(data, dtype=np.float32)

        x_data = arr[0:24]
        dict = sess.run(hypothesis, feed_dict={X: x_data})

        if dict[0] < 0.5:
            result = 1
        elif dict[0] > 4:
            result = 4
        else:
            result= int(np.round(dict[0]))

    return result

def Tcd_load(ch_l):

    p_list=Protein_check(ch_l['protein'])
    x_list=X_ray_check(ch_l['x_ray'])

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, 25])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([25, 1]), name='DN')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.matmul(X, W) + b

    # 저장된 모델을 불러오는 객체를 초기화

    model = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(model)
        saver_tcd = tf.train.Saver()
        save_path = "./model/TCD/tcd.cpkt"
        saver_tcd.restore(sess, save_path)
        data = ((ch_l['sex'], ch_l['age'], ch_l['length'], ch_l['weight'], ch_l['obesity'], ch_l['max'], ch_l['min'],
         ch_l['hgb'], ch_l['tchol'], ch_l['hdl'], ch_l['triglyceride'], ch_l['ldl'], ch_l['sugar'], p_list[0],p_list[1],
         p_list[2], ch_l['ast'], ch_l['alt'], ch_l['gam_gpt'], ch_l['crea'], ch_l['bmi'], x_list[0],x_list[1],
         x_list[2], x_list[3]),)
        arr = np.array(data, dtype=np.float32)

        x_data = arr[0:24]
        dict = sess.run(hypothesis, feed_dict={X: x_data})

        if dict[0] < 0.5:
            result = 1
        elif dict[0] > 4:
            result = 4
        else:
            result= int(np.round(dict[0]))

    return result

def Thorax_load(ch_l):

    p_list=Protein_check(ch_l['protein'])
    x_list=X_ray_check(ch_l['x_ray'])

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, 25])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([25, 1]), name='DN')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.matmul(X, W) + b

    # 저장된 모델을 불러오는 객체를 초기화

    model = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(model)
        saver_thorax = tf.train.Saver()
        save_path = "./model/THORAX/hyungbu_ct.cpkt"
        saver_thorax.restore(sess, save_path)
        data = ((ch_l['sex'], ch_l['age'], ch_l['length'], ch_l['weight'], ch_l['obesity'], ch_l['max'], ch_l['min'],
         ch_l['hgb'], ch_l['tchol'], ch_l['hdl'], ch_l['triglyceride'], ch_l['ldl'], ch_l['sugar'], p_list[0],p_list[1],
         p_list[2], ch_l['ast'], ch_l['alt'], ch_l['gam_gpt'], ch_l['crea'], ch_l['bmi'], x_list[0],x_list[1],
         x_list[2], x_list[3]),)
        arr = np.array(data, dtype=np.float32)

        x_data = arr[0:24]
        dict = sess.run(hypothesis, feed_dict={X: x_data})

        if dict[0] < 0.5:
            result = 1
        elif dict[0] > 4:
            result = 4
        else:
            result= int(np.round(dict[0]))

    return result

def Thyroid_load(ch_l):

    p_list=Protein_check(ch_l['protein'])
    x_list=X_ray_check(ch_l['x_ray'])

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, 25])
    Y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([25, 1]), name='DN')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = tf.matmul(X, W) + b

    # 저장된 모델을 불러오는 객체를 초기화

    model = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(model)
        saver_thyroid = tf.train.Saver()
        save_path = "./model/THYROID/gabsang.cpkt"
        saver_thyroid.restore(sess, save_path)
        data = ((ch_l['sex'], ch_l['age'], ch_l['length'], ch_l['weight'], ch_l['obesity'], ch_l['max'], ch_l['min'],
         ch_l['hgb'], ch_l['tchol'], ch_l['hdl'], ch_l['triglyceride'], ch_l['ldl'], ch_l['sugar'], p_list[0],p_list[1],
         p_list[2], ch_l['ast'], ch_l['alt'], ch_l['gam_gpt'], ch_l['crea'], ch_l['bmi'], x_list[0],x_list[1],
         x_list[2], x_list[3]),)
        arr = np.array(data, dtype=np.float32)

        x_data = arr[0:24]
        dict = sess.run(hypothesis, feed_dict={X: x_data})

        if dict[0] < 0.5:
            result = 1
        elif dict[0] > 4:
            result = 4
        else:
            result= int(np.round(dict[0]))

    return result