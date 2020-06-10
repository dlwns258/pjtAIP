# _*_coding: utf-8 _*_

import tensorflow.compat.v1 as tf
import numpy as np
from pandas.io.parsers import read_csv

tf.disable_v2_behavior()



model = tf.global_variables_initializer();

data = read_csv('./model/TCD/tcd.csv', sep=',', encoding='CP949')

xy = np.array(data, dtype=np.float32)

# 2개의 변인 입력
x_data = xy[:, 0:-1]

# BMI 값을 입력 받습니다.m
y_data = xy[:, [-1]]

# 플레이스 홀더를 설정합니다.
X = tf.placeholder(tf.float32, shape=[None, 25])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([25, 1]), name='DN')
# 가중치 변수에 [25,1] 구조로 랜덤값 생성
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = tf.matmul(X,W) + b
# 예측값을 찾는 방법에 대한 가설***//

cost = tf.reduce_mean(tf.square(hypothesis - Y))
# 비용 함수 설정

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100001):  # 학습을 100000 번 시행
    cost_, hypo_, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 1000 == 0:  # 1000번 단위로 중간 값 산출
        print("#", step, "cost : ", cost_)  # 비용값이 점점 작아져야 옳은 값이 나올 수 있다.
        print("표준편차의 중간값 : ", hypo_[0])  # 가설에 성립하는 표준 편차의 중간 값을 확인해본다.

saver = tf.train.Saver()
save_path = saver.save(sess, "./model/TCD/tcd.cpkt")
print("학습된 모델을 저장했습니다.")