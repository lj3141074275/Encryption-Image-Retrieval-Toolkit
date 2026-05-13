#KNN加密查询特征
import time
import math
import numpy as np
import random
from sklearn.decomposition import PCA
import torch

def pac_fea(all_fea,n_com):
    pca = PCA(n_components=n_com)
    fea_de=pca.fit_transform(all_fea)
    return fea_de

def generate_random_binary_string(length):#生成随机二进制串
    return ''.join(str(random.randint(0, 1)) for _ in range(length))

def generate_random_array(length, min_value, max_value):#长度，最大值最小值
    return [random.randint(min_value, max_value) for _ in range(length)]

def generate_invertible_matrix(size):#生成可逆矩阵，大小size，[0,1]区间大小
    while True:
        A = np.random.randint(1, 10, size=(size, size))
        det_A = np.linalg.det(A)
        if det_A != 0:
            return A,np.linalg.inv(A)
def gen_key(d_stre,d):#生成密钥：可逆矩阵A,B，二进制串S，随机数W。
    A, A_inv = generate_invertible_matrix(d_stre)
    B, B_inv = generate_invertible_matrix(d_stre)
    S=generate_random_binary_string(d_stre)
    W=generate_random_array(d_stre-d,1,9)
    return A,A_inv,B,B_inv,S,W

def encrypt_p(p,S,W,A,B,d_stre,WT):#对一个特征向量加密
    d=len(p)#d<d_stre
    arr_leng = np.zeros(d_stre)
    cou=0.0
    for i in range(0, d):
        arr_leng[i] = p[i]
        cou=cou+(p[i]*p[i])
    arr_leng[d]=(-0.5)*cou
    j=1
    sum_zero = 0
    last_zero_position = S.rfind('0')
    for i in range(d+1,d_stre):#对于P和-0.5P后面的随机数，s=1时两者分割和。s=0时，设置为随机数99
        if i==last_zero_position:
            arr_leng[i]=-sum_zero/WT[j]
        elif S[i]=='1':
            arr_leng[i]=W[j]
        else:
            arr_leng[i]=99#随机数
            sum_zero=sum_zero+99*WT[j]
        j=j+1

    pa=np.zeros(d_stre)
    pb=np.zeros(d_stre)
    for i in range(d_stre):
        if S[i]=='1':
            pa[i]=arr_leng[i]/2
            pb[i]=arr_leng[i]-pa[i]
        else:
            pa[i]=arr_leng[i]
            pb[i]=arr_leng[i]
    M1_pa=np.dot(A.T,pa)
    M2_pb=np.dot(B.T,pb)
    return M1_pa,M2_pb,arr_leng

def euo_distance(a1,a2):
    euclidean_distance = np.linalg.norm(a1 - a2)
    return euclidean_distance

def euo_distance2(a1,a2):#比较两个向量的欧几里得距离
    cou=0
    for i in range(len(a1)):
        cou=(a1[i]-a2[i])**2+cou
    return math.sqrt(cou)

def encrypt_query(q,S,A_inv,B_inv,W,d_stre,WT):
    r=8
    d = len(q)  # d<d_stre
    arr_leng = np.zeros(d_stre)
    for i in range(0, d):
        arr_leng[i] = q[i]*r
    arr_leng[d] = r
    j = 1
    last_one_position = S.rfind('1')
    sum_one=0
    for i in range(d + 1, d_stre):# 对于P和-0.5P后面的随机数，s=1时两者分割和。s=0时，设置为随机数99
        if i==last_one_position:
            arr_leng[i]=-sum_one/WT[j]
        elif S[i] == '0':
            arr_leng[i] = W[j]
        else:
            arr_leng[i] = 99  # 随机数
            sum_one=sum_one+99*WT[j]
        j = j + 1
    qa = np.zeros(d_stre)
    qb = np.zeros(d_stre)

    for i in range(d_stre):
        if S[i] == '0':
            qa[i] = arr_leng[i] / 2
            qb[i] = arr_leng[i] - qa[i]
        else:
            qa[i] = arr_leng[i]
            qb[i] = arr_leng[i]
    M1_qa = np.dot(A_inv, qa)
    M2_qb = np.dot(B_inv, qb)
    return M1_qa,M2_qb,arr_leng
    #return qa,qb,arr_leng

def distance_comparison(pa_en1,pb_en1,pa_en2,pb_en2,qa_en,qb_en):#取两个p1和p2与q进行比较
    res1=np.dot((pa_en1-pa_en2),qa_en)
    res2=np.dot((pb_en1-pb_en2),qb_en)
    print("sum_res:",res1+res2)
    return res1+res2

def distance_IP(pa_en,pb_en,qa_en,qb_en):#采用的是加密后向量p与q的乘积,各有两个分量
    res1=np.dot(qa_en,pa_en)
    res2=np.dot(qb_en,pb_en)
    return res1+res2

def re_pq_en(pa_en,pb_en,qa_en,qb_en,y_ac,y_test,topk):#放入所有数据库加密特征，以及一个查询特征
    reeuc = np.zeros(len(arrp))
    for i in range(len(arrp)):
        dist = distance_IP(pa_en[i], pb_en[i], qa_en, qb_en)#采用的是加密后向量p与q的乘积
        reeuc[i] = dist
    sort_re = np.argsort(reeuc)[::-1]
    cou_ac = 0
    ap=0.0
    for i in range(0, topk):
        if y_test[int(sort_re[i])] == y_ac:
            cou_ac = cou_ac + 1
            tt=cou_ac/(i+1)
            ap=ap+tt
    ap=ap/cou_ac
    #print("本次正确检索个数 ",cou_ac)
    return cou_ac,ap

def re_pq(p,q):
    reeuc = np.zeros(len(arrp))
    for i in range(len(arrp)):
        dist = euo_distance2(p[i], q)#比较明文向量的欧几里得距离
        reeuc[i] = dist
    sort_re = np.argsort(reeuc)
    sorted_arr = reeuc[sort_re]
    print(reeuc[:9])  # 对应欧距离值
    print("原始排序",sort_re)  # 排序后索引
    print(sort_re[:9])
    print(sorted_arr[:9])

if __name__ == '__main__':

    fea_all = np.load("features/feature_corel1k_vgg16_dim128.npy")
    pca_dim = len(fea_all[0])
    y_test = np.arange(10)
    y_test = np.repeat(y_test, 100)  # 生成特征的标签，计算检索精度
    for i in range(1000):
        y_test[i] = y_test[i] + 1  # corel标签从1开始
    print("---------数据和标签预处理结束------------")

    arrp=fea_all
    arrq=fea_all[0]
    d=len(fea_all[0])
    d_stre=2*d
    A_inv=np.load("datas/matrix_A_inv.npy")
    B_inv = np.load("datas/matrix_B_inv.npy")
    S = np.load("datas/bin_string_S.npy")
    S=str(S)
    W = np.load("datas/bin_string_W.npy")

    last_one_position = S.rfind('1')
    last_zero_position = S.rfind('0')
    WT=generate_random_array(d_stre-d,1,9)

    # 加密所有查询向量
    sss2 = time.time()
    qa_en=np.zeros((len(arrp), d_stre))
    qb_en=np.zeros((len(arrp), d_stre))
    q_en=np.zeros((len(arrp), d_stre))
    for i in range(len(arrp)):
        qa_en[i],qb_en[i],q_en[i] = encrypt_query(arrp[i],S,A_inv,B_inv,WT,d_stre,W)
    eee2 = time.time()
    sss2time = eee2 - sss2
    print("加密所有查询向量的时间", sss2time)


