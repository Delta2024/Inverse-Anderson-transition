from numpy import random
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib import rcParams

config = {
    "font.family":'serif',
    "font.size": 20,
    'axes.labelsize' : 20,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
    'axes.unicode_minus':False}
rcParams.update(config)

#返回哈密顿矩阵，N是晶胞数量，delta是概率函数的参数，ratio时nu/kappa，varphi是磁通量
def Hamin_H(N = 201,delta=0.5,ratio = 1.0,varphi = np.pi):
    H = np.zeros((3*N,3*N),dtype = complex)
    kapa = 1 #能量量化的最小单位

    if ratio > 0:  # nu / kapa > 0的情况才会有无序
        uni_lis = random.uniform(-1,1,(1,N))#N个相互独立的分布在[-1,1]的无序
        disorder = [] 
        flag = 0 #flag索引uni_lis
        if delta>0:
            for i in range(3*N):
                if (i%3) == 0:
                    disorder.append(0) #一维金刚石链的中心格点位势能为0
                elif ((i-1)%3) == 0:
                    if uni_lis[0][flag]>0:
                        disorder.append(uni_lis[0][flag]*delta-delta/2+kapa*ratio)
                        flag += 1
                    else:
                        disorder.append(uni_lis[0][flag]*delta+delta/2-kapa*ratio)
                        flag += 1                 
                else:
                    disorder.append(-disorder[i-1]) #反对称无序，如果是对称无序就把负号去了
        else:
            for i in range(3*N):
                if (i%3) == 0:
                    disorder.append(0) #一维金刚石链的中心格点位势能为0
                elif ((i-1)%3) == 0:
                    if bool(random.binomial(1,0.5)):
                        disorder.append(kapa*ratio)  #默认nu为1倍的κ
                    else:
                        disorder.append(-kapa*ratio)             
                else:
                    disorder.append(-disorder[i-1])
        np.fill_diagonal(H, disorder)
  
    for i in range(N):
        H[3*i,3*i+1] = kapa*np.exp(1j*varphi)
        H[3*i,3*i+2] = kapa
        H[3*i+1,3*i] = kapa*np.exp(1j*varphi)
        H[3*i+2,3*i] = kapa
        H[3*i-1,3*i] = kapa
        H[3*i-2,3*i] = kapa
        H[3*i,3*i-1] = kapa
        H[3*i,3*i-2] = kapa  
    return H

#返回演变图矩阵和偏差图，H为哈密顿矩阵，initial_v是初始时刻的波函数，t_total是演化的总时间
def evo_with_site(H,initial_v,t_total=100):#仅返回演变矩阵和标准差sigma_z，这个耗时巨大
    N = int(np.shape(H)[0]/3)
    sigma_z = [0]
    n = []
    for i in range(-100,101):#默认格点原胞编号从-100到100
        n += [i,i,i]
    n = np.array(n)
    evo_201 = np.transpose(initial_v)

    for i in range(1,t_total+1):
        result = expm(-1j*H*i)@initial_v
        result = np.transpose(result)
        evo_201 = np.append(evo_201, abs(result), axis=0)
        sigma_z.append(float(np.sqrt(np.sum(n*n*abs(result)*abs(result)))))#每个波的平方和均为一
    return evo_201,sigma_z

#返回本征值和对应的逆参与比值，H为哈密顿矩阵
def compute_IPR(H):
    e,vector = np.linalg.eigh(H) #能量大小是排序的
    N = int(np.shape(H)[0]/3)
    IPR = []
    for i in range(3*N):
        IPR.append(abs(sum(vector[:,i]*vector[:,i]*vector[:,i]*vector[:,i])))#IPR是个普通列表
    return e,IPR

#返回本征值，H为哈密顿矩阵
def compute_energy(H):
    energies = np.linalg.eigvalsh(H)#大小是排序的
    return energies

if __name__=='__main__':
    import time
    t1 = time.time()
    #计算能谱
    N = 201
    H = Hamin_H(N,0.5,1,np.pi)
    energy = compute_energy(H)
    print("第一部分耗时{}".format(time.time()-t1))
    n = [i for i in range(3*N)]
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(n,energy,"ro",markersize=2)
    plt.xlabel(r"态数量")
    plt.ylabel(r"能量 E/κ")
    plt.yticks(np.arange(-4,6,2),np.arange(-4,6,2))
    

    #计算伯努利分布下的IPR
    t2 = time.time()
    e,IPR = compute_IPR(H)
    print("第二部分耗时{}".format(time.time()-t2))
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(e,IPR,".")
    plt.xlabel(r"Energy $E/κ$")
    plt.ylabel(r"IPR")
    plt.title(r"IPR distribution with energy")
    


    t3 = time.time()
    initial_v = np.zeros((3*N,1)) #列向量
    initial_v[300] = 1
    H = Hamin_H(201,0,np.sqrt(2))
    evo,sigma_z = evo_with_site(H,initial_v,100)
    propa_κz = [m for m in range(0,101)]
    print("第三部分耗时{}".format(time.time()-t3))

    plt.figure(figsize=(6, 6))
    plt.imshow(evo,cmap='hot',origin="lower",aspect="auto")  #origin可以改变从下到上排列，则无需翻转evo_201
    plt.xticks(np.arange(0,601,300),np.arange(-100,200,100))
    plt.xlabel("lattice site n")
    plt.ylabel("propagation distance κz")

    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(propa_κz,sigma_z,"ro",markersize=3)
    plt.xlabel("propagation distance κz")
    plt.ylim(0,80)
    plt.ylabel("$\sigma(z)$")
    plt.show() 
    


