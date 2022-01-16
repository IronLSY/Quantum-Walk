import math
import numpy as np
from scipy import interpolate
import matplotlib.pylab as plt
from scipy.linalg import kron
import markdown as md
from scipy.fftpack import fft,ifft
from PIL import Image
from Linear_Auxiliary import basis,hilbert_space,Hadamard,wave_func,project,decompose,\
     print_wf,Hadamard_matrix,bit,one_D_transfer_matrix,Graph_transfer_matrix,poly_graph,\
     fft_process,colors,one_D_transfer_markov_matrix,min_value,max_value
# 量子直线/闭环行走
def Random_Walk(initial_pos,N,ksi=0,theta=np.pi/4,initial_coin=\
                      [1/np.sqrt(2),1j/np.sqrt(2)],plot_op=False,poly_op=False,Shape=1):
    H=Hadamard_matrix(ksi,theta)
       
    Indices=[]
    if poly_op==False:
        R_transfer_matrix,L_transfer_matrix=one_D_transfer_matrix(N)
        n=len(R_transfer_matrix)
        for i in range(int(-n/2),int((n+1)/2)):
            Indices.append(i)
    else:
        R_transfer_matrix,L_transfer_matrix=Graph_transfer_matrix(Shape)
        n=len(R_transfer_matrix)
        for i in range(n):
            Indices.append(i)
    Bits=[]
    for bi in hilbert_space(int(np.log(n)/np.log(2))):
        Bits.append(basis(bi))
    initial_pos_bit=Bits[Indices.index(initial_pos)]
    initial_coin_bit=np.matrix(initial_coin[0]*bit[0]+initial_coin[1]*bit[1])
    Dyadic_bit=np.matrix(kron(initial_coin_bit,initial_pos_bit))
    Action_1=np.matrix(kron(H,np.identity(n)))
    Action_2=np.matrix(kron(bit[0]*bit[0].T,R_transfer_matrix))+np.matrix(kron(bit[1]*bit[1].T,L_transfer_matrix))
    Action_3=np.matrix(kron(bit[0].T,np.identity(n)))
    Action_4=np.matrix(kron(bit[1].T,np.identity(n)))
    for i in range(N):
        Dyadic_bit=Action_1*Dyadic_bit
        Dyadic_bit=Action_2*Dyadic_bit
    final_bit_1=Action_3*Dyadic_bit
    final_bit_2=Action_4*Dyadic_bit
    magn=[]
    for i in range(len(final_bit_1)):
        magn.append((abs(final_bit_1[i,0]))**2+(abs(final_bit_2[i,0]))**2)
    if plot_op==True:
        plt.plot(Indices,magn)
        plt.show()
    return Indices,magn
# 经典直线/闭环行走
def Random_Walk_Markov(Steps,Shape=1,poly_op=False,plot_op=False):
#     默认Shape=1，为一维直线行走
    Indices=[]
    if poly_op==False:
        N=2*Steps+2
        for i in range(int(-N/2),int((N+1)/2)):
            Indices.append(i)
        state=np.zeros(shape=(1,N))
        state[0,Indices.index(0)]=1
    else:
        N=Shape
        for i in range(N):
            Indices.append(i)
        state=np.zeros(shape=(1,N))
        state[0,0]=1
    Markov=one_D_transfer_markov_matrix(N)
    state=state.T
    for i in range(Steps):
        state=Markov*state 
    if plot_op==True:
        plt.plot(Indices,state.tolist())
        plt.show()
    return Indices,(state.T).tolist()
# 量子闭环行走回归分析
def poly_walk_statis_show(N,Steps):
#     R,L=Graph_transfer_matrix(N)
    if N>10:
        row_number=3
    else:
        row_number=2
    col_number=int((N+1)/row_number)+1
    plt.subplot(row_number,col_number,1)
    poly_graph(N)
    prob=[]
    for i in range(N):
        prob.append([])
    for i in range(Steps):
        _,y=Random_Walk(0,i,poly_op=True,Shape=N)
        for j in range(N):
            prob[j].append(y[j])
    for i in range(N):
        plt.subplot(row_number,col_number,i+2)
        plt.plot(range(Steps),prob[i])
        plt.grid()
        plt.title(str(i)+'th node')
#         plt.xlabel('step')
    fft_step_0,fft_prob_0=fft_process(prob[0])
    g=interpolate.interp1d(fft_step_0,fft_prob_0,kind='cubic')
    fft_step_0_new=np.linspace(0,max(fft_step_0),10*len(fft_step_0))
    fft_prob_0_new=g(fft_step_0_new)
    plt.subplot(row_number,col_number,N+2)
    plt.plot(fft_step_0_new,fft_prob_0_new,color=colors[1])
    plt.title('FFT analysis')
    plt.xlabel('Frequency')
    plt.grid()
    peak_pos,peak_magn=max_value(fft_step_0_new,fft_prob_0_new)
    text_temp=str(peak_pos[peak_magn.index(max(peak_magn))])
    plt.text(peak_pos[peak_magn.index(max(peak_magn))]+0.03,max(peak_magn)+0.03,text_temp,fontsize=10)
    plt.show()

# 经典闭环行走统计
def poly_walk_statis_Markov_show(N,Steps):
    if N>10:
        row_number=3
    else:
        row_number=2
    col_number=int((N+1)/row_number)+1
    plt.subplot(row_number,col_number,1)
    poly_graph(N)
    prob=[]
    for i in range(N):
        prob.append([])
    for i in range(Steps):
        _,y=Random_Walk_Markov(i,Shape=N,poly_op=True)
        for j in range(N):
            prob[j].append(y[0][j])
    for i in range(N):
        plt.subplot(row_number,col_number,i+2)
        plt.plot(range(Steps),prob[i])
        plt.grid()
        plt.title(str(i)+'th node')
#         plt.xlabel('step')
    fft_step_0,fft_prob_0=fft_process(prob[0])
    g=interpolate.interp1d(fft_step_0,fft_prob_0,kind='cubic')
    fft_step_0_new=np.linspace(0,max(fft_step_0),10*len(fft_step_0))
    fft_prob_0_new=g(fft_step_0_new)
    plt.subplot(row_number,col_number,N+2)
    plt.plot(fft_step_0_new,fft_prob_0_new,color=colors[1])
    plt.title('FFT analysis')
    plt.xlabel('Frequency')
    plt.grid()
    peak_pos,peak_magn=max_value(fft_step_0_new,fft_prob_0_new)
    text_temp=str(peak_pos[peak_magn.index(max(peak_magn))])
    plt.text(peak_pos[peak_magn.index(max(peak_magn))]+0.03,max(peak_magn)+0.03,text_temp,fontsize=10)
    plt.show()