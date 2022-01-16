import math
import numpy as np
from scipy import interpolate
import matplotlib.pylab as plt
from scipy.linalg import kron
import markdown as md
from scipy.fftpack import fft,ifft

spin_up = np.matrix([[1, 0]]).T
spin_down = np.matrix([[0, 1]]).T
# bit[0] = |0>, bit[1] = |1>
bit = [spin_up, spin_down]
colors = ["cornflowerblue", "crimson", "lightseagreen", "gold", "lightskyblue",'olive']

def max_value(x,y,level=0):
    pos,value=[],[]
    for i in range(1,np.size(x)-1):
        if y[i]>=y[i-1] and y[i]>=y[i+1] and x[i]>=level:
            pos.append(x[i])
            value.append(y[i])
    return pos,value

def min_value(x,y,level=0):
    pos,value=[],[]
    for i in range(1,np.size(x)-1):
        if y[i]<=y[i-1] and y[i]<=y[i+1] and x[i]>=level:
            pos.append(x[i])
            value.append(y[i])
    return pos,value

def basis(string='00010'):
    '''string: the qubits sequence'''
    res = np.array([[1]])
    # 从最后一位开始往前数，做直积
    for idx in string[::-1]:
        res = kron(bit[int(idx)], res)    
    return np.matrix(res)

def hilbert_space(nbit=5):
    nspace = 2**nbit
    for i in range(nspace):
        #bin(7) = 0b100
        binary = bin(i)[2:]
        nzeros = nbit - len(binary)
        yield '0'*nzeros + binary 


def Hadamard(A=[0,1]):
    a=(1/np.sqrt(2))*(A[0]+A[1])
    b=(1/np.sqrt(2))*(A[0]-A[1])
    return [a,b]

def wave_func(coef=[], seqs=[]):
    '''返回由振幅和几个Qubit序列表示的叠加态波函数，
       sum_i coef_i |psi_i> '''
    res = 0
    for i, a in enumerate(coef):
        res += a * basis(seqs[i])
    return np.matrix(res)

def project(wave_func, direction):
    '''<Psi | phi_i> to get the amplitude '''
    return wave_func.H * direction

def decompose(wave_func):
    '''将叠加态波函数分解'''
    nbit = int(np.log2(len(wave_func)))
    amplitudes = []
    direct_str = []
    for seq in hilbert_space(nbit):
        direct = basis(seq)
        amp = project(wave_func, direct).A1[0]
        if np.linalg.norm(amp) != 0:
            amplitudes.append(amp)
            direct_str.append(seq)
    return amplitudes, direct_str


def print_wf(wf):
    coef, seqs = decompose(wf)
    '''
    latex = ""
    for i, seq in enumerate(seqs):
        latex += r"%s$|%s\rangle$"%(coef[i], seq)
        if i != len(seqs) - 1:
            latex += "+"
            '''
    print(coef)
    print(seqs)
    return [coef,seqs]
def Hadamard_matrix(ksi=0,theta=np.pi/4):
    H=[[np.exp(1j*ksi)*np.cos(theta),np.exp(1j*ksi)*np.sin(theta)],\
       [np.exp(1j*ksi)*np.sin(theta),-np.exp(-1j*ksi)*np.cos(theta)]]
    return np.matrix(H)

def one_D_transfer_matrix(N):
    n=int((np.log(2*N)-1)/np.log(2))+1
    Bits=[]
    for bi in hilbert_space(n):
        Bits.append(basis(bi))
    R_transfer_matrix=np.zeros(shape=(2**n,2**n))
    L_transfer_matrix=np.zeros(shape=(2**n,2**n))
    for i in range(1,2**n-1):
        R_transfer_matrix+=np.matrix(Bits[i+1]*Bits[i].T)
        L_transfer_matrix+=np.matrix(Bits[i-1]*Bits[i].T)
    R_transfer_matrix+=np.matrix(Bits[1]*Bits[0].T)
    L_transfer_matrix+=np.matrix(Bits[2**n-2]*Bits[2**n-1].T)
    return np.matrix(R_transfer_matrix),np.matrix(L_transfer_matrix)

def Graph_transfer_matrix(N):
    n=int(np.log(N-1)/np.log(2))+1
    Bits=[]
    for bi in hilbert_space(n):
        Bits.append(basis(bi))
    R_transfer_matrix=np.zeros(shape=(2**n,2**n))
    L_transfer_matrix=np.zeros(shape=(2**n,2**n))
    for i in range(N):
        for j in range(N):
            if j==i+1:
                R_transfer_matrix+=np.matrix(Bits[i]*Bits[j].T)
            if j==i-1:
                L_transfer_matrix+=np.matrix(Bits[i]*Bits[j].T)
    R_transfer_matrix+=np.matrix(Bits[N-1]*Bits[0].T)
    L_transfer_matrix+=np.matrix(Bits[0]*Bits[N-1].T)
    return np.matrix(R_transfer_matrix),np.matrix(L_transfer_matrix)

def one_D_transfer_markov_matrix(N):
    M_transfer_matrix=np.zeros(shape=(N,N))
    for i in range(N):
        for j in range(N):
            if j==i+1 or j==i-1:
                M_transfer_matrix[i,j]=0.5
    M_transfer_matrix[0,N-1],M_transfer_matrix[N-1,0]=0.5,0.5
    return np.matrix(M_transfer_matrix)

def fft_process(x):
    fft_x=fft(x)                          
    N=np.size(x)
    indices=np.arange(int(N/4))/N               
    fft_x_new=np.abs(fft_x)                
    return indices,fft_x_new[range(int(N/4))]

def poly_graph(N):
    angle=(N-2)/N*np.pi
    x,y=[0,np.cos(-angle/2)],[0,np.sin(-angle/2)]
    for i in range(2,N):
        x.append(x[i-1]+np.cos((-1/2)*angle+(i-1)*(np.pi-angle)))
        y.append(y[i-1]+np.sin((-1/2)*angle+(i-1)*(np.pi-angle)))
    x.append(0)
    y.append(0)  
    plt.plot(x,y,color=colors[2])
    plt.axis('equal')
    plt.grid()
    plt.title('Configuration')
    for i in range(N):
        text_temp=str(i)
        plt.text(x[i]+0.03,y[i]+0.03,text_temp,fontsize=15)