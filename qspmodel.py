import numpy as np
import scipy.optimize as op
from itertools import permutations
from scipy.integrate import odeint
from scipy.integrate import solve_ivp


def concatenate_variables(A, B, pos_A):
    # concatenates A and B putting A in pos_A places and B in the remaining
    # assumes pos_A is boolean with length being sum of lengths of A and B
    C=np.empty(len(pos_A))
    C[pos_A]=A
    C[~pos_A]=B
    return C
    
# -----
# Class that works as a vector function with step-functions as components

# Standrad basis vector from index and vector length
sbv = lambda index, length: np.array([1.0 if i == index-1 else 0.0 for i in range(length)])

class step_vector:
    # Vector step function
    def __init__(self, length, indices=[], intervals=[], values=[]):
        # length    - the size of the vector
        # indices   - list of natural indices (from 1 to length) of non-zero components to be defined as step functions
        #             indices are allowed to repeat if necessary for different intervals
        # intervals - list of time-intervals where the step function is positive.  
        #             intervals include starting point and don't include end point.
        # values    - height of the step function for each index 
        self.length=length
        self.steps=len(indices)
        self.indices=np.copy(indices)
        self.intervals=np.copy(intervals)
        self.values=np.copy(values)
    def __call__(self, t):
        return sum([self.values[i]*sbv(self.indices[i], self.length) if (t>=self.intervals[i,0])&(t<self.intervals[i,1]) else 0.0 for i in range(self.steps)], np.zeros(self.length))


#-------------    

def Sensitivity_function(t,x,QSP,r):
    # time derivative for sensitivity ODE
    nparam=QSP.qspcore.nparam
    nvar=QSP.qspcore.nvar
    x=x.reshape(nparam+1,nvar)
    dxdt=np.empty(x.shape)
    dxdt[0]=QSP(t,x[0])+r(t)
    dxdt[1:]=np.dot(QSP.Ju(t,x[0]),x[1:].T).T+ QSP.Jp(t,x[0])
    return dxdt.flatten()



# Object class that defines the functions for the appropriate QSP Model
class OS_QSP_Functions(object):
    def __init__(self,SSrestrictions=np.ones(23)):
        self.nparam=55
        self.nvar=14
        self.variable_names=['Naive Macrophages  ($M_N$)', 'Macrophages  ($M$)', 'Naive T-cells  ($T_N$)', 'helper T-cells  ($T_h$)', 'Treg-cells  ($T_r$)', 
                             'cytotoxic cells  ($T_c$)', 'Naive Dendritic cells  ($D_N$)', 'Dendritic cells  ($D$)', 'Cancer cells  ($C$)', 'Necrotic cells  ($N$)', 
                             'IFN-$\gamma$  ($I_\gamma$)', '$\mu_1$', '$\mu_2$', 'HMGB1  ($H$)']
        self.SSscale = SSrestrictions

    def __call__(self,t,x,par):
        # ODE right-hand side
        return np.array([par[45] - par[24]*x[0] - par[49]*x[0]*(par[0]*x[10] + par[1]*x[11]), 
                         (-par[25])*x[1] + x[0]*(par[0]*x[10] + par[1]*x[11]), 
                         par[46] - par[26]*x[2] - par[50]*x[2]*(par[2]*x[1] + par[3]*x[7]) - par[52]*x[2]*(par[6]*x[1] + par[5]*x[3] + par[7]*x[7]) - par[4]*par[51]*x[2]*x[11], 
                         x[2]*(par[2]*x[1] + par[3]*x[7]) - x[3]*(par[29] + par[27]*x[4] + par[28]*x[11]), 
                         (-par[30])*x[4] + par[4]*x[2]*x[11], 
                         x[2]*(par[6]*x[1] + par[5]*x[3] + par[7]*x[7]) - x[5]*(par[33] + par[31]*x[4] + par[32]*x[11]), 
                         par[47] - par[34]*x[6] - par[53]*x[6]*(par[8]*x[8] + par[9]*x[13]), 
                         (-x[7])*(par[36] + par[35]*x[8]) + x[6]*(par[8]*x[8] + par[9]*x[13]), 
                         (-x[8])*(par[39] + par[37]*x[5] + par[38]*x[10]) + x[8]*(1 - x[8]/par[48])*(2*par[10] + par[11]*x[11] + par[12]*x[12]), 
                         (-par[40])*x[9] + par[54]*x[8]*(par[39] + par[37]*x[5] + par[38]*x[10]), 
                         par[13]*x[3] + par[14]*x[5] - par[41]*x[10], 
                         par[16]*x[1] + par[15]*x[3] + par[17]*x[8] - par[42]*x[11], 
                         par[19]*x[1] + par[18]*x[3] + par[20]*x[8] - par[43]*x[12], 
                         par[21]*x[1] + par[22]*x[7] + par[23]*x[9] - par[44]*x[13]])
    
    
    def Ju(self,t,x,par):
        # Jacobian with respect to variables
        return np.array([[-par[24] - par[49]*(par[0]*x[10] + par[1]*x[11]), 0, 0, 0, 0, 0, 0, 0, 0, 0, (-par[0])*par[49]*x[0], 
                          (-par[1])*par[49]*x[0], 0, 0], 
                         [par[0]*x[10] + par[1]*x[11], -par[25], 0, 0, 0, 0, 0, 0, 0, 0, par[0]*x[0], par[1]*x[0], 0, 0], 
                         [0, (-par[2])*par[50]*x[2] - par[6]*par[52]*x[2], -par[26] - par[50]*(par[2]*x[1] + par[3]*x[7]) - 
                          par[52]*(par[6]*x[1] + par[5]*x[3] + par[7]*x[7]) - par[4]*par[51]*x[11], (-par[5])*par[52]*x[2], 
                          0, 0, 0, (-par[3])*par[50]*x[2] - par[7]*par[52]*x[2], 0, 0, 0, (-par[4])*par[51]*x[2], 0, 0], 
                         [0, par[2]*x[2], par[2]*x[1] + par[3]*x[7], -par[29] - par[27]*x[4] - par[28]*x[11], (-par[27])*x[3], 
                          0, 0, par[3]*x[2], 0, 0, 0, (-par[28])*x[3], 0, 0], 
                         [0, 0, par[4]*x[11], 0, -par[30], 0, 0, 0, 0, 0, 0, par[4]*x[2], 0, 0], 
                         [0, par[6]*x[2], par[6]*x[1] + par[5]*x[3] + par[7]*x[7], par[5]*x[2], (-par[31])*x[5], -par[33] - 
                          par[31]*x[4] - par[32]*x[11], 0, par[7]*x[2], 0, 0, 0, (-par[32])*x[5], 0, 0], 
                         [0, 0, 0, 0, 0, 0, -par[34] - par[53]*(par[8]*x[8] + par[9]*x[13]), 0, (-par[8])*par[53]*x[6], 
                          0, 0, 0, 0, (-par[9])*par[53]*x[6]], 
                         [0, 0, 0, 0, 0, 0, par[8]*x[8] + par[9]*x[13], -par[36] - par[35]*x[8], par[8]*x[6] - par[35]*x[7], 
                          0, 0, 0, 0, par[9]*x[6]], 
                         [0, 0, 0, 0, 0, (-par[37])*x[8], 0, 0, -par[39] - par[37]*x[5] - par[38]*x[10] - 
                          (x[8]*(2*par[10] + par[11]*x[11] + par[12]*x[12]))/par[48] + 
                          (1 - x[8]/par[48])*(2*par[10] + par[11]*x[11] + par[12]*x[12]), 0, (-par[38])*x[8], 
                          par[11]*x[8]*(1 - x[8]/par[48]), par[12]*x[8]*(1 - x[8]/par[48]), 0], 
                         [0, 0, 0, 0, 0, par[37]*par[54]*x[8], 0, 0, par[54]*(par[39] + par[37]*x[5] + par[38]*x[10]), 
                          -par[40], par[38]*par[54]*x[8], 0, 0, 0], 
                         [0, 0, 0, par[13], 0, par[14], 0, 0, 0, 0, -par[41], 0, 0, 0], 
                         [0, par[16], 0, par[15], 0, 0, 0, 0, par[17], 0, 0, -par[42], 0, 0], 
                         [0, par[19], 0, par[18], 0, 0, 0, 0, par[20], 0, 0, 0, -par[43], 0], 
                         [0, par[21], 0, 0, 0, 0, 0, par[22], 0, par[23], 0, 0, 0, -par[44]]])
    

    def Jp(self,t,x,par):
        # Jacobian with respect to the parameters
        return np.array([[(-par[49])*x[0]*x[10], x[0]*x[10], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [(-par[49])*x[0]*x[11], x[0]*x[11], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, (-par[50])*x[1]*x[2], x[1]*x[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, (-par[50])*x[2]*x[7], x[2]*x[7], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, (-par[51])*x[2]*x[11], 0, x[2]*x[11], 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, (-par[52])*x[2]*x[3], 0, 0, x[2]*x[3], 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, (-par[52])*x[1]*x[2], 0, 0, x[1]*x[2], 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, (-par[52])*x[2]*x[7], 0, 0, x[2]*x[7], 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, (-par[53])*x[6]*x[8], x[6]*x[8], 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, (-par[53])*x[6]*x[13], x[6]*x[13], 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 2*x[8]*(1 - x[8]/par[48]), 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, x[8]*(1 - x[8]/par[48])*x[11], 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, x[8]*(1 - x[8]/par[48])*x[12], 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[3], 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[5], 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[3], 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[1], 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[8], 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[3], 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[1], 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[8], 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[1]], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[7]], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[9]], 
                         [-x[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, -x[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, -x[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, (-x[3])*x[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, (-x[3])*x[11], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, -x[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, -x[4], 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, (-x[4])*x[5], 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, (-x[5])*x[11], 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, -x[5], 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, -x[6], 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, (-x[7])*x[8], 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, -x[7], 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, (-x[5])*x[8], par[54]*x[5]*x[8], 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, (-x[8])*x[10], par[54]*x[8]*x[10], 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, -x[8], par[54]*x[8], 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, -x[9], 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[10], 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[11], 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[12], 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[13]], 
                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, (x[8]**2*(2*par[10] + par[11]*x[11] + par[12]*x[12]))/par[48]**2, 0, 0, 0, 0, 0], 
                         [(-x[0])*(par[0]*x[10] + par[1]*x[11]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, (-x[2])*(par[2]*x[1] + par[3]*x[7]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, (-par[4])*x[2]*x[11], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, (-x[2])*(par[6]*x[1] + par[5]*x[3] + par[7]*x[7]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, (-x[6])*(par[8]*x[8] + par[9]*x[13]), 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, x[8]*(par[39] + par[37]*x[5] + par[38]*x[10]), 0, 0, 0, 0]])

    
    def SS_system(self,par,frac,meanvals):
        # compute the system and restrictions with non-dimensional steady states at 1 pre-defined rates
        # meanvals as given as [M1 M2 M Th Tr Tc D C N Ig mu1 mu2 H]
        x=np.ones(self.nvar);
        # rates acquired from bio research [delMn delM delTn delTh delTr delTc delDn delD delIg delmu1 delmu2 delH]
        globvals=np.array([0.693, 0.015, 0.00042, 0.231, 0.063, 0.406, 1.664, 0.277, 33.27, 487.48, 5.15, 58.7])

        return np.array([par[45] - par[24]*x[0] - par[49]*x[0]*(par[0]*x[10] + par[1]*x[11]), 
                         (-par[25])*x[1] + x[0]*(par[0]*x[10] + par[1]*x[11]), 
                         par[46] - par[26]*x[2] - par[50]*x[2]*(par[2]*x[1] + par[3]*x[7]) - par[52]*x[2]*(par[6]*x[1] + par[5]*x[3] + par[7]*x[7]) - par[4]*par[51]*x[2]*x[11], 
                         x[2]*(par[2]*x[1] + par[3]*x[7]) - x[3]*(par[29] + par[27]*x[4] + par[28]*x[11]), 
                         (-par[30])*x[4] + par[4]*x[2]*x[11], 
                         x[2]*(par[6]*x[1] + par[5]*x[3] + par[7]*x[7]) - x[5]*(par[33] + par[31]*x[4] + par[32]*x[11]), 
                         par[47] - par[34]*x[6] - par[53]*x[6]*(par[8]*x[8] + par[9]*x[13]), 
                         (-x[7])*(par[36] + par[35]*x[8]) + x[6]*(par[8]*x[8] + par[9]*x[13]), 
                         (-x[8])*(par[39] + par[37]*x[5] + par[38]*x[10]) + x[8]*(1 - x[8]/par[48])*(2*par[10] + par[11]*x[11] + par[12]*x[12]), 
                         (-par[40])*x[9] + par[54]*x[8]*(par[39] + par[37]*x[5] + par[38]*x[10]), 
                         par[13]*x[3] + par[14]*x[5] - par[41]*x[10], 
                         par[16]*x[1] + par[15]*x[3] + par[17]*x[8] - par[42]*x[11], 
                         par[19]*x[1] + par[18]*x[3] + par[20]*x[8] - par[43]*x[12], 
                         par[21]*x[1] + par[22]*x[7] + par[23]*x[9] - par[44]*x[13],
                         # cancer growth rate
                         self.SSscale[0]*par[10] - 0.01662,
                         # assumptions (21)
                         self.SSscale[1]*par[10] - 20*par[12]*meanvals[11]/frac[12],
                         self.SSscale[2]*par[10] - 40*par[11]*meanvals[10]/frac[11],
                         self.SSscale[3]*par[38]*meanvals[9]/frac[10] - 10*par[39],
                         self.SSscale[4]*par[37]*meanvals[5]/frac[5] - 20*par[39],
                         self.SSscale[5]*par[0]*(meanvals[9]/frac[10])*meanvals[1] - par[1]*(meanvals[10]/frac[11])*meanvals[0],
                         self.SSscale[6]*par[3]*meanvals[6]/frac[7] - 200*par[2]*meanvals[2]/frac[1],
                         self.SSscale[7]*par[28]*meanvals[10]/frac[11] - 20*par[29],
                         self.SSscale[8]*par[27]*meanvals[4]/frac[4] - 20*par[29],
                         self.SSscale[9]*par[7]*meanvals[6]/frac[7] - 2*par[5]*meanvals[3]/frac[3],
                         self.SSscale[10]*par[7]*meanvals[6]/frac[7] - 4*par[6]*meanvals[2]/frac[1],
                         self.SSscale[11]*par[32]*meanvals[10]/frac[11] - 20*par[33],
                         self.SSscale[12]*par[31]*meanvals[4]/frac[4] - 20*par[33],
                         self.SSscale[13]*par[9]*meanvals[12]/frac[13] - 2*par[8]*meanvals[7]/frac[8],
                         self.SSscale[14]*par[35]*meanvals[7]/frac[8] - par[36],
                         self.SSscale[15]*par[14]*meanvals[5]/frac[5] - 4*par[13]*meanvals[3]/frac[3],
                         self.SSscale[16]*par[15]*meanvals[3]/frac[3] - par[16]*meanvals[2]/frac[1],
                         self.SSscale[17]*par[15]*meanvals[3]/frac[3] - par[17]*meanvals[7]/frac[8],
                         self.SSscale[18]*par[19]*meanvals[2]/frac[1] - par[20]*meanvals[7]/frac[8],
                         self.SSscale[19]*par[19]*meanvals[2]/frac[1] - 2*par[18]*meanvals[3]/frac[3],
                         self.SSscale[20]*par[23]*meanvals[8]/frac[9] - 10*par[21]*meanvals[2]/frac[1],
                         self.SSscale[21]*par[23]*meanvals[8]/frac[9] - 20*par[22]*meanvals[6]/frac[7],
                         # decay rates (12)
                         par[24] - globvals[0],
                         par[25] - globvals[1],
                         par[26] - globvals[2],
                         par[29] - globvals[3],
                         par[30] - globvals[4],
                         par[33] - globvals[5],
                         par[34] - globvals[6],
                         par[36] - globvals[7],
                         par[41] - globvals[8],
                         par[42] - globvals[9],
                         par[43] - globvals[10],
                         par[44] - globvals[11],
                         # alphas (5)
                         par[49] - frac[1]/frac[0],
                         par[50] - frac[3]/frac[2],
                         par[51] - frac[4]/frac[2],
                         par[52] - frac[5]/frac[2],
                         par[53] - frac[7]/frac[6],
                         # C0 and alpha_NC
                         par[48] - 2,
                         self.SSscale[22]*0.75*frac[8]/frac[9] - par[54]])



class QSP:
    def __init__(self,parameters,qspcore=OS_QSP_Functions()):
        self.qspcore=qspcore
        self.par=parameters;
    def set_parameters(self,parameters):
        self.par=parameters;
    def steady_state(self):
        # compute steady state with current parameters
        IC=np.ones(self.qspcore.nvar);
        return op.fsolve((lambda x: self.qspcore(0,x,self.par)),IC,fprime=(lambda x: self.qspcore.Ju(0,x,self.par)),xtol=1e-7,maxfev=10000)
        # return op.root((lambda x,QSP: QSP(0,x,self.par)),IC,args=(self.qspcore,), method='hybr')
    def Sensitivity(self,method='steady',t=None,IC=None,params=None,variables=None,form='absolute', inhomogeneity=None, jumps=False, events=None, u_after_treatment=None):
        # Sensitivity matrix
        # method: (default) 'steady' - steady state sensitivity
                # 'time' - time-integrated sensitivity
                        # requires time array t and initial conditions IC
                        # 'time-full' would return full time-dependent data, otherwise returns average
                # 'split' - steady state sensitivity with respect to chosen parameters
                        # requires initiate_parameter_split to have been run
                        # requires params argument with updated parameters
        # variables: optional argument for sensitivity of specific variables.
        # form: sensitivity type
                # 'absolute' (default) - du/dp
                # 'relative' - (du/dp)*(p/u) 
                # relative will not work if the variable can be zero
        # inhomogeneity - optional function of t to be added to the rhs of ode
                # for time-dependent sensitivity only
        # jumps - indicates whether inhomogeneity has discontinuities
        # events - callable function of t that has roots at points of discontinuity
        #           if left default when jumps=True, assumes the use of step_vector class
        if variables is None:
            variables=np.arange(self.qspcore.nvar)
        if method[:4]=='time':
            if inhomogeneity is None:
                r=lambda t: 0
                if jumps:
                    raise Exception('error: cannot handle jumps without inhomogeneity')
                    return None
            else:
                r=inhomogeneity
                if jumps and (events is None):
                    jump_indicator=lambda t, x, Q, r: np.prod(t-r.intervals.flatten())
                elif jumps:
                    jump_indicator=lambda t, x, Q, r: events(t)
                    
            if IC is None:
                raise Exception('Error: Need initial conditions for time integration. Set IC=')
                return None
            if t is None:
                raise Exception('Error: Need time values for time integration. Set t=')
                return None
            
            nparam=self.qspcore.nparam
            nvar=self.qspcore.nvar
            N=len(t)
            initial=np.zeros((nparam+1,nvar));
            initial[0]=IC
            if jumps:
                sol=solve_ivp(Sensitivity_function, (min(t), max(t)), initial.flatten(), 
                                t_eval=t, args=(self, r), events=jump_indicator)
                result=sol.y.T.reshape(N,nparam+1,nvar)
            else:
                result=odeint(Sensitivity_function, initial.flatten(), t, args=(self, r), tfirst=True).reshape(N,nparam+1,nvar)
            u=result[:,0,:]
            if form=='relative':
                S=result[:,1:,variables]*self.par[np.newaxis,:,np.newaxis]/u[:,np.newaxis,variables]
            else:
                S=result[:,1:,variables]
            if method=='time-full':
                return S
            else:
                return np.mean(S, axis=0)
        elif method=='split':
            if not hasattr(self,'variable_par'):
                raise Exception('error: parameter splitting is not set. use "initiate_parameter_split" method')
                return None
            if params is None:
                raise Exception('error: Need parameter values for split sensitivity. Set params=')
                return None
            elif len(params)!=sum(self.variable_par):
                raise Exception('error: wrong number of parameters given')
                return None
            
            if IC is None:
                IC=np.ones(self.qspcore.nvar);
            par=np.copy(self.par)
            par[self.variable_par]=np.copy(params)
            
            u=op.fsolve((lambda x: self.qspcore(0,x,par)),IC,fprime=(lambda x: self.qspcore.Ju(0,x,par)),xtol=1e-7,maxfev=10000)
            S=-np.dot(self.qspcore.Jp(0,u,self.par),np.linalg.inv(self.qspcore.Ju(0,u,self.par).T))[self.variable_par,variables]
            
            if form=='relative':
                return S*par[:,np.newaxis]/u[np.newaxis,variables]
            else:
                return S
        else:
            if method=='after_treatment':
                u = u_after_treatment
            else: u=self.steady_state()
            S=-np.dot(self.qspcore.Jp(0,u,self.par),np.linalg.inv(self.qspcore.Ju(0,u,self.par).T))[:,variables]
            
            if form=='relative':
                return S*self.par[:,np.newaxis]/u[np.newaxis,variables]
            else:
                return S
        
    def __call__(self,t,x):
        return self.qspcore(t,x,self.par)
    def Ju(self,t,x):
        return self.qspcore.Ju(t,x,self.par)
    def Jp(self,t,x):
        return self.qspcore.Jp(t,x,self.par)
    def variable_names(self):return self.qspcore.variable_names
    def parameter_names(self):return self.qspcore.parameter_names
    
    def solve_ode(self, t, IC, method='default', inhomogeneity=None, jumps=False, events=None):
        # Solve ode system with either default 1e4 time steps or given time discretization
        # t - time: for 'default' needs start and end time
        #           for 'given' needs full array of time discretization points
        # IC - initial conditions
        # method: 'default' - given interval divided by 10000 time steps 
        #         'given' - given time discretization
        # inhomogeneity - optional function of t to be added to the rhs of ode
        # jumps - indicates whether inhomogeneity has discontinuities
        # events - callable function of t that has roots at points of discontinuity
        #           if left default when jumps=True, assumes the use of step_vector class
        if inhomogeneity is None:
            r=lambda t: 0
            if jumps:
                raise Exception('error: cannot handle jumps without inhomogeneity')
                return None
        else:
            r=inhomogeneity
            if jumps and (events is None):
                jump_indicator=lambda t, x: np.prod(t-r.intervals.flatten())
            elif jumps:
                jump_indicator=lambda t, x: events(t)
        if (method=='given') and jumps: 
            sol = solve_ivp((lambda t, x: (self.qspcore(t,x,self.par)+r(t))), 
                            (min(t), max(t)), IC, t_eval=t, events=jump_indicator)
            return sol.y.T, t
        elif jumps: 
            sol = solve_ivp((lambda t, x: (self.qspcore(t,x,self.par)+r(t))), 
                            (min(t), max(t)), IC, events=jump_indicator)
            return sol.y.T, sol.t
        elif method=='given': 
            return odeint((lambda t,x: (self.qspcore(t,x,self.par)+r(t))), IC, t, 
                            Dfun=(lambda t,x: self.qspcore.Ju(t,x,self.par)), tfirst=True), t
        else: 
            return odeint((lambda t,x: (self.qspcore(t,x,self.par)+r(t))), IC, np.linspace(min(t), max(t), 10001), 
                            Dfun=(lambda t,x: self.qspcore.Ju(t,x,self.par)), tfirst=True), np.linspace(min(t), max(t), 10001)
        
    
    @classmethod
    def from_cell_data(class_object, fracs, meanvals, qspcore=OS_QSP_Functions()):
        params=op.fsolve((lambda par,frac,meanvals: qspcore.SS_system(par,frac,meanvals)),np.ones(qspcore.nparam),
                         args=(fracs, meanvals))
        return class_object(params,qspcore)
    @classmethod    
    def from_data(class_object, data, qspcore=OS_QSP_Functions()):
        return class_object(qspcore.parameters_from_assumptions(data),qspcore)
