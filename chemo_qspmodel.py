'''
chemo_qspmodel: file with class containing functions and jacobians related to
            Investigating optimal chemotherapy options for osteosarcoma patients through a data-driven mathematical model*
OS_MAP_Functions: class containing functions and jacobians related to MAP treatment
OS_AP_Functions: class containing functions related to Doxorubicin and Cisplatin combination treatment
OS_MTX_Functions: class containing functions related to Methotrexate as a single agent treatment
'''
import numpy as np

class OS_MAP_Functions(object):
    def __init__(self,parameters=None):
        self.nparam=19
        self.nvar=17
        self.variable_names=['Naive Macrophages  ($M_N$)', 'Macrophages  ($M$)', 'Naive T-cells  ($T_N$)', 'helper T-cells  ($T_h$)', 'Treg-cells  ($T_r$)', 
                             'cytotoxic cells  ($T_c$)', 'Naive Dendritic cells  ($D_N$)', 'Dendritic cells  ($D$)', 'Cancer cells  ($C$)', 'Necrotic cells  ($N$)', 
                             'IFN-$\gamma$  ($I_\gamma$)', '$\mu_1$', '$\mu_2$', 'HMGB1  ($H$)', 'A1', 'A2', 'A3']
        
        self.parameter_names=['$\\beta_{1}$', '$\\beta_{2}$', '$\\beta_{3}$', '$K_{M_N}$',
                              '$K_{M}$', '$K_{T_N}$', '$K_{T_h}$', '$K_{T_r}$', '$K_{T_c}$', '$K_{D_N}$', '$K_{D}$', '$K_{C}$', '$\\delta_{A_1}$',
                              '$\\delta_{A_2}$', '$\\delta_{A_3}$', '$\\alpha_{NCA}$', '$\\delta_{C T_c A_3}$', '$f$', '$a$']
        self.par0 = parameters

    def __call__(self,t,x,par):
        # ODE right-hand side
        tau = min(t, par[17]*par[18])
        x=x*(x>0) #Q
        return np.array([self.par0[45] - ((1 - np.exp((-par[1])*x[15]))*par[3] + (1 - np.exp((-par[2])*x[16]))*par[3] + (1 - np.exp((-par[0])*x[14]))*par[3]*
                        (par[17] + 1/par[18] - tau/par[18]))*x[0] - self.par0[24]*x[0] - self.par0[49]*x[0]*(self.par0[0]*x[10] + self.par0[1]*x[11]), 
                        -(((1 - np.exp((-par[1])*x[15]))*par[4] + (1 - np.exp((-par[2])*x[16]))*par[4] + (1 - np.exp((-par[0])*x[14]))*par[4]*
                        (par[17] + 1/par[18] - tau/par[18]))*x[1]) - self.par0[25]*x[1] + x[0]*(self.par0[0]*x[10] + self.par0[1]*x[11]), 
                        self.par0[46] - ((1 - np.exp((-par[1])*x[15]))*par[5] + (1 - np.exp((-par[2])*x[16]))*par[5] + (1 - np.exp((-par[0])*x[14]))*par[5]*
                        (par[17] + 1/par[18] - tau/par[18]))*x[2] - self.par0[26]*x[2] - self.par0[50]*x[2]*(self.par0[2]*x[1] + self.par0[3]*x[7]) - 
                        self.par0[52]*x[2]*(self.par0[6]*x[1] + self.par0[5]*x[3] + self.par0[7]*x[7]) - self.par0[4]*self.par0[51]*x[2]*x[11], 
                        -(((1 - np.exp((-par[1])*x[15]))*par[6] + (1 - np.exp((-par[2])*x[16]))*par[6] + (1 - np.exp((-par[0])*x[14]))*par[6]*
                        (par[17] + 1/par[18] - tau/par[18]))*x[3]) + x[2]*(self.par0[2]*x[1] + self.par0[3]*x[7]) - x[3]*(self.par0[29] + self.par0[27]*x[4] + self.par0[28]*x[11]), 
                        -(((1 - np.exp((-par[1])*x[15]))*par[7] + (1 - np.exp((-par[2])*x[16]))*par[7] + (1 - np.exp((-par[0])*x[14]))*par[7]*
                        (par[17] + 1/par[18] - tau/par[18]))*x[4]) - self.par0[30]*x[4] + self.par0[4]*x[2]*x[11], 
                        -(((1 - np.exp((-par[1])*x[15]))*par[8] + (1 - np.exp((-par[2])*x[16]))*par[8] + (1 - np.exp((-par[0])*x[14]))*par[8]*
                        (par[17] + 1/par[18] - tau/par[18]))*x[5]) + x[2]*(self.par0[6]*x[1] + self.par0[5]*x[3] + self.par0[7]*x[7]) - 
                        x[5]*(self.par0[33] + self.par0[31]*x[4] + self.par0[32]*x[11]), 
                        self.par0[47] - ((1 - np.exp((-par[1])*x[15]))*par[9] + (1 - np.exp((-par[2])*x[16]))*par[9] + (1 - np.exp((-par[0])*x[14]))*par[9]*
                        (par[17] + 1/par[18] - tau/par[18]))*x[6] - self.par0[34]*x[6] - self.par0[53]*x[6]*(self.par0[8]*x[8] + self.par0[9]*x[13]), 
                        -(((1 - np.exp((-par[1])*x[15]))*par[10] + (1 - np.exp((-par[2])*x[16]))*par[10] + (1 - np.exp((-par[0])*x[14]))*par[10]*
                        (par[17] + 1/par[18] - tau/par[18]))*x[7]) - x[7]*(self.par0[36] + self.par0[35]*x[8]) + x[6]*(self.par0[8]*x[8] + self.par0[9]*x[13]), 
                        -(((1 - np.exp((-par[1])*x[15]))*par[11] + (1 - np.exp((-par[2])*x[16]))*par[11] + (1 - np.exp((-par[0])*x[14]))*par[11]*
                        (par[17] + 1/par[18] - tau/par[18]))*x[8]) - x[8]*(self.par0[39] + (1 - np.exp((-par[2])*x[16]))*par[16]*x[5] + self.par0[37]*x[5] + 
                        self.par0[38]*x[10]) + x[8]*(1 - x[8]/self.par0[48])*(2*self.par0[10] + self.par0[11]*x[11] + self.par0[12]*x[12]), 
                        par[15]*((1 - np.exp((-par[1])*x[15]))*par[11] + (1 - np.exp((-par[2])*x[16]))*par[11] + (1 - np.exp((-par[0])*x[14]))*par[11]*
                        (par[17] + 1/par[18] - tau/par[18]))*x[8] - self.par0[40]*x[9] + self.par0[54]*x[8]*(self.par0[39] + (1 - np.exp((-par[2])*x[16]))*par[16]*x[5] + 
                        self.par0[37]*x[5] + self.par0[38]*x[10]), self.par0[13]*x[3] + self.par0[14]*x[5] - self.par0[41]*x[10], self.par0[16]*x[1] + self.par0[15]*x[3] + self.par0[17]*x[8] - 
                        self.par0[42]*x[11], self.par0[19]*x[1] + self.par0[18]*x[3] + self.par0[20]*x[8] - self.par0[43]*x[12], self.par0[21]*x[1] + self.par0[22]*x[7] + self.par0[23]*x[9] - 
                        self.par0[44]*x[13], (-par[12])*x[14], (-par[13])*x[15], (-par[14])*x[16]])

        

    def Ju(self,t,x,par):
        # Jacobian with respect to variables
        tau = min(t, par[17]*par[18])
        x=x*(x>0)
        return np.array([[-((1 - np.exp((-par[1])*x[15]))*par[3]) - (1 - np.exp((-par[2])*x[16]))*par[3] - (1 - np.exp((-par[0])*x[14]))*par[3]*
                         (par[17] + 1/par[18] - tau/par[18]) - self.par0[24] - self.par0[49]*(self.par0[0]*x[10] + self.par0[1]*x[11]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                         (-self.par0[0])*self.par0[49]*x[0], (-self.par0[1])*self.par0[49]*x[0], 0, 0, (-np.exp((-par[0])*x[14]))*par[0]*par[3]*(par[17] + 1/par[18] - tau/par[18])*x[0], 
                         (-np.exp((-par[1])*x[15]))*par[1]*par[3]*x[0], (-np.exp((-par[2])*x[16]))*par[2]*par[3]*x[0]], 
                         [self.par0[0]*x[10] + self.par0[1]*x[11], -((1 - np.exp((-par[1])*x[15]))*par[4]) - (1 - np.exp((-par[2])*x[16]))*par[4] - 
                         (1 - np.exp((-par[0])*x[14]))*par[4]*(par[17] + 1/par[18] - tau/par[18]) - self.par0[25], 0, 0, 0, 0, 0, 0, 0, 0, self.par0[0]*x[0], self.par0[1]*x[0], 0, 0, 
                         (-np.exp((-par[0])*x[14]))*par[0]*par[4]*(par[17] + 1/par[18] - tau/par[18])*x[1], (-np.exp((-par[1])*x[15]))*par[1]*par[4]*x[1], 
                         (-np.exp((-par[2])*x[16]))*par[2]*par[4]*x[1]], [0, (-self.par0[2])*self.par0[50]*x[2] - self.par0[6]*self.par0[52]*x[2], 
                         -((1 - np.exp((-par[1])*x[15]))*par[5]) - (1 - np.exp((-par[2])*x[16]))*par[5] - (1 - np.exp((-par[0])*x[14]))*par[5]*
                         (par[17] + 1/par[18] - tau/par[18]) - self.par0[26] - self.par0[50]*(self.par0[2]*x[1] + self.par0[3]*x[7]) - 
                         self.par0[52]*(self.par0[6]*x[1] + self.par0[5]*x[3] + self.par0[7]*x[7]) - self.par0[4]*self.par0[51]*x[11], (-self.par0[5])*self.par0[52]*x[2], 0, 0, 0, 
                         (-self.par0[3])*self.par0[50]*x[2] - self.par0[7]*self.par0[52]*x[2], 0, 0, 0, (-self.par0[4])*self.par0[51]*x[2], 0, 0, 
                         (-np.exp((-par[0])*x[14]))*par[0]*par[5]*(par[17] + 1/par[18] - tau/par[18])*x[2], (-np.exp((-par[1])*x[15]))*par[1]*par[5]*x[2], 
                         (-np.exp((-par[2])*x[16]))*par[2]*par[5]*x[2]], [0, self.par0[2]*x[2], self.par0[2]*x[1] + self.par0[3]*x[7], 
                         -((1 - np.exp((-par[1])*x[15]))*par[6]) - (1 - np.exp((-par[2])*x[16]))*par[6] - (1 - np.exp((-par[0])*x[14]))*par[6]*
                         (par[17] + 1/par[18] - tau/par[18]) - self.par0[29] - self.par0[27]*x[4] - self.par0[28]*x[11], (-self.par0[27])*x[3], 0, 0, self.par0[3]*x[2], 0, 0, 0, 
                         (-self.par0[28])*x[3], 0, 0, (-np.exp((-par[0])*x[14]))*par[0]*par[6]*(par[17] + 1/par[18] - tau/par[18])*x[3], 
                         (-np.exp((-par[1])*x[15]))*par[1]*par[6]*x[3], (-np.exp((-par[2])*x[16]))*par[2]*par[6]*x[3]], 
                         [0, 0, self.par0[4]*x[11], 0, -((1 - np.exp((-par[1])*x[15]))*par[7]) - (1 - np.exp((-par[2])*x[16]))*par[7] - 
                         (1 - np.exp((-par[0])*x[14]))*par[7]*(par[17] + 1/par[18] - tau/par[18]) - self.par0[30], 0, 0, 0, 0, 0, 0, self.par0[4]*x[2], 0, 0, 
                         (-np.exp((-par[0])*x[14]))*par[0]*par[7]*(par[17] + 1/par[18] - tau/par[18])*x[4], (-np.exp((-par[1])*x[15]))*par[1]*par[7]*x[4], 
                         (-np.exp((-par[2])*x[16]))*par[2]*par[7]*x[4]], [0, self.par0[6]*x[2], self.par0[6]*x[1] + self.par0[5]*x[3] + self.par0[7]*x[7], self.par0[5]*x[2], (-self.par0[31])*x[5], 
                         -((1 - np.exp((-par[1])*x[15]))*par[8]) - (1 - np.exp((-par[2])*x[16]))*par[8] - (1 - np.exp((-par[0])*x[14]))*par[8]*
                         (par[17] + 1/par[18] - tau/par[18]) - self.par0[33] - self.par0[31]*x[4] - self.par0[32]*x[11], 0, self.par0[7]*x[2], 0, 0, 0, (-self.par0[32])*x[5], 0, 0, 
                         (-np.exp((-par[0])*x[14]))*par[0]*par[8]*(par[17] + 1/par[18] - tau/par[18])*x[5], (-np.exp((-par[1])*x[15]))*par[1]*par[8]*x[5], 
                         (-np.exp((-par[2])*x[16]))*par[2]*par[8]*x[5]], [0, 0, 0, 0, 0, 0, -((1 - np.exp((-par[1])*x[15]))*par[9]) - (1 - np.exp((-par[2])*x[16]))*par[9] - 
                         (1 - np.exp((-par[0])*x[14]))*par[9]*(par[17] + 1/par[18] - tau/par[18]) - self.par0[34] - self.par0[53]*(self.par0[8]*x[8] + self.par0[9]*x[13]), 0, 
                         (-self.par0[8])*self.par0[53]*x[6], 0, 0, 0, 0, (-self.par0[9])*self.par0[53]*x[6], (-np.exp((-par[0])*x[14]))*par[0]*par[9]*(par[17] + 1/par[18] - tau/par[18])*
                         x[6], (-np.exp((-par[1])*x[15]))*par[1]*par[9]*x[6], (-np.exp((-par[2])*x[16]))*par[2]*par[9]*x[6]], 
                         [0, 0, 0, 0, 0, 0, self.par0[8]*x[8] + self.par0[9]*x[13], -((1 - np.exp((-par[1])*x[15]))*par[10]) - (1 - np.exp((-par[2])*x[16]))*par[10] - 
                         (1 - np.exp((-par[0])*x[14]))*par[10]*(par[17] + 1/par[18] - tau/par[18]) - self.par0[36] - self.par0[35]*x[8], self.par0[8]*x[6] - self.par0[35]*x[7], 0, 0, 0, 
                         0, self.par0[9]*x[6], (-np.exp((-par[0])*x[14]))*par[0]*par[10]*(par[17] + 1/par[18] - tau/par[18])*x[7], 
                         (-np.exp((-par[1])*x[15]))*par[1]*par[10]*x[7], (-np.exp((-par[2])*x[16]))*par[2]*par[10]*x[7]], 
                         [0, 0, 0, 0, 0, -(((1 - np.exp((-par[2])*x[16]))*par[16] + self.par0[37])*x[8]), 0, 0, -((1 - np.exp((-par[1])*x[15]))*par[11]) - 
                         (1 - np.exp((-par[2])*x[16]))*par[11] - (1 - np.exp((-par[0])*x[14]))*par[11]*(par[17] + 1/par[18] - tau/par[18]) - self.par0[39] - 
                         (1 - np.exp((-par[2])*x[16]))*par[16]*x[5] - self.par0[37]*x[5] - self.par0[38]*x[10] - (x[8]*(2*self.par0[10] + self.par0[11]*x[11] + self.par0[12]*x[12]))/self.par0[48] + 
                         (1 - x[8]/self.par0[48])*(2*self.par0[10] + self.par0[11]*x[11] + self.par0[12]*x[12]), 0, (-self.par0[38])*x[8], self.par0[11]*x[8]*(1 - x[8]/self.par0[48]), 
                         self.par0[12]*x[8]*(1 - x[8]/self.par0[48]), 0, (-np.exp((-par[0])*x[14]))*par[0]*par[11]*(par[17] + 1/par[18] - tau/par[18])*x[8], 
                         (-np.exp((-par[1])*x[15]))*par[1]*par[11]*x[8], (-np.exp((-par[2])*x[16]))*par[2]*par[11]*x[8] - (par[2]*par[16]*x[5]*x[8])/np.exp(par[2]*x[16])], 
                         [0, 0, 0, 0, 0, ((1 - np.exp((-par[2])*x[16]))*par[16] + self.par0[37])*self.par0[54]*x[8], 0, 0, 
                         par[15]*((1 - np.exp((-par[1])*x[15]))*par[11] + (1 - np.exp((-par[2])*x[16]))*par[11] + (1 - np.exp((-par[0])*x[14]))*par[11]*
                         (par[17] + 1/par[18] - tau/par[18])) + self.par0[54]*(self.par0[39] + (1 - np.exp((-par[2])*x[16]))*par[16]*x[5] + self.par0[37]*x[5] + self.par0[38]*x[10]), 
                         -self.par0[40], self.par0[38]*self.par0[54]*x[8], 0, 0, 0, (par[0]*par[11]*par[15]*(par[17] + 1/par[18] - tau/par[18])*x[8])/np.exp(par[0]*x[14]), 
                         (par[1]*par[11]*par[15]*x[8])/np.exp(par[1]*x[15]), (par[2]*par[11]*par[15]*x[8])/np.exp(par[2]*x[16]) + 
                         (par[2]*par[16]*self.par0[54]*x[5]*x[8])/np.exp(par[2]*x[16])], [0, 0, 0, self.par0[13], 0, self.par0[14], 0, 0, 0, 0, -self.par0[41], 0, 0, 0, 0, 0, 0], 
                         [0, self.par0[16], 0, self.par0[15], 0, 0, 0, 0, self.par0[17], 0, 0, -self.par0[42], 0, 0, 0, 0, 0], [0, self.par0[19], 0, self.par0[18], 0, 0, 0, 0, self.par0[20], 0, 0, 0, 
                         -self.par0[43], 0, 0, 0, 0], [0, self.par0[21], 0, 0, 0, 0, 0, self.par0[22], 0, self.par0[23], 0, 0, 0, -self.par0[44], 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -par[12], 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -par[13], 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -par[14]]])



    def Jp(self,t,x,par):
        # Jacobian with respect to the parameters
        tau = min(t, par[17]*par[18])
        # dF/dtau
        dF_dtau = np.array([((1 - np.exp((-par[0])*x[14]))*par[3]*x[0])/par[18], ((1 - np.exp((-par[0])*x[14]))*par[4]*x[1])/par[18], 
                            ((1 - np.exp((-par[0])*x[14]))*par[5]*x[2])/par[18], ((1 - np.exp((-par[0])*x[14]))*par[6]*x[3])/par[18], 
                            ((1 - np.exp((-par[0])*x[14]))*par[7]*x[4])/par[18], ((1 - np.exp((-par[0])*x[14]))*par[8]*x[5])/par[18], 
                            ((1 - np.exp((-par[0])*x[14]))*par[9]*x[6])/par[18], ((1 - np.exp((-par[0])*x[14]))*par[10]*x[7])/par[18], 
                            ((1 - np.exp((-par[0])*x[14]))*par[11]*x[8])/par[18], -(((1 - np.exp((-par[0])*x[14]))*par[11]*par[15]*x[8])/par[18]), 0, 0, 0, 0, 0, 
                            0, 0])
        # dtau/da and dtau/df
        dtau_df = par[17] if tau == par[17]*par[18] else 0
        dtau_da = par[18] if tau == par[17]*par[18] else 0
        
        x=x*(x>0)
        return np.array([[(-np.exp((-par[0])*x[14]))*par[3]*(par[17] + 1/par[18] - tau/par[18])*x[0]*x[14], (-np.exp((-par[0])*x[14]))*par[4]*
                         (par[17] + 1/par[18] - tau/par[18])*x[1]*x[14], (-np.exp((-par[0])*x[14]))*par[5]*(par[17] + 1/par[18] - tau/par[18])*x[2]*x[14], 
                         (-np.exp((-par[0])*x[14]))*par[6]*(par[17] + 1/par[18] - tau/par[18])*x[3]*x[14], (-np.exp((-par[0])*x[14]))*par[7]*
                         (par[17] + 1/par[18] - tau/par[18])*x[4]*x[14], (-np.exp((-par[0])*x[14]))*par[8]*(par[17] + 1/par[18] - tau/par[18])*x[5]*x[14], 
                         (-np.exp((-par[0])*x[14]))*par[9]*(par[17] + 1/par[18] - tau/par[18])*x[6]*x[14], (-np.exp((-par[0])*x[14]))*par[10]*
                         (par[17] + 1/par[18] - tau/par[18])*x[7]*x[14], (-np.exp((-par[0])*x[14]))*par[11]*(par[17] + 1/par[18] - tau/par[18])*x[8]*x[14], 
                         (par[11]*par[15]*(par[17] + 1/par[18] - tau/par[18])*x[8]*x[14])/np.exp(par[0]*x[14]), 0, 0, 0, 0, 0, 0, 0], 
                         [(-np.exp((-par[1])*x[15]))*par[3]*x[0]*x[15], (-np.exp((-par[1])*x[15]))*par[4]*x[1]*x[15], (-np.exp((-par[1])*x[15]))*par[5]*x[2]*x[15], 
                         (-np.exp((-par[1])*x[15]))*par[6]*x[3]*x[15], (-np.exp((-par[1])*x[15]))*par[7]*x[4]*x[15], (-np.exp((-par[1])*x[15]))*par[8]*x[5]*x[15], 
                         (-np.exp((-par[1])*x[15]))*par[9]*x[6]*x[15], (-np.exp((-par[1])*x[15]))*par[10]*x[7]*x[15], (-np.exp((-par[1])*x[15]))*par[11]*x[8]*x[15], 
                         (par[11]*par[15]*x[8]*x[15])/np.exp(par[1]*x[15]), 0, 0, 0, 0, 0, 0, 0], [(-np.exp((-par[2])*x[16]))*par[3]*x[0]*x[16], 
                         (-np.exp((-par[2])*x[16]))*par[4]*x[1]*x[16], (-np.exp((-par[2])*x[16]))*par[5]*x[2]*x[16], (-np.exp((-par[2])*x[16]))*par[6]*x[3]*x[16], 
                         (-np.exp((-par[2])*x[16]))*par[7]*x[4]*x[16], (-np.exp((-par[2])*x[16]))*par[8]*x[5]*x[16], (-np.exp((-par[2])*x[16]))*par[9]*x[6]*x[16], 
                         (-np.exp((-par[2])*x[16]))*par[10]*x[7]*x[16], (-np.exp((-par[2])*x[16]))*par[11]*x[8]*x[16] - (par[16]*x[5]*x[8]*x[16])/np.exp(par[2]*x[16]), 
                         (par[11]*par[15]*x[8]*x[16])/np.exp(par[2]*x[16]) + (par[16]*self.par0[54]*x[5]*x[8]*x[16])/np.exp(par[2]*x[16]), 0, 0, 0, 0, 0, 0, 0], 
                         [-((2 - np.exp((-par[1])*x[15]) - np.exp((-par[2])*x[16]) + (1 - np.exp((-par[0])*x[14]))*(par[17] + 1/par[18] - tau/par[18]))*x[0]), 0, 0, 0, 0, 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, -((2 - np.exp((-par[1])*x[15]) - np.exp((-par[2])*x[16]) + (1 - np.exp((-par[0])*x[14]))*(par[17] + 1/par[18] - tau/par[18]))*x[1]), 0, 0, 0, 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, -((2 - np.exp((-par[1])*x[15]) - np.exp((-par[2])*x[16]) + (1 - np.exp((-par[0])*x[14]))*(par[17] + 1/par[18] - tau/par[18]))*x[2]), 0, 0, 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, -((2 - np.exp((-par[1])*x[15]) - np.exp((-par[2])*x[16]) + (1 - np.exp((-par[0])*x[14]))*(par[17] + 1/par[18] - tau/par[18]))*x[3]), 0, 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, -((2 - np.exp((-par[1])*x[15]) - np.exp((-par[2])*x[16]) + (1 - np.exp((-par[0])*x[14]))*(par[17] + 1/par[18] - tau/par[18]))*x[4]), 0, 0, 
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 
                         -((2 - np.exp((-par[1])*x[15]) - np.exp((-par[2])*x[16]) + (1 - np.exp((-par[0])*x[14]))*(par[17] + 1/par[18] - tau/par[18]))*x[5]), 0, 0, 0, 0, 0, 0, 
                         0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, -((2 - np.exp((-par[1])*x[15]) - np.exp((-par[2])*x[16]) + (1 - np.exp((-par[0])*x[14]))*
                         (par[17] + 1/par[18] - tau/par[18]))*x[6]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                         [0, 0, 0, 0, 0, 0, 0, -((2 - np.exp((-par[1])*x[15]) - np.exp((-par[2])*x[16]) + (1 - np.exp((-par[0])*x[14]))*(par[17] + 1/par[18] - tau/par[18]))*
                         x[7]), 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 
                         -((2 - np.exp((-par[1])*x[15]) - np.exp((-par[2])*x[16]) + (1 - np.exp((-par[0])*x[14]))*(par[17] + 1/par[18] - tau/par[18]))*x[8]), 
                         par[15]*(2 - np.exp((-par[1])*x[15]) - np.exp((-par[2])*x[16]) + (1 - np.exp((-par[0])*x[14]))*(par[17] + 1/par[18] - tau/par[18]))*x[8], 0, 0, 0, 0, 
                         0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[14], 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[15], 0], 
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[16]], [0, 0, 0, 0, 0, 0, 0, 0, 0, 
                         ((1 - np.exp((-par[1])*x[15]))*par[11] + (1 - np.exp((-par[2])*x[16]))*par[11] + (1 - np.exp((-par[0])*x[14]))*par[11]*
                         (par[17] + 1/par[18] - tau/par[18]))*x[8], 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, -((1 - np.exp((-par[2])*x[16]))*x[5]*x[8]), 
                         (1 - np.exp((-par[2])*x[16]))*self.par0[54]*x[5]*x[8], 0, 0, 0, 0, 0, 0, 0], 
#                          list(map(add, dF_df, dF_dtau*dtau_df)), 
                         np.array([-((1 - np.exp((-par[0])*x[14]))*par[3]*x[0]), 
                         -((1 - np.exp((-par[0])*x[14]))*par[4]*x[1]), -((1 - np.exp((-par[0])*x[14]))*par[5]*x[2]), -((1 - np.exp((-par[0])*x[14]))*par[6]*x[3]), 
                         -((1 - np.exp((-par[0])*x[14]))*par[7]*x[4]), -((1 - np.exp((-par[0])*x[14]))*par[8]*x[5]), -((1 - np.exp((-par[0])*x[14]))*par[9]*x[6]), 
                         -((1 - np.exp((-par[0])*x[14]))*par[10]*x[7]), -((1 - np.exp((-par[0])*x[14]))*par[11]*x[8]), (1 - np.exp((-par[0])*x[14]))*par[11]*par[15]*x[8], 0, 0, 
                         0, 0, 0, 0, 0]) + dF_dtau*dtau_df,
                         np.array([-((1 - np.exp((-par[0])*x[14]))*par[3]*(-(1/par[18]**2) + tau/par[18]**2)*x[0]), 
                         -((1 - np.exp((-par[0])*x[14]))*par[4]*(-(1/par[18]**2) + tau/par[18]**2)*x[1]), 
                         -((1 - np.exp((-par[0])*x[14]))*par[5]*(-(1/par[18]**2) + tau/par[18]**2)*x[2]), 
                         -((1 - np.exp((-par[0])*x[14]))*par[6]*(-(1/par[18]**2) + tau/par[18]**2)*x[3]), 
                         -((1 - np.exp((-par[0])*x[14]))*par[7]*(-(1/par[18]**2) + tau/par[18]**2)*x[4]), 
                         -((1 - np.exp((-par[0])*x[14]))*par[8]*(-(1/par[18]**2) + tau/par[18]**2)*x[5]), 
                         -((1 - np.exp((-par[0])*x[14]))*par[9]*(-(1/par[18]**2) + tau/par[18]**2)*x[6]), 
                         -((1 - np.exp((-par[0])*x[14]))*par[10]*(-(1/par[18]**2) + tau/par[18]**2)*x[7]), 
                         -((1 - np.exp((-par[0])*x[14]))*par[11]*(-(1/par[18]**2) + tau/par[18]**2)*x[8]), (1 - np.exp((-par[0])*x[14]))*par[11]*par[15]*
                         (-(1/par[18]**2) + tau/par[18]**2)*x[8], 0, 0, 0, 0, 0, 0, 0]) + dF_dtau*dtau_da ])
                            
    def parameters_from_assumptions(self,data):
        #Function returning parameter values from literature
        # data is a list of arrays in the following order 
        # [drug_deltas, betas, K_C, K_immune, a, alphaNCA]     
        
        drug_deltas, betas, K_C, K_immune, a, alphaNCA, delCTcA3, drug_dose, SS_vals = data
        par=np.empty(19)
        
        for i in range(3):
            par[i] = betas[i]*drug_dose[i]/drug_deltas[i]
        for i in range(3, 11):
            par[i] = K_immune
        for i in range(12, 15):
            par[i] = drug_deltas[i-12]
            
        par[11] = K_C
        par[15] = alphaNCA*SS_vals[8]/SS_vals[9] # alphaNCA*C**inf/N**inf
        par[16] = delCTcA3 # delCTcA3 here is also non-dimensional
        par[17] = 0.5 # f
        par[18] = a

        return par



class OS_MAP_resistant_Functions(object):
    def __init__(self,parameters=None):
        self.nparam=22
        self.nvar=17
        self.variable_names=['Naive Macrophages  ($M_N$)', 'Macrophages  ($M$)', 'Naive T-cells  ($T_N$)', 'helper T-cells  ($T_h$)', 'Treg-cells  ($T_r$)',
                             'cytotoxic cells  ($T_c$)', 'Naive Dendritic cells  ($D_N$)', 'Dendritic cells  ($D$)', 'Cancer cells  ($C$)', 'Necrotic cells  ($N$)',
                             'IFN-$\gamma$  ($I_\gamma$)', '$\mu_1$', '$\mu_2$', 'HMGB1  ($H$)', 'A1', 'A2', 'A3']

        self.parameter_names=['$\\beta_{1I}$', '$\\beta_{2I}$', '$\\beta_{3I}$', '$\\beta_{1C}$', '$\\beta_{2C}$', '$\\beta_{3C}$', '$K_{M_N}$',
                              '$K_{M}$', '$K_{T_N}$', '$K_{T_h}$', '$K_{T_r}$', '$K_{T_c}$', '$K_{D_N}$', '$K_{D}$', '$K_{C}$', '$\\delta_{A_1}$',
                              '$\\delta_{A_2}$', '$\\delta_{A_3}$', '$\\alpha_{NCA}$', '$\\delta_{C T_c A_3}$', '$f$', '$a$']
        self.par0 = parameters

    def __call__(self,t,x,par):
        # ODE right-hand side
        tau = min(t, par[20]*par[21])
        x=x*(x>0) #Q
        return np.array([self.par0[45] - ((1 - np.exp((-par[1])*x[15]))*par[6] + (1 - np.exp((-par[2])*x[16]))*par[6] + (1 - np.exp((-par[0])*x[14]))*par[6]*
                        (par[20] + 1/par[21] - tau/par[21]))*x[0] - self.par0[24]*x[0] - self.par0[49]*x[0]*(self.par0[0]*x[10] + self.par0[1]*x[11]),
                        -(((1 - np.exp((-par[1])*x[15]))*par[7] + (1 - np.exp((-par[2])*x[16]))*par[7] + (1 - np.exp((-par[0])*x[14]))*par[7]*(par[20] + 1/par[21] - tau/par[21]))*
                        x[1]) - self.par0[25]*x[1] + x[0]*(self.par0[0]*x[10] + self.par0[1]*x[11]),
                        self.par0[46] - ((1 - np.exp((-par[1])*x[15]))*par[8] + (1 - np.exp((-par[2])*x[16]))*par[8] + (1 - np.exp((-par[0])*x[14]))*par[8]*
                        (par[20] + 1/par[21] - tau/par[21]))*x[2] - self.par0[26]*x[2] - self.par0[50]*x[2]*(self.par0[2]*x[1] + self.par0[3]*x[7]) -
                        self.par0[52]*x[2]*(self.par0[6]*x[1] + self.par0[5]*x[3] + self.par0[7]*x[7]) - self.par0[4]*self.par0[51]*x[2]*x[11],
                        -(((1 - np.exp((-par[1])*x[15]))*par[9] + (1 - np.exp((-par[2])*x[16]))*par[9] + (1 - np.exp((-par[0])*x[14]))*par[9]*(par[20] + 1/par[21] - tau/par[21]))*
                        x[3]) + x[2]*(self.par0[2]*x[1] + self.par0[3]*x[7]) - x[3]*(self.par0[29] + self.par0[27]*x[4] + self.par0[28]*x[11]),
                        -(((1 - np.exp((-par[1])*x[15]))*par[10] + (1 - np.exp((-par[2])*x[16]))*par[10] + (1 - np.exp((-par[0])*x[14]))*par[10]*
                        (par[20] + 1/par[21] - tau/par[21]))*x[4]) - self.par0[30]*x[4] + self.par0[4]*x[2]*x[11],
                        -(((1 - np.exp((-par[1])*x[15]))*par[11] + (1 - np.exp((-par[2])*x[16]))*par[11] + (1 - np.exp((-par[0])*x[14]))*par[11]*
                        (par[20] + 1/par[21] - tau/par[21]))*x[5]) + x[2]*(self.par0[6]*x[1] + self.par0[5]*x[3] + self.par0[7]*x[7]) -
                        x[5]*(self.par0[33] + self.par0[31]*x[4] + self.par0[32]*x[11]),
                        self.par0[47] - ((1 - np.exp((-par[1])*x[15]))*par[12] + (1 - np.exp((-par[2])*x[16]))*par[12] + (1 - np.exp((-par[0])*x[14]))*par[12]*
                        (par[20] + 1/par[21] - tau/par[21]))*x[6] - self.par0[34]*x[6] - self.par0[53]*x[6]*(self.par0[8]*x[8] + self.par0[9]*x[13]),
                        -(((1 - np.exp((-par[1])*x[15]))*par[13] + (1 - np.exp((-par[2])*x[16]))*par[13] + (1 - np.exp((-par[0])*x[14]))*par[13]*
                        (par[20] + 1/par[21] - tau/par[21]))*x[7]) - x[7]*(self.par0[36] + self.par0[35]*x[8]) + x[6]*(self.par0[8]*x[8] + self.par0[9]*x[13]),
                        -(((1 - np.exp((-par[4])*x[15]))*par[14] + (1 - np.exp((-par[5])*x[16]))*par[14] + (1 - np.exp((-par[3])*x[14]))*par[14]*
                        (par[20] + 1/par[21] - tau/par[21]))*x[8]) - x[8]*(self.par0[39] + (1 - np.exp((-par[5])*x[16]))*par[19]*x[5] + self.par0[37]*x[5] + self.par0[38]*x[10]) +
                        x[8]*(1 - x[8]/self.par0[48])*(2*self.par0[10] + self.par0[11]*x[11] + self.par0[12]*x[12]),
                        par[18]*((1 - np.exp((-par[4])*x[15]))*par[14] + (1 - np.exp((-par[5])*x[16]))*par[14] + (1 - np.exp((-par[3])*x[14]))*par[14]*
                        (par[20] + 1/par[21] - tau/par[21]))*x[8] - self.par0[40]*x[9] + self.par0[54]*x[8]*(self.par0[39] + (1 - np.exp((-par[5])*x[16]))*par[19]*x[5] +
                        self.par0[37]*x[5] + self.par0[38]*x[10]), self.par0[13]*x[3] + self.par0[14]*x[5] - self.par0[41]*x[10], self.par0[16]*x[1] + self.par0[15]*x[3] + self.par0[17]*x[8] -
                        self.par0[42]*x[11], self.par0[19]*x[1] + self.par0[18]*x[3] + self.par0[20]*x[8] - self.par0[43]*x[12], self.par0[21]*x[1] + self.par0[22]*x[7] + self.par0[23]*x[9] -
                        self.par0[44]*x[13], (-par[15])*x[14], (-par[16])*x[15], (-par[17])*x[16]])


    def Ju(self,t,x,par):
        # Jacobian with respect to variables
        tau = min(t, par[20]*par[21])
        x=x*(x>0)
        return np.array([[-((1 - np.exp((-par[1])*x[15]))*par[6]) - (1 - np.exp((-par[2])*x[16]))*par[6] - (1 - np.exp((-par[0])*x[14]))*par[6]*
                        (par[20] + 1/par[21] - tau/par[21]) - self.par0[24] - self.par0[49]*(self.par0[0]*x[10] + self.par0[1]*x[11]), 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        (-self.par0[0])*self.par0[49]*x[0], (-self.par0[1])*self.par0[49]*x[0], 0, 0, (-np.exp((-par[0])*x[14]))*par[0]*par[6]*(par[20] + 1/par[21] - tau/par[21])*x[0],
                        (-np.exp((-par[1])*x[15]))*par[1]*par[6]*x[0], (-np.exp((-par[2])*x[16]))*par[2]*par[6]*x[0]],
                        [self.par0[0]*x[10] + self.par0[1]*x[11], -((1 - np.exp((-par[1])*x[15]))*par[7]) - (1 - np.exp((-par[2])*x[16]))*par[7] -
                        (1 - np.exp((-par[0])*x[14]))*par[7]*(par[20] + 1/par[21] - tau/par[21]) - self.par0[25], 0, 0, 0, 0, 0, 0, 0, 0, self.par0[0]*x[0], self.par0[1]*x[0], 0, 0,
                        (-np.exp((-par[0])*x[14]))*par[0]*par[7]*(par[20] + 1/par[21] - tau/par[21])*x[1], (-np.exp((-par[1])*x[15]))*par[1]*par[7]*x[1],
                        (-np.exp((-par[2])*x[16]))*par[2]*par[7]*x[1]], [0, (-self.par0[2])*self.par0[50]*x[2] - self.par0[6]*self.par0[52]*x[2],
                        -((1 - np.exp((-par[1])*x[15]))*par[8]) - (1 - np.exp((-par[2])*x[16]))*par[8] - (1 - np.exp((-par[0])*x[14]))*par[8]*
                        (par[20] + 1/par[21] - tau/par[21]) - self.par0[26] - self.par0[50]*(self.par0[2]*x[1] + self.par0[3]*x[7]) -
                        self.par0[52]*(self.par0[6]*x[1] + self.par0[5]*x[3] + self.par0[7]*x[7]) - self.par0[4]*self.par0[51]*x[11], (-self.par0[5])*self.par0[52]*x[2], 0, 0, 0,
                        (-self.par0[3])*self.par0[50]*x[2] - self.par0[7]*self.par0[52]*x[2], 0, 0, 0, (-self.par0[4])*self.par0[51]*x[2], 0, 0,
                        (-np.exp((-par[0])*x[14]))*par[0]*par[8]*(par[20] + 1/par[21] - tau/par[21])*x[2], (-np.exp((-par[1])*x[15]))*par[1]*par[8]*x[2],
                        (-np.exp((-par[2])*x[16]))*par[2]*par[8]*x[2]], [0, self.par0[2]*x[2], self.par0[2]*x[1] + self.par0[3]*x[7],
                        -((1 - np.exp((-par[1])*x[15]))*par[9]) - (1 - np.exp((-par[2])*x[16]))*par[9] - (1 - np.exp((-par[0])*x[14]))*par[9]*
                        (par[20] + 1/par[21] - tau/par[21]) - self.par0[29] - self.par0[27]*x[4] - self.par0[28]*x[11], (-self.par0[27])*x[3], 0, 0, self.par0[3]*x[2], 0, 0, 0,
                        (-self.par0[28])*x[3], 0, 0, (-np.exp((-par[0])*x[14]))*par[0]*par[9]*(par[20] + 1/par[21] - tau/par[21])*x[3],
                        (-np.exp((-par[1])*x[15]))*par[1]*par[9]*x[3], (-np.exp((-par[2])*x[16]))*par[2]*par[9]*x[3]],
                        [0, 0, self.par0[4]*x[11], 0, -((1 - np.exp((-par[1])*x[15]))*par[10]) - (1 - np.exp((-par[2])*x[16]))*par[10] -
                        (1 - np.exp((-par[0])*x[14]))*par[10]*(par[20] + 1/par[21] - tau/par[21]) - self.par0[30], 0, 0, 0, 0, 0, 0, self.par0[4]*x[2], 0, 0,
                        (-np.exp((-par[0])*x[14]))*par[0]*par[10]*(par[20] + 1/par[21] - tau/par[21])*x[4], (-np.exp((-par[1])*x[15]))*par[1]*par[10]*x[4],
                        (-np.exp((-par[2])*x[16]))*par[2]*par[10]*x[4]], [0, self.par0[6]*x[2], self.par0[6]*x[1] + self.par0[5]*x[3] + self.par0[7]*x[7], self.par0[5]*x[2], (-self.par0[31])*x[5],
                        -((1 - np.exp((-par[1])*x[15]))*par[11]) - (1 - np.exp((-par[2])*x[16]))*par[11] - (1 - np.exp((-par[0])*x[14]))*par[11]*
                        (par[20] + 1/par[21] - tau/par[21]) - self.par0[33] - self.par0[31]*x[4] - self.par0[32]*x[11], 0, self.par0[7]*x[2], 0, 0, 0, (-self.par0[32])*x[5], 0, 0,
                        (-np.exp((-par[0])*x[14]))*par[0]*par[11]*(par[20] + 1/par[21] - tau/par[21])*x[5], (-np.exp((-par[1])*x[15]))*par[1]*par[11]*x[5],
                        (-np.exp((-par[2])*x[16]))*par[2]*par[11]*x[5]], [0, 0, 0, 0, 0, 0, -((1 - np.exp((-par[1])*x[15]))*par[12]) - (1 - np.exp((-par[2])*x[16]))*par[12] -
                        (1 - np.exp((-par[0])*x[14]))*par[12]*(par[20] + 1/par[21] - tau/par[21]) - self.par0[34] - self.par0[53]*(self.par0[8]*x[8] + self.par0[9]*x[13]), 0,
                        (-self.par0[8])*self.par0[53]*x[6], 0, 0, 0, 0, (-self.par0[9])*self.par0[53]*x[6], (-np.exp((-par[0])*x[14]))*par[0]*par[12]*(par[20] + 1/par[21] - tau/par[21])*
                        x[6], (-np.exp((-par[1])*x[15]))*par[1]*par[12]*x[6], (-np.exp((-par[2])*x[16]))*par[2]*par[12]*x[6]],
                        [0, 0, 0, 0, 0, 0, self.par0[8]*x[8] + self.par0[9]*x[13], -((1 - np.exp((-par[1])*x[15]))*par[13]) - (1 - np.exp((-par[2])*x[16]))*par[13] -
                        (1 - np.exp((-par[0])*x[14]))*par[13]*(par[20] + 1/par[21] - tau/par[21]) - self.par0[36] - self.par0[35]*x[8], self.par0[8]*x[6] - self.par0[35]*x[7], 0, 0, 0,
                        0, self.par0[9]*x[6], (-np.exp((-par[0])*x[14]))*par[0]*par[13]*(par[20] + 1/par[21] - tau/par[21])*x[7],
                        (-np.exp((-par[1])*x[15]))*par[1]*par[13]*x[7], (-np.exp((-par[2])*x[16]))*par[2]*par[13]*x[7]],
                        [0, 0, 0, 0, 0, -(((1 - np.exp((-par[5])*x[16]))*par[19] + self.par0[37])*x[8]), 0, 0, -((1 - np.exp((-par[4])*x[15]))*par[14]) -
                        (1 - np.exp((-par[5])*x[16]))*par[14] - (1 - np.exp((-par[3])*x[14]))*par[14]*(par[20] + 1/par[21] - tau/par[21]) - self.par0[39] -
                        (1 - np.exp((-par[5])*x[16]))*par[19]*x[5] - self.par0[37]*x[5] - self.par0[38]*x[10] - (x[8]*(2*self.par0[10] + self.par0[11]*x[11] + self.par0[12]*x[12]))/self.par0[48] +
                        (1 - x[8]/self.par0[48])*(2*self.par0[10] + self.par0[11]*x[11] + self.par0[12]*x[12]), 0, (-self.par0[38])*x[8], self.par0[11]*x[8]*(1 - x[8]/self.par0[48]),
                        self.par0[12]*x[8]*(1 - x[8]/self.par0[48]), 0, (-np.exp((-par[3])*x[14]))*par[3]*par[14]*(par[20] + 1/par[21] - tau/par[21])*x[8],
                        (-np.exp((-par[4])*x[15]))*par[4]*par[14]*x[8], (-np.exp((-par[5])*x[16]))*par[5]*par[14]*x[8] - (par[5]*par[19]*x[5]*x[8])/np.exp(par[5]*x[16])],
                        [0, 0, 0, 0, 0, ((1 - np.exp((-par[5])*x[16]))*par[19] + self.par0[37])*self.par0[54]*x[8], 0, 0,
                        par[18]*((1 - np.exp((-par[4])*x[15]))*par[14] + (1 - np.exp((-par[5])*x[16]))*par[14] + (1 - np.exp((-par[3])*x[14]))*par[14]*
                        (par[20] + 1/par[21] - tau/par[21])) + self.par0[54]*(self.par0[39] + (1 - np.exp((-par[5])*x[16]))*par[19]*x[5] + self.par0[37]*x[5] + self.par0[38]*x[10]),
                        -self.par0[40], self.par0[38]*self.par0[54]*x[8], 0, 0, 0, (par[3]*par[14]*par[18]*(par[20] + 1/par[21] - tau/par[21])*x[8])/np.exp(par[3]*x[14]),
                        (par[4]*par[14]*par[18]*x[8])/np.exp(par[4]*x[15]), (par[5]*par[14]*par[18]*x[8])/np.exp(par[5]*x[16]) +
                        (par[5]*par[19]*self.par0[54]*x[5]*x[8])/np.exp(par[5]*x[16])], [0, 0, 0, self.par0[13], 0, self.par0[14], 0, 0, 0, 0, -self.par0[41], 0, 0, 0, 0, 0, 0],
                        [0, self.par0[16], 0, self.par0[15], 0, 0, 0, 0, self.par0[17], 0, 0, -self.par0[42], 0, 0, 0, 0, 0], [0, self.par0[19], 0, self.par0[18], 0, 0, 0, 0, self.par0[20], 0, 0, 0,
                        -self.par0[43], 0, 0, 0, 0], [0, self.par0[21], 0, 0, 0, 0, 0, self.par0[22], 0, self.par0[23], 0, 0, 0, -self.par0[44], 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -par[15], 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -par[16], 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -par[17]]])


    def Jp(self,t,x,par):
        # Jacobian with respect to the parameters
        tau = min(t, par[20]*par[21])
        # dF/dtau
        dF_dtau = np.array([((1 - np.exp((-par[0])*x[14]))*par[6]*x[0])/par[21], ((1 - np.exp((-par[0])*x[14]))*par[7]*x[1])/par[21],
                           ((1 - np.exp((-par[0])*x[14]))*par[8]*x[2])/par[21], ((1 - np.exp((-par[0])*x[14]))*par[9]*x[3])/par[21],
                           ((1 - np.exp((-par[0])*x[14]))*par[10]*x[4])/par[21], ((1 - np.exp((-par[0])*x[14]))*par[11]*x[5])/par[21],
                           ((1 - np.exp((-par[0])*x[14]))*par[12]*x[6])/par[21], ((1 - np.exp((-par[0])*x[14]))*par[13]*x[7])/par[21],
                           ((1 - np.exp((-par[3])*x[14]))*par[14]*x[8])/par[21], -(((1 - np.exp((-par[3])*x[14]))*par[14]*par[18]*x[8])/par[21]),
                           0, 0, 0, 0, 0, 0, 0])
        # dtau/da and dtau/df
        dtau_df = par[20] if tau == par[20]*par[21] else 0
        dtau_da = par[21] if tau == par[20]*par[21] else 0

        x=x*(x>0)
        return np.array([[(-np.exp((-par[0])*x[14]))*par[6]*(par[20] + 1/par[21] - tau/par[21])*x[0]*x[14], (-np.exp((-par[0])*x[14]))*par[7]*
                         (par[20] + 1/par[21] - tau/par[21])*x[1]*x[14], (-np.exp((-par[0])*x[14]))*par[8]*(par[20] + 1/par[21] - tau/par[21])*x[2]*x[14],
                         (-np.exp((-par[0])*x[14]))*par[9]*(par[20] + 1/par[21] - tau/par[21])*x[3]*x[14], (-np.exp((-par[0])*x[14]))*par[10]*
                         (par[20] + 1/par[21] - tau/par[21])*x[4]*x[14], (-np.exp((-par[0])*x[14]))*par[11]*(par[20] + 1/par[21] - tau/par[21])*x[5]*x[14],
                         (-np.exp((-par[0])*x[14]))*par[12]*(par[20] + 1/par[21] - tau/par[21])*x[6]*x[14], (-np.exp((-par[0])*x[14]))*par[13]*
                         (par[20] + 1/par[21] - tau/par[21])*x[7]*x[14], 0, 0, 0, 0, 0, 0, 0, 0, 0], [(-np.exp((-par[1])*x[15]))*par[6]*x[0]*x[15],
                         (-np.exp((-par[1])*x[15]))*par[7]*x[1]*x[15], (-np.exp((-par[1])*x[15]))*par[8]*x[2]*x[15], (-np.exp((-par[1])*x[15]))*par[9]*x[3]*x[15],
                         (-np.exp((-par[1])*x[15]))*par[10]*x[4]*x[15], (-np.exp((-par[1])*x[15]))*par[11]*x[5]*x[15], (-np.exp((-par[1])*x[15]))*par[12]*x[6]*x[15],
                         (-np.exp((-par[1])*x[15]))*par[13]*x[7]*x[15], 0, 0, 0, 0, 0, 0, 0, 0, 0], [(-np.exp((-par[2])*x[16]))*par[6]*x[0]*x[16],
                         (-np.exp((-par[2])*x[16]))*par[7]*x[1]*x[16], (-np.exp((-par[2])*x[16]))*par[8]*x[2]*x[16], (-np.exp((-par[2])*x[16]))*par[9]*x[3]*x[16],
                         (-np.exp((-par[2])*x[16]))*par[10]*x[4]*x[16], (-np.exp((-par[2])*x[16]))*par[11]*x[5]*x[16], (-np.exp((-par[2])*x[16]))*par[12]*x[6]*x[16],
                         (-np.exp((-par[2])*x[16]))*par[13]*x[7]*x[16], 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, (-np.exp((-par[3])*x[14]))*par[14]*(par[20] + 1/par[21] - tau/par[21])*x[8]*x[14],
                         (par[14]*par[18]*(par[20] + 1/par[21] - tau/par[21])*x[8]*x[14])/np.exp(par[3]*x[14]), 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, (-np.exp((-par[4])*x[15]))*par[14]*x[8]*x[15], (par[14]*par[18]*x[8]*x[15])/np.exp(par[4]*x[15]), 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, (-np.exp((-par[5])*x[16]))*par[14]*x[8]*x[16] - (par[19]*x[5]*x[8]*x[16])/np.exp(par[5]*x[16]),
                         (par[14]*par[18]*x[8]*x[16])/np.exp(par[5]*x[16]) + (par[19]*self.par0[54]*x[5]*x[8]*x[16])/np.exp(par[5]*x[16]), 0, 0, 0, 0, 0, 0, 0],
                         [-((2 - np.exp((-par[1])*x[15]) - np.exp((-par[2])*x[16]) + (1 - np.exp((-par[0])*x[14]))*(par[20] + 1/par[21] - tau/par[21]))*x[0]), 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, -((2 - np.exp((-par[1])*x[15]) - np.exp((-par[2])*x[16]) + (1 - np.exp((-par[0])*x[14]))*(par[20] + 1/par[21] - tau/par[21]))*x[1]), 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, -((2 - np.exp((-par[1])*x[15]) - np.exp((-par[2])*x[16]) + (1 - np.exp((-par[0])*x[14]))*(par[20] + 1/par[21] - tau/par[21]))*x[2]), 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, -((2 - np.exp((-par[1])*x[15]) - np.exp((-par[2])*x[16]) + (1 - np.exp((-par[0])*x[14]))*(par[20] + 1/par[21] - tau/par[21]))*x[3]), 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, -((2 - np.exp((-par[1])*x[15]) - np.exp((-par[2])*x[16]) + (1 - np.exp((-par[0])*x[14]))*(par[20] + 1/par[21] - tau/par[21]))*x[4]), 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0,
                         -((2 - np.exp((-par[1])*x[15]) - np.exp((-par[2])*x[16]) + (1 - np.exp((-par[0])*x[14]))*(par[20] + 1/par[21] - tau/par[21]))*x[5]), 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, -((2 - np.exp((-par[1])*x[15]) - np.exp((-par[2])*x[16]) + (1 - np.exp((-par[0])*x[14]))*
                         (par[20] + 1/par[21] - tau/par[21]))*x[6]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, -((2 - np.exp((-par[1])*x[15]) - np.exp((-par[2])*x[16]) + (1 - np.exp((-par[0])*x[14]))*(par[20] + 1/par[21] - tau/par[21]))*
                         x[7]), 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0,
                         -((2 - np.exp((-par[4])*x[15]) - np.exp((-par[5])*x[16]) + (1 - np.exp((-par[3])*x[14]))*(par[20] + 1/par[21] - tau/par[21]))*x[8]),
                         par[18]*(2 - np.exp((-par[4])*x[15]) - np.exp((-par[5])*x[16]) + (1 - np.exp((-par[3])*x[14]))*(par[20] + 1/par[21] - tau/par[21]))*x[8], 0, 0, 0, 0,
                         0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[14], 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[15], 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[16]], [0, 0, 0, 0, 0, 0, 0, 0, 0,
                         ((1 - np.exp((-par[4])*x[15]))*par[14] + (1 - np.exp((-par[5])*x[16]))*par[14] + (1 - np.exp((-par[3])*x[14]))*par[14]*
                         (par[20] + 1/par[21] - tau/par[21]))*x[8], 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, -((1 - np.exp((-par[5])*x[16]))*x[5]*x[8]),
                         (1 - np.exp((-par[5])*x[16]))*self.par0[54]*x[5]*x[8], 0, 0, 0, 0, 0, 0, 0],
                         np.array([-((1 - np.exp((-par[0])*x[14]))*par[6]*x[0]),
                         -((1 - np.exp((-par[0])*x[14]))*par[7]*x[1]), -((1 - np.exp((-par[0])*x[14]))*par[8]*x[2]), -((1 - np.exp((-par[0])*x[14]))*par[9]*x[3]),
                         -((1 - np.exp((-par[0])*x[14]))*par[10]*x[4]), -((1 - np.exp((-par[0])*x[14]))*par[11]*x[5]), -((1 - np.exp((-par[0])*x[14]))*par[12]*x[6]),
                         -((1 - np.exp((-par[0])*x[14]))*par[13]*x[7]), -((1 - np.exp((-par[3])*x[14]))*par[14]*x[8]), (1 - np.exp((-par[3])*x[14]))*par[14]*par[18]*x[8], 0, 0,
                         0, 0, 0, 0, 0]) + dF_dtau*dtau_df,
                         np.array([-((1 - np.exp((-par[0])*x[14]))*par[6]*(-(1/par[21]**2) + tau/par[21]**2)*x[0]),
                         -((1 - np.exp((-par[0])*x[14]))*par[7]*(-(1/par[21]**2) + tau/par[21]**2)*x[1]),
                         -((1 - np.exp((-par[0])*x[14]))*par[8]*(-(1/par[21]**2) + tau/par[21]**2)*x[2]),
                         -((1 - np.exp((-par[0])*x[14]))*par[9]*(-(1/par[21]**2) + tau/par[21]**2)*x[3]),
                         -((1 - np.exp((-par[0])*x[14]))*par[10]*(-(1/par[21]**2) + tau/par[21]**2)*x[4]),
                         -((1 - np.exp((-par[0])*x[14]))*par[11]*(-(1/par[21]**2) + tau/par[21]**2)*x[5]),
                         -((1 - np.exp((-par[0])*x[14]))*par[12]*(-(1/par[21]**2) + tau/par[21]**2)*x[6]),
                         -((1 - np.exp((-par[0])*x[14]))*par[13]*(-(1/par[21]**2) + tau/par[21]**2)*x[7]),
                         -((1 - np.exp((-par[3])*x[14]))*par[14]*(-(1/par[21]**2) + tau/par[21]**2)*x[8]), (1 - np.exp((-par[3])*x[14]))*par[14]*par[18]*
                         (-(1/par[21]**2) + tau/par[21]**2)*x[8], 0, 0, 0, 0, 0, 0, 0]) + dF_dtau*dtau_da])


    def parameters_from_assumptions(self,data):
        #Function returning parameter values from literature or assumptions
        # data is a list of arrays in the following order
        # [drug_deltas, betas, K_C, K_immune, a, alphaNCA]

        drug_deltas, betasI, betasC, K_C, K_immune, a, alphaNCA, delCTcA3, baseline_drug_dose, SS_vals = data
        par=np.empty(22)

        for i in range(3):
            par[i] = betasI[i]*baseline_drug_dose[i]/drug_deltas[i]
            par[i+3] = betasC[i]*baseline_drug_dose[i]/drug_deltas[i]
            par[i+15] = drug_deltas[i]
        for i in range(6, 14):
            par[i] = K_immune

        par[14] = K_C
        par[18] = alphaNCA*SS_vals[8]/SS_vals[9] # alphaNCA*C**inf/N**inf
        par[19] = delCTcA3 # delCTcA3 taken from steady states calculation is also non-dimensional
        par[20] = 0.5 # f
        par[21] = a

        return par



class OS_AP_Functions(object):
    def __init__(self,parameters=None):
        self.nparam=17
        self.nvar=16
        self.variable_names=['Naive Macrophages  ($M_N$)', 'Macrophages  ($M$)', 'Naive T-cells  ($T_N$)', 'helper T-cells  ($T_h$)', 'Treg-cells  ($T_r$)',
                             'cytotoxic cells  ($T_c$)', 'Naive Dendritic cells  ($D_N$)', 'Dendritic cells  ($D$)', 'Cancer cells  ($C$)', 'Necrotic cells  ($N$)',
                             'IFN-$\gamma$  ($I_\gamma$)', '$\mu_1$', '$\mu_2$', 'HMGB1  ($H$)', 'A2', 'A3']

        self.parameter_names=['$\\beta_{1I}$', '$\\beta_{2I}$', '$\\beta_{3I}$', '$\\beta_{1C}$', '$\\beta_{2C}$', '$\\beta_{3C}$', '$K_{M_N}$',
                              '$K_{M}$', '$K_{T_N}$', '$K_{T_h}$', '$K_{T_r}$', '$K_{T_c}$', '$K_{D_N}$', '$K_{D}$', '$K_{C}$', '$\\delta_{A_1}$',
                              '$\\delta_{A_2}$', '$\\delta_{A_3}$', '$\\alpha_{NCA}$', '$\\delta_{C T_c A_3}$', '$f$', '$a$']
        self.par0 = parameters

    def __call__(self,t,x,par):
        # ODE right-hand side
        x=x*(x>0) #Q
        return np.array([self.par0[45] - ((1 - np.exp((-par[0])*x[14]))*par[4] + (1 - np.exp((-par[1])*x[15]))*par[4])*x[0] - self.par0[24]*x[0] -
                         self.par0[49]*x[0]*(self.par0[0]*x[10] + self.par0[1]*x[11]), -(((1 - np.exp((-par[0])*x[14]))*par[5] + (1 - np.exp((-par[1])*x[15]))*par[5])*x[1]) -
                         self.par0[25]*x[1] + x[0]*(self.par0[0]*x[10] + self.par0[1]*x[11]), self.par0[46] - ((1 - np.exp((-par[0])*x[14]))*par[6] + (1 - np.exp((-par[1])*x[15]))*par[6])*
                         x[2] - self.par0[26]*x[2] - self.par0[50]*x[2]*(self.par0[2]*x[1] + self.par0[3]*x[7]) - self.par0[52]*x[2]*(self.par0[6]*x[1] + self.par0[5]*x[3] + self.par0[7]*x[7]) -
                         self.par0[4]*self.par0[51]*x[2]*x[11], -(((1 - np.exp((-par[0])*x[14]))*par[7] + (1 - np.exp((-par[1])*x[15]))*par[7])*x[3]) +
                         x[2]*(self.par0[2]*x[1] + self.par0[3]*x[7]) - x[3]*(self.par0[29] + self.par0[27]*x[4] + self.par0[28]*x[11]),
                         -(((1 - np.exp((-par[0])*x[14]))*par[8] + (1 - np.exp((-par[1])*x[15]))*par[8])*x[4]) - self.par0[30]*x[4] + self.par0[4]*x[2]*x[11],
                         -(((1 - np.exp((-par[0])*x[14]))*par[9] + (1 - np.exp((-par[1])*x[15]))*par[9])*x[5]) + x[2]*(self.par0[6]*x[1] + self.par0[5]*x[3] + self.par0[7]*x[7]) -
                         x[5]*(self.par0[33] + self.par0[31]*x[4] + self.par0[32]*x[11]), self.par0[47] - ((1 - np.exp((-par[0])*x[14]))*par[10] + (1 - np.exp((-par[1])*x[15]))*par[10])*x[6] -
                         self.par0[34]*x[6] - self.par0[53]*x[6]*(self.par0[8]*x[8] + self.par0[9]*x[13]),
                         -(((1 - np.exp((-par[0])*x[14]))*par[11] + (1 - np.exp((-par[1])*x[15]))*par[11])*x[7]) - x[7]*(self.par0[36] + self.par0[35]*x[8]) +
                         x[6]*(self.par0[8]*x[8] + self.par0[9]*x[13]), -(((1 - np.exp((-par[2])*x[14]))*par[12] + (1 - np.exp((-par[3])*x[15]))*par[12])*x[8]) -
                         x[8]*(self.par0[39] + (1 - np.exp((-par[3])*x[15]))*par[16]*x[5] + self.par0[37]*x[5] + self.par0[38]*x[10]) +
                         x[8]*(1 - x[8]/self.par0[48])*(2*self.par0[10] + self.par0[11]*x[11] + self.par0[12]*x[12]),
                         ((1 - np.exp((-par[2])*x[14]))*par[12] + (1 - np.exp((-par[3])*x[15]))*par[12])*par[15]*x[8] - self.par0[40]*x[9] +
                         self.par0[54]*x[8]*(self.par0[39] + (1 - np.exp((-par[3])*x[15]))*par[16]*x[5] + self.par0[37]*x[5] + self.par0[38]*x[10]),
                         self.par0[13]*x[3] + self.par0[14]*x[5] - self.par0[41]*x[10], self.par0[16]*x[1] + self.par0[15]*x[3] + self.par0[17]*x[8] - self.par0[42]*x[11],
                         self.par0[19]*x[1] + self.par0[18]*x[3] + self.par0[20]*x[8] - self.par0[43]*x[12], self.par0[21]*x[1] + self.par0[22]*x[7] + self.par0[23]*x[9] - self.par0[44]*x[13],
                         (-par[13])*x[14], (-par[14])*x[15]])


    def parameters_from_assumptions(self,data):
        #Function returning parameter values from literature
        # data is a list of arrays in the following order
        # [drug_deltas, betas, K_C, K_immune, a, alphaNCA]

        drug_deltas, betasI, betasC, K_C, K_immune, a, alphaNCA, delCTcA3, baseline_drug_dose, SS_vals = data
        par=np.empty(17)

        for i in range(2):
            par[i] = betasI[i]*baseline_drug_dose[i]/drug_deltas[i]
            par[i+2] = betasC[i]*baseline_drug_dose[i]/drug_deltas[i]
            par[i+13] = drug_deltas[i]
        for i in range(4, 12):
            par[i] = K_immune

        par[12] = K_C
        par[15] = alphaNCA*SS_vals[8]/SS_vals[9] # alphaNCA*C^inf/N^inf
        par[16] = delCTcA3 # delCTcA3 taken from steady states calculation is also non-dimensional

        return par



class OS_MTX_Functions(object):
    def __init__(self,parameters=None):
        self.nparam=15
        self.nvar=15
        self.variable_names=['Naive Macrophages  ($M_N$)', 'Macrophages  ($M$)', 'Naive T-cells  ($T_N$)', 'helper T-cells  ($T_h$)', 'Treg-cells  ($T_r$)',
                             'cytotoxic cells  ($T_c$)', 'Naive Dendritic cells  ($D_N$)', 'Dendritic cells  ($D$)', 'Cancer cells  ($C$)', 'Necrotic cells  ($N$)',
                             'IFN-$\gamma$  ($I_\gamma$)', '$\mu_1$', '$\mu_2$', 'HMGB1  ($H$)', 'A2', 'A3']

        self.parameter_names=['$\\beta_{1I}$', '$\\beta_{2I}$', '$\\beta_{3I}$', '$\\beta_{1C}$', '$\\beta_{2C}$', '$\\beta_{3C}$', '$K_{M_N}$',
                              '$K_{M}$', '$K_{T_N}$', '$K_{T_h}$', '$K_{T_r}$', '$K_{T_c}$', '$K_{D_N}$', '$K_{D}$', '$K_{C}$', '$\\delta_{A_1}$',
                              '$\\delta_{A_2}$', '$\\delta_{A_3}$', '$\\alpha_{NCA}$', '$\\delta_{C T_c A_3}$', '$f$', '$a$']
        self.par0 = parameters

    def __call__(self,t,x,par):
        # ODE right-hand side
        tau = min(t, par[13]*par[14])
        x=x*(x>0) #Q
        return np.array([self.par0[45] - (1 - np.exp((-par[0])*x[14]))*par[2]*(par[13] + 1/par[14] - tau/par[14])*x[0] - self.par0[24]*x[0] -
                         self.par0[49]*x[0]*(self.par0[0]*x[10] + self.par0[1]*x[11]), -((1 - np.exp((-par[0])*x[14]))*par[3]*(par[13] + 1/par[14] - tau/par[14])*x[1]) -
                         self.par0[25]*x[1] + x[0]*(self.par0[0]*x[10] + self.par0[1]*x[11]), self.par0[46] - (1 - np.exp((-par[0])*x[14]))*par[4]*(par[13] + 1/par[14] - tau/par[14])*
                         x[2] - self.par0[26]*x[2] - self.par0[50]*x[2]*(self.par0[2]*x[1] + self.par0[3]*x[7]) - self.par0[52]*x[2]*(self.par0[6]*x[1] + self.par0[5]*x[3] + self.par0[7]*x[7]) -
                         self.par0[4]*self.par0[51]*x[2]*x[11], -((1 - np.exp((-par[0])*x[14]))*par[5]*(par[13] + 1/par[14] - tau/par[14])*x[3]) +
                         x[2]*(self.par0[2]*x[1] + self.par0[3]*x[7]) - x[3]*(self.par0[29] + self.par0[27]*x[4] + self.par0[28]*x[11]),
                         -((1 - np.exp((-par[0])*x[14]))*par[6]*(par[13] + 1/par[14] - tau/par[14])*x[4]) - self.par0[30]*x[4] + self.par0[4]*x[2]*x[11],
                         -((1 - np.exp((-par[0])*x[14]))*par[7]*(par[13] + 1/par[14] - tau/par[14])*x[5]) + x[2]*(self.par0[6]*x[1] + self.par0[5]*x[3] + self.par0[7]*x[7]) -
                         x[5]*(self.par0[33] + self.par0[31]*x[4] + self.par0[32]*x[11]), self.par0[47] - (1 - np.exp((-par[0])*x[14]))*par[8]*(par[13] + 1/par[14] - tau/par[14])*x[6] -
                         self.par0[34]*x[6] - self.par0[53]*x[6]*(self.par0[8]*x[8] + self.par0[9]*x[13]),
                         -((1 - np.exp((-par[0])*x[14]))*par[9]*(par[13] + 1/par[14] - tau/par[14])*x[7]) - x[7]*(self.par0[36] + self.par0[35]*x[8]) +
                         x[6]*(self.par0[8]*x[8] + self.par0[9]*x[13]), -((1 - np.exp((-par[1])*x[14]))*par[10]*(par[13] + 1/par[14] - tau/par[14])*x[8]) -
                         x[8]*(self.par0[39] + self.par0[37]*x[5] + self.par0[38]*x[10]) + x[8]*(1 - x[8]/self.par0[48])*(2*self.par0[10] + self.par0[11]*x[11] + self.par0[12]*x[12]),
                         (1 - np.exp((-par[1])*x[14]))*par[10]*par[12]*(par[13] + 1/par[14] - tau/par[14])*x[8] - self.par0[40]*x[9] +
                         self.par0[54]*x[8]*(self.par0[39] + self.par0[37]*x[5] + self.par0[38]*x[10]), self.par0[13]*x[3] + self.par0[14]*x[5] - self.par0[41]*x[10],
                         self.par0[16]*x[1] + self.par0[15]*x[3] + self.par0[17]*x[8] - self.par0[42]*x[11], self.par0[19]*x[1] + self.par0[18]*x[3] + self.par0[20]*x[8] - self.par0[43]*x[12],
                         self.par0[21]*x[1] + self.par0[22]*x[7] + self.par0[23]*x[9] - self.par0[44]*x[13], (-par[11])*x[14]])


    def parameters_from_assumptions(self,data):
        #Function returning parameter values from literature
        # data is a list of arrays in the following order
        # [drug_deltas, betas, K_C, K_immune, a, alphaNCA]

        drug_deltas, betasI, betasC, K_C, K_immune, a, alphaNCA, delCTcA3, baseline_drug_dose, SS_vals = data
        par=np.empty(17)

        for i in range(1):
            par[i] = betasI[i]*baseline_drug_dose[i]/drug_deltas[i]
            par[i+1] = betasC[i]*baseline_drug_dose[i]/drug_deltas[i]
            par[i+11] = drug_deltas[i]
        for i in range(2, 10):
            par[i] = K_immune

        par[10] = K_C
        par[12] = alphaNCA*SS_vals[8]/SS_vals[9] # alphaNCA*C^inf/N^inf
        par[13] = 0.5 # f
        par[14] = a

        return par
