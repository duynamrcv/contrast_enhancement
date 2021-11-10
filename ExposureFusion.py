import cv2
import numpy as np
import scipy, scipy.misc, scipy.signal

class EFF():
    def __init__(self, mu=0.5, lamda = 0.5, sigma = 5, sharpness=0.001, a=-0.3293, b=1.1258):
        # The parameters is considered in https://github.com/baidut/OpenCE/blob/master/ours/Ying_2017_CAIP.m
        self.mu = mu
        self.lamda = lamda
        self.sigma = sigma
        self.sharpness = sharpness
        self.a = a; self.b = b
    
    def contrast_enhancement(self, img):
        I = cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)

        # Weight matrix estimation
        t_b = np.max(I, axis=2)
        t_our = cv2.resize(self.tsmooth(scipy.misc.imresize(t_b, 0.5, interp='bicubic', mode='F')),
                            (t_b.shape[1], t_b.shape[0]),
                            interpolation=cv2.INTER_AREA)
        
        # Apply camera model with k(exposure ratio)
        isBad = t_our < 0.5
        J = self.max_entropy_enhance(I, isBad)

        # W: Weight Matrix
        t = np.zeros((t_our.shape[0], t_our.shape[1], I.shape[2]))
        for i in range(I.shape[2]):
            t[:,:,i] = t_our
        W = t**self.mu

        I2 = I*W
        J2 = J*(1-W)

        result = I2 + J2
        result = result * 255
        result[result > 255] = 255
        result[result<0] = 0
        return result.astype(np.uint8)

    def tsmooth(self, img):
        I = cv2.normalize(img.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        x = np.copy(I)
        wx, wy = self.computeTextureWeights(x)
        S = self.solveLinearEquation(I, wx, wy)
        return S

    def computeTextureWeights(self, fin):
        dt0_v = np.vstack((np.diff(fin, n=1, axis=0), fin[0,:]-fin[-1,:]))
        dt0_h = np.vstack((np.diff(fin, n=1, axis=1).conj().T, fin[:,0].conj().T-fin[:,-1].conj().T)).conj().T

        gauker_h = scipy.signal.convolve2d(dt0_h, np.ones((1,self.sigma)), mode='same')
        gauker_v = scipy.signal.convolve2d(dt0_v, np.ones((self.sigma,1)), mode='same')

        W_h = 1/(np.abs(gauker_h)*np.abs(dt0_h) + self.sharpness)
        W_v = 1/(np.abs(gauker_v)*np.abs(dt0_v) + self.sharpness)

        return  W_h, W_v
    
    def solveLinearEquation(self, IN, wx, wy):
        [r, c] = IN.shape
        k = r * c
        dx =  -self.lamda * wx.flatten('F')
        dy =  -self.lamda * wy.flatten('F')
        tempx = np.roll(wx, 1, axis=1)
        tempy = np.roll(wy, 1, axis=0)
        dxa = -self.lamda *tempx.flatten('F')
        dya = -self.lamda *tempy.flatten('F')
        tmp = wx[:,-1]
        tempx = np.concatenate((tmp[:,None], np.zeros((r,c-1))), axis=1)
        tmp = wy[-1,:]
        tempy = np.concatenate((tmp[None,:], np.zeros((r-1,c))), axis=0)
        dxd1 = -self.lamda * tempx.flatten('F')
        dyd1 = -self.lamda * tempy.flatten('F')
        
        wx[:,-1] = 0
        wy[-1,:] = 0
        dxd2 = -self.lamda * wx.flatten('F')
        dyd2 = -self.lamda * wy.flatten('F')
        
        Ax = scipy.sparse.spdiags(np.concatenate((dxd1[:,None], dxd2[:,None]), axis=1).T, np.array([-k+r,-r]), k, k)
        Ay = scipy.sparse.spdiags(np.concatenate((dyd1[None,:], dyd2[None,:]), axis=0), np.array([-r+1,-1]), k, k)
        D = 1 - ( dx + dy + dxa + dya)
        A = ((Ax+Ay) + (Ax+Ay).conj().T + scipy.sparse.spdiags(D, 0, k, k)).T
        
        tin = IN[:,:]
        tout = scipy.sparse.linalg.spsolve(A, tin.flatten('F'))
        OUT = np.reshape(tout, (r, c), order='F')
        
        return OUT

    def max_entropy_enhance(self, I, isBad):
        # Esatimate k
        tmp = cv2.resize(I, (50,50), interpolation=cv2.INTER_AREA)
        tmp[tmp<0] = 0
        tmp = tmp.real
        Y = self.rgb2gm(tmp)
        
        isBad = isBad * 1
        isBad = scipy.misc.imresize(isBad, (50,50), interp='bicubic', mode='F')
        isBad[isBad<0.5] = 0
        isBad[isBad>=0.5] = 1
        Y = Y[isBad==1]
        
        if Y.size == 0:
            J = I
            return J
        
        f = lambda k: -self.entropy(self.applyK(Y, k))
        opt_k = scipy.optimize.fminbound(f, 1, 7)
        
        # Apply k
        J = self.applyK(I, opt_k) - 0.01
        return J
    
    def applyK(self, I, k):
        f = lambda x: np.exp((1-x**self.a)*self.b)
        beta = f(k)
        gamma = k**self.a
        J = (I**gamma)*beta
        return J

    @staticmethod
    def rgb2gm(I):
        if (I.shape[2] == 3):
            I = cv2.normalize(I.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            I = np.abs((I[:,:,0]*I[:,:,1]*I[:,:,2]))**(1/3)
        return I
    
    @staticmethod
    def entropy(X):
        tmp = X * 255
        tmp[tmp > 255] = 255
        tmp[tmp<0] = 0
        tmp = tmp.astype(np.uint8)
        _, counts = np.unique(tmp, return_counts=True)
        pk = np.asarray(counts)
        pk = 1.0*pk / np.sum(pk, axis=0)
        S = -np.sum(pk * np.log2(pk), axis=0)
        return S