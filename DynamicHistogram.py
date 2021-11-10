import cv2
import numpy as np
import scipy, scipy.signal

class DHE():
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def contrast_enhancement(self, img):
        hist_i, hist_s = self.build_is_hist(img)
        hist_c = self.alpha*hist_s + (1-self.alpha)*hist_i
        hist_sum = np.sum(hist_c)
        hist_cum = hist_c.cumsum(axis=0)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        i = hsv[:,:,2]
        
        c = hist_cum / hist_sum
        s_r = (c * 255)
        i_s = np.zeros(i.shape)
        for n in range(0,255):
            i_s[i==n] = s_r[n+1]
        i_s[i==255] = 255
        hsi_o = np.stack((h,s,i_s), axis=2) 
        result = cv2.cvtColor(hsi_o.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return result

    @staticmethod
    def build_is_hist(img):
        hei = img.shape[0]
        wid = img.shape[1]
        ch = img.shape[2]
        Img = np.zeros((hei+4, wid+4, ch), dtype=np.uint8)
        for i in range(ch):
            Img[:,:,i] = np.pad(img[:,:,i], (2,2), 'edge')
        hsv = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float64)
        fh = np.array([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])
        fv = fh.conj().T
        
        H = hsv[:,:,0]
        S = hsv[:,:,1]
        I = hsv[:,:,2]

        dIh = scipy.signal.convolve2d(I, np.rot90(fh, 2), mode='same')
        dIv = scipy.signal.convolve2d(I, np.rot90(fv, 2), mode='same')
        dI = np.sqrt(dIh**2+dIv**2).astype(np.uint32)
        di = dI[2:hei+2,2:wid+2]
        
        dSh = scipy.signal.convolve2d(S, np.rot90(fh, 2), mode='same')
        dSv = scipy.signal.convolve2d(S, np.rot90(fv, 2), mode='same')
        dS = np.sqrt(dSh**2+dSv**2).astype(np.uint32)
        ds = dS[2:hei+2,2:wid+2]

        
        h = H[2:hei+2,2:wid+2]
        s = S[2:hei+2,2:wid+2]
        i = I[2:hei+2,2:wid+2]
        
        rho = np.zeros((hei,wid))
        rd = (rho*ds).astype(np.uint32)
        Hist_I = np.zeros((256,1))
        Hist_S = np.zeros((256,1))
        
        for n in range(0,255):
            temp = np.zeros(di.shape)
            temp[i==n] = di[i==n]
            Hist_I[n+1] = np.sum(temp.flatten('F'))
            temp = np.zeros(di.shape)
            temp[i==n] = rd[i==n]
            Hist_S[n+1] = np.sum(temp.flatten('F'))

        return Hist_I, Hist_S

class BPDHE():
    def contrast_enhancement(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h,s,v = cv2.split(hsv)
        h = h/255.0; s = s/255.0

        ma = np.max(v)
        mi = np.min(v)
        bins = ma - mi + 1

        hist_i = np.histogram(v,bins=bins)
        hist_i = hist_i[0].reshape(1,len(hist_i[0]))
        
        gausFilter = self.matlab_style_gauss2D()
        
        blur_hist = cv2.filter2D(hist_i.astype('float32'), -1, gausFilter, borderType=cv2.BORDER_REPLICATE)
        derivFilter = np.array([[-1,1]])
        deriv_hist = cv2.filter2D(blur_hist.astype('float32'), -1, derivFilter, borderType=cv2.BORDER_REPLICATE)
        sign_hist = np.sign(deriv_hist)
        meanFilter = np.array([[1/3,1/3,1/3]])
        smooth_sign_hist = np.sign(cv2.filter2D(sign_hist.astype('float32'),-1,meanFilter, borderType=cv2.BORDER_REPLICATE))
        cmpFilter = np.array([[1,1,1,-1,-1,-1,-1,-1]])

        p = 1
        index = [0]
        for n in range(0,bins-7):
            C = (smooth_sign_hist[0][n:n+8] == cmpFilter)*1
            if np.sum(C) == 8.0:
                p += 1
            index.append(n+3)

        index.append(bins)

        factor = np.zeros(shape=(len(index)-1,1))
        span = factor.copy()
        M = factor.copy()
        rangee = factor.copy()
        start = factor.copy()
        endd = factor.copy()
        sub_hist = []
        for m in range(0,len(index)-1):
            sub_hist.append( np.array(hist_i[0][index[m]:index[m+1]]) ) 
            M[m] = np.sum(sub_hist[m])
            low = mi + index[m]
            high = mi + index[m+1] - 1
            span[m] = high - low + 1
            factor[m] = span[m] * np.log10(M[m])
            factor_sum = np.sum(factor)
        for m in range(0,len(index)-1):
            rangee[m] = np.round((256 - mi)*factor[m]/factor_sum)
        start[0] = mi
        endd[0] = mi + rangee[0] - 1
        for m in range(1,len(index)-1):
            start[m] = start[m-1] + rangee[m-1]
            endd[m] = endd[m-1] + rangee[m]

        y = []
        s_r = np.zeros(shape=(1,mi))
        s_r = s_r.tolist()
        s_r = (s_r[0])
        for m in range(0, len(index)-1):
            hist_cum = np.cumsum(sub_hist[m]) 
            c = hist_cum/M[m]
            y.append( np.array(np.round(start[m] + (endd[m] - start[m])*c)) )
            x = y[m].tolist()
            s_r = s_r + x
        i_s = np.zeros(shape=v.shape)

        for n in range(mi , ma+1):
            lc = (v== n)
            i_s[lc] = (s_r[n])/255
        
        hsi_0 = cv2.merge([h, s, i_s])*255
        hsi_0[hsi_0>255] = 255
        hsi_0[hsi_0<0] = 0
        result = cv2.cvtColor(hsi_0.astype('uint8'), cv2.COLOR_HSV2RGB)

        return result

    @staticmethod
    def matlab_style_gauss2D(shape=(1,9),sigma=1.0762):
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp(-(x*x + y*y) / (2.*sigma*sigma) )
        h[h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h