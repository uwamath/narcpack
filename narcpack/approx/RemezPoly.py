import numpy as np

class RemezPoly:
    
    def __init__(self, func, interval=[-1,1], n=10):
        self.func=func;
        self.interval=interval;
        
        maxIter=100


        def solve_coeff(x, n, f):
            """
            Solve the linear system of equations:
            a0+a1*xi+a2*xi^2+...+an*x^n+(-1)^i E=f(xi)
            for the unknowns a0, a1, ... an, E
            
            Input:
            x: list of (n+2) points
            n: degree of polynomial
            f: given function
            
            Return:
            p: list of coefficients
            """
            A=np.ones(n+2)
            for i in range(0, n):
                A=np.c_[A, np.power(x, i+1)]
            A=np.c_[A, np.power((-1),range(0,n+2))]
            b=np.transpose(f(x))        
            p=np.linalg.solve(A,b)      
            return p
    
    
        thresh=1e-32
        f=func
        
        y=np.linspace(interval[0], interval[1],n+2, endpoint=True)
        xf=np.linspace(interval[0], interval[1], n*10+2, endpoint=True)
        fval=f(xf)
        for i in range(0, maxIter):
            p0=solve_coeff(y, n, f)
            p=p0[0:-1]
            c=range(0,len(p))
            data=np.zeros(len(xf))
            for j in range(0,len(xf)):
                pf=np.dot(p,np.power(xf[j],c))
                data[j]=fval[j]-pf
            
            if max(data)<thresh:
                break
            
            y1=np.zeros(n+2)
            y1[0]=interval[0]
            y1[n+1]=interval[1]
            extrema_ind=np.diff(np.sign(np.diff(data))).nonzero()[0] + 1
            for j in range(0, n):
                y1[j+1]=xf[extrema_ind[j]]
            
         
            v=[data[0]]
            for j in range(0, n):
                v.append(abs(data[extrema_ind[j]]))
            v.append(data[len(data)-1])
                  
            m=max(v)
            ind=np.argmin(v)
            
            if (abs(y[ind]-y1[ind])<thresh):
                break
            if (ind<len(y) and abs(y[ind+1]-y1[ind])<thresh):
                break
            y=y1
        
        self.p=p
        
        pfun=lambda x: np.dot(p,np.power(x,c))
        self.pfun=pfun
    
    def __call__(self,x):
        y=[]
        for j in x:
            y.append(self.pfun(j))
        return y

