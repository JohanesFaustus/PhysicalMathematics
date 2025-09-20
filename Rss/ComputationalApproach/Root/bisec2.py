import numpy as np
def bisec(f,x1,x2,eps=10**(-17)):
    loop=0
    if f(x1)*f(x2)>=0:
        return ValueError("")
    else:
        xt=(x1+x2)/2
        while np.abs(f(xt))>eps:
            if f(x1)*f(xt)<0:
                x2=xt 
            else:
                x1=xt 
            xt=(x1+x2)/2
            loop+=1
        return xt,loop

def f(x):
    f= lambda x: x**2-x-6
    return f(x)

x1=-1
x2=4
print(bisec(f,x1,x2))