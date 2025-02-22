# FunctionAnalysis.py

import matplotlib.pyplot as _plt
import numpy as _np
from array import array as _array
            
if __name__ == '__main__': #run as an executable to get __name__ == '__main__'   
    def f(x: float) -> 'f(x)':
        '''This function was initialized by FunctionAnalysis'''
        return _np.tan(_np.sqrt(x**2+1)) 
    
    def df(x: float) -> "f'(x)":
        '''This function was initialized by FunctionAnalysis'''
        return _np.tan(_np.sqrt(x**2+1))




class Function():
    '''Add additional tools for analysis to a numerical function.
    **Seperate multiple functions with a comma**
            Ex: Function(f,g,h) or Function(f)'''
    
    def __new__(cls,*args):
        if len(args) == 1: #if only one function
            return cls._fromfunction(*args)
        else:
            return cls._fromlist(*args)
            
    @classmethod
    def _fromfunction(cls, function):
        '''Returns a callable object wiht methods'''
        return _Analysis(function)
    
    @classmethod
    def _fromlist(cls, *args) -> 'functions list':
        '''Returns an iterable list of objects that are each callable with methods'''
        functionlist = []
        for func in args: 
            functionlist.append(_Analysis(func)) 
        return functionlist

   
class _Analysis():
    '''Internal API'''  
    def __init__(self, function: object):
        if isinstance(function, int | float | str | list | dict | tuple | None ):
            raise ValueError('Must input a non-iterable, numerical function object.')
        self._function = function
    
    def __repr__(self):
        return f'Function({self._function})'
    
    @property
    def function(self):
        return self._function

    @function.setter
    def function(self, possible_function):
        if isinstance(possible_function, int | float | str | list | dict | tuple | None ):
            raise ValueError('Must create a non-iterable, numerical function object for analysis.')        
        self._function = possible_function
        
    def calc(self, *args) -> 'f(x) list': 
        '''Returns values of the instantiated function 
        **Seperate multiple inputs with commas**
            Ex: calc(2,3,4,5)'''   
        results = _np.array(args, float)
        func_vector = _np.vectorize(self._function)
        results = func_vector(results)
        '''for x in args:
            results.append(self._function(x))
        '''
        return results
    
    def plotfx(self, start: int, end: int, label = None, magnification: float = None, clear: bool = None, block: bool = None):
        '''Displays the result of the instantiated function, needs start/end of domain.'''
        '''"num" is value where line will appear on plot.'''
        
        # by default clear = True
        if clear == None or clear == True:
            _plt.cla()
               
        # by default the magnification = 0.01
        if magnification == None:
            magnification = 0.01

        # by default block = True
        if block == None:
            block = True

        # by default the label = _Analysis.__repr__    
        if label == None:
            label = self.function
        # set the domain 
        XVALS = [x for x in _np.arange(start,end,magnification)]

        # display results
        func_values = self.calc(*XVALS)
            
        _plt.axhline(y=0) # plot f(x)=num
        _plt.plot(XVALS,func_values, label=label)
        _plt.legend(loc=1)
        _plt.show(block=block)
        
    '''Analysis Methods'''    
    def mdpt(self,n,a,b):
        h=(b-a)/n
        x=a+h/2
        for i in range(n):
            s=s+self.calc(x)[0]
            x=x+h
        return s*h


    def trap(self,n,a,b):
        h=(b-a)/n
        x=a+h
        s=(self.calc(a)[0]+self.calc(b)[0])/2
        for i in range(n-1):
            s=s+self.calc(x)[0]
            x=x+h
        return s*h

    def simpsons(self,n,a,b):
        return 1/3*self.trap(n,a,b)+2/3*self.mdpt(n,a,b)
    
    def secant(self, p_n_minus1: float, p_n: float):
        '''Using a given interval: (p_n_minus1, p_n) find zeros of the function.
        Choose p_n_minus1 within a sufficiently small distance from  
        a suspected zero of the instantiated function'''

        # secant method
        p_n_plus1 = lambda p_n_minus1, p_n: p_n-self.calc(p_n)[0]*(p_n-p_n_minus1)/(self.calc(p_n)[0]-self.calc(p_n_minus1)[0])

        # initializing the value of current_p
        current_p = p_n_plus1(p_n_minus1, p_n)

        # Check if the error is close to zero
        while abs(current_p - p_n) > 10e-12:
            
            # move one step forward 
            p_n, p_n_minus1 = current_p, p_n 
            current_p = p_n_plus1(p_n_minus1, current_p) # and get a new p_n_plus1
            
        return current_p
    
    def bisect(self, start: int, end: int, lazy: bool=False):
        '''Perform bisection method between the given interval: ([start, end])
            choose start and end so this interval brackets at least one zero.'''

        # bisection expects start <= end
        if start>=end:
            return print(f'ERROR: ({start=} is greater than {end=}')
        
        # Check if a and b bracket any zeros
        if self.calc(start)[0]*self.calc(end)[0] >= 0:
                    return print(f'ERROR: f({start})*f({end}) >= 0')
        p_n = (1/2)*((b:=end) + (a:=start)) # n=0 here and assigning a=start and b=end
        
        #check if p_n is a zero
        while abs(a - (p_n) )  > 1e-12:

            #check for 0; if not lazy evaluation 
            if self.calc(p_n)[0]==0 and lazy == False: 
                return p_n
            elif self.calc(a)[0]*self.calc(p_n)[0] <= 0:
                b = p_n
            else:
                a = p_n
            p_n = (1/2)*(a+b)
            
        return p_n
        

    def findall(self, start: float, end: float, N: int=10000) -> list:
        '''Taking in an interval and scanning that interval for when the
            instantiated function equals zero. Returns number of zeros, and a list of zeros.
            N is the number of steps or "Magnification'''

        if start>=end:
            return print(f'{start} is not less than {b}')
        
        zero_list = _array('d', []) # this will store the values for all zeros
        dx: float = (end - start) / N
        XVAL_start, XVAL_end = int(start-3*dx), int(end+3*dx)  #adding padding proportional to dx
        zero_count: int = 0
        x_n: float = start # N=0 here        
        
        for i in range(N):
            x_n_plus1: float = x_n + dx
            
            if self.calc(x_n)[0]*self.calc(x_n_plus1)[0] < 0: # if zero was detected
                zero_list.append(self.secant(x_n,x_n_plus1)) # add a zero to list
                zero_count += 1
                
            x_n = x_n_plus1 # move x_n to x_n+1

        # Display result
        _plt.cla() #clear axes
        _plt.plot(zero_list,self.calc(*zero_list),'ro')# red dot at each zero
        self.plotfx(XVAL_start, XVAL_end, clear=False,label=f'{zero_count} zeros')# plot the function
            
        return [zero for zero in zero_list], zero_count


    def findnumber(self, start, end, M: int= 10000):
        '''
        Scans the given interval for when the function changes sign and marks it as a zero.
        "M" is the number of steps or magnification.
        '''
        
        # Step is designed to travel from left to right, not from right to left
        if start >= end: 
            return print(f'{start} was greater than or equal to {end}')

        p_m: float = start  # M=0 here
        step: float = (end - start) / M  
        num_zeros: int = 0
        
        for i in range(M):
            
            # Begin to move one step forward
            p_m_plus1 = p_m + step
            
            if self.calc(p_m)[0]*self.calc(p_m_plus1)[0] < 0:    # If a zero was detected
                num_zeros += 1 
            p_m = p_m_plus1  # move p_{m} = P_{m+1}
        
        return num_zeros


    def romb(self, tol: 'tolerance',a: 'start',b: 'end') -> float:
        """This approximates the integral of the instantiated function
        from a to b. Max level will be n"""
        n: int = 30
        rtable = _np.zeros((n,n))
        """trap(function,1,a,b) loaded into rtable[0,0]"""
        rtable[0,0]=(self.calc(a)[0]+self.calc(b)[0])*(b-a)/2
        for i in range(n):
            rtable[i+1,0]=rtable[i,0]/2+self.mdpt(2**i,a,b)/2
            for j in [k+1 for k in range(i+1)]:
                rtable[i+1-j,j]=rtable[i+2-j,j-1]+1/(4**(j)-1)*(rtable[i+2-j,j-1]-rtable[i+1-j,j-1])
            if _np.abs(rtable[0,i+1]-rtable[0,i]) < tol:
                return [rtable[0,i+1],i+2]
        return print("tolerance not met")


    def richmid(self, tol: 'tolerance',a: 'start',b: 'end') -> float:
        """This approximates the integral of the instantiated function
        from a to b. Max level will be n"""
        """this makes many more function evals than romb"""
        n: int = 30
        rtable=_np.zeros((n,n))
        """mdpt(function loaded into rtable[0,0]"""
        rtable[0,0]=self.calc((a+b)/2)[0]*(b-a)  
        for i in range(n):
            rtable[i+1,0]=self.mdpt(2**(i+1),a,b)
            for j in [k+1 for k in range(i+1)]: #j needs to start at 1
                rtable[i+1-j,j]=rtable[i+2-j,j-1]+1/(4**(j)-1)*(rtable[i+2-j,j-1]-rtable[i+1-j,j-1])
            if _np.abs(rtable[0,i+1]-rtable[0,i]) < tol:
                return [rtable[0,i+1],i+2]
        return print("tolerance not met")

    def newtons(self, df: "f'(x)", x: 'starting value', iters: 'numer of iterations' = None):
        '''Newtons Method to approximate a zero. Must input instantiated function derivative.
        Can use "f1 = Function.fromlist(f,df)" and "f1[0].newtons(f1[1].calc,x)"'''

        # default value of iterations is 8
        if iters == None:
            iters = 8
        for i in range(iters):
            x=x-self.calc(x)[0]/df(x) #Newtons method
        return f'x={_np.ravel(x)}, f(x)={_np.ravel(self.calc(x)[0])}' # return (x, f(x)) as non-iterable

    def endpoint(self, value: 'integral_0-to-z_f(x) = value', x_0: "First x for Newton's Method")-> 'z':
        '''Using Newton's method with the approximation of an integral to approximate z.
        f(x) is instantiated function.
        ***Convergence is not guaranteed, interrupt if it does not return right away.***'''
        
        def H(z: float) -> 'H(z)':
            return self.romb(10**-10,0,z)[0] - value
        
        f2 = Function._fromfunction(H)

        return f2.newtons(self._function,x_0)

    def localsqrt(self,a: float) -> 'sqrt(a)':
            
        #by default p_0 = a/2
        current_p = a/2 
        
        # Fixed point function whose attractors will be the sqrt{a}
        p_n = lambda p: (p/2)+(a/(2*p))

        while current_p - p_n(current_p) > 10**-12:
            current_p = p_n(current_p)
        return current_p

        
 
