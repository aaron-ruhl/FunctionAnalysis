# FunctionAnalysis 
University of Houston - Numerical Analysis, Dr. Jeffrey Morgan


This was made purely for learning purposes based on a course taught by Dr. Jeffrey Morgan. These functions behave reasonably well together as one large class. However, one must use the tools carefully. Each method has assumptions that must be satisfied so bugs can also be of a mathematical nature; especially in terms of convergence. I have not added sophisticated error or case handling.


For example, 

try 'f1 = Function(f)', then 'findall(0,2)'. This will produce the wrong result, but 'findall(1,2)' will work as intended and find the correct zero of f on the interval (0,2). 
