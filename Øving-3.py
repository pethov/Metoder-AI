import numpy

t=0

TP= numpy.array([[0.7,0.3],[0.3,0.7]])

EP= numpy.array([[0.9,0.2],[0.1,0.8]])

e15= (True,True,False,True,True)

pi= numpy.array([0.5,0.5])

for t in range(5):
    prediction = numpy.dot(TP, pi)
    obs_index = 0 if e15[t] else 1
    pi = EP[obs_index] * prediction
    pi= pi/sum(pi)
    print(f"f_1:{t+1} =", pi) 

