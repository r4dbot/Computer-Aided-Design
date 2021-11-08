#!/usr/bin/python3

import numpy as np
import modern_robotics as mr
import math

'''
Course 2 Project
'''

def IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev,
        filename='iterates.csv'):
    """
    this function is modified function of IKinBody from modern_robotics package
    the function consist of
    
    IKinBody
    printIterationsummary  (print all iteration information)
    writeJointValues  (Writes iteration joint values to CSV file.)
    
    
    
    Computes inverse kinematics in the body frame for an open chain robot
    outputting information at each iteration.
    
    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :param filename: The CSV filename to which joint value for each iteration
                     are output. Default is 'iterates.csv'.
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.
    
    Example Input:
        Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
    Output:
        (np.array([1.57073819, 2.999667, 3.14153913]), True)
    Writes information about each iteration to console and writes joint values
    for each iteration to CSV file.
    """

    '''
  printIterationSummary  Prints iteration information to console.
    '''
    def printIterationSummary(i, thetalist, Tb, Vb, errorAngular, errorLinear):
        print('Iteration %d' %i)
        print('joint vector: %s' %thetalist)
        print('SE(3) end effector configuration:','\n','%s' %Tb)
        print('twist error: %s' %Vb)
        print('angular error magnitude: %s' %errorAngular)
        print('linear error magnitude: %s' %errorLinear)
        print('')

        
    '''
  writeJointValues  Writes iteration joint values to CSV file.
    '''
    def writeJointValues(file, thetalist):
        for i in range(len(thetalist)):
            file.write('%s' %thetalist[i])
            if i < len(thetalist) - 1:
                file.write(',')
        file.write('\n')

    file = open(filename, 'w')

    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 40
    error = True
    Vb = np.zeros(6) #this us= twist error
    while error and i < maxiterations:
        thetalist = thetalist + np.dot(np.linalg.pinv(mr.JacobianBody(Blist, thetalist)), Vb)
        Tb = mr.FKinBody(M, Blist, thetalist) # current end effector configuration
        Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(Tb), T)))
        errorAngular = np.linalg.norm([Vb[0], Vb[1], Vb[2]]) # twist angular error
        errorLinear = np.linalg.norm([Vb[3], Vb[4], Vb[5]]) # twist linear error
        error = (errorAngular > eomg) or (errorLinear > ev)
        printIterationSummary(i, thetalist, Tb, Vb, errorAngular, errorLinear)
        writeJointValues(file, thetalist)
        i += 1
    file.close()

    return (thetalist, not error)

def example4_5():
    W1 = 0.109 # meters
    W2 = 0.082
    L1 = 0.425
    L2 = 0.392
    H1 = 0.089
    H2 = 0.095

    M = np.array([[-1, 0, 0, L1 + L2],
                  [0, 0, 1, W1 + W2],
                  [0, 1, 0, H1 - H2],
                  [0, 0, 0, 1]])
    
    Blist = np.array([[       0,        0,   0,  0,   0,   0],
                      [       0,        0,   0,  0,  -1,   0],
                      [       1,        1,   1,  1,   0,   1],
                      [ W1 + W2,       H2,  H2, H2, -W2,   0],
                      [       0, -(L1+L2), -L2,  0,   0,   0],
                      [ L1 + L2,        0,   0,  0,   0,   0]]) 
    
    T = np.array([[0, 1, 0, -0.5],
                  [0, 0, -1, 0.1],
                  [-1, 0, 0, 0.1],
                  [0, 0, 0, 1]])
    eomg = 0.001
    ev = 0.0001
    thetalist0 = np.array([0, 4, 5, 4, 3, 5]) #initial guest
    
    IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev)
    
    #thetalist0 = np.array([0, 3*math.pi/4, 2*math.pi/2, 5*math.pi/4, math.pi, 5*math.pi/2])
    #thetalist0 = np.array([-0.2, 5, 3.9, 4.1, 3, 4.5])

if __name__ == '__main__':
    example4_5()
    
