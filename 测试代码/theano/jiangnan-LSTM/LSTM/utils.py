import numpy as np

def data8(n=100):
    # Generates '8' shaped data
    y = np.linspace(0,1, n)  #创建一个一位数组，第一个参数表示起始点、第二个参数表示终止点，第三个参数表示数列的个数。

    x = np.append(np.sin(2*np.pi*y), (-np.sin(2*np.pi*y)))

    return np.column_stack((x,np.append(y,y))).astype(dtype=np.float32)  #Stack 1-D arrays as columns into a 2-D array.

