# coding:utf-8
import numpy as np
from matplotlib import pyplot as plt

class dummy_Hand:
    def __init__(self, init_z):
        self.present_z=init_z
        self.array_size = 1

    def simu_z(self, limit_max, limit_min):
        assumed_dz = np.random.uniform(low=-0.01,high=0.01,size=self.array_size)
        assumed_z = assumed_dz + self.present_z
        if assumed_z > limit_max:
            assumed_z = assumed_z - assumed_dz
        elif assumed_z < limit_min:
            assumed_z = assumed_z - assumed_dz

        return assumed_z

    def get_ir(self):
        limit_max=1
        limit_min=0
        z = self.simu_z(limit_max,limit_min)
        return z

if __name__ == '__main__':
    #python hand_motion.py <with no argument>
    z_depth = 0
    hand_simu=dummy_Hand(z_depth)
    loop_val = 100
    zlimit_max = 1
    zlimit_min = 0
    z=np.zeros(loop_val)
    for i in range(loop_val):
        z[i]=z_depth
        z_depth = hand_simu.simu_z(zlimit_max,zlimit_min)
    plt.plot(z)
    plt.show()
    print("end")

    #np.savetxt('test_trajectory.csv',z,delimiter=',')
