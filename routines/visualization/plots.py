#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np

class Plot():
    def __init__(self):
        plt.ion()
        self.f, (self.ax1, self.ax2, self.ax3) = plt.subplots(3,1)
    
    def update(self, t, error_list, p_list, smm_list):
        self.ax1.clear()
        y_ax1 = np.linspace(0.0,15.0,100)
        x_ax = np.linspace(2.0,2.0,100)
        self.ax1.plot(t, error_list)
        #self.ax1.plot(x_ax, y_ax1, 'r--')
        self.ax1.set_ylabel(r'|prediction error| (m)')
        self.ax1.set_ylim([0.0, 15.0])
        self.ax1.set_xlim([0,7])

        self.ax2.clear()
        y_ax2 = np.linspace(-0.1,1.0,100)
        self.ax2.plot(t, p_list)
        #self.ax2.plot(x_ax, y_ax2, 'r--')
        self.ax2.set_ylabel(r'$p$')
        self.ax2.set_ylim([-0.1,1])
        self.ax2.set_xlim([0,7])

        self.ax3.clear()
        y_ax3 = np.linspace(-5.0,30.0,100)
        self.ax3.plot(t, smm_list)
        #self.ax3.plot(x_ax, y_ax3, 'r--')
        self.ax3.set_ylabel(r'$\log M$')
        self.ax3.set_ylim([-5.0,30.0])
        self.ax3.set_xlim([0,7])

        plt.pause(0.01)
    
    def close(self):
        plt.close()
