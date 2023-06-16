import numpy as np
import matplotlib.pyplot as plt
import time

SIR_THRESHOLD=1.0 #SIR threshold in Watt for outage
np.random.seed(1)
PB=0.01 #BS Transmit power in Watt
PN = -105   # dB
Fc=2e9 # Operating Frequency in GHz
LightSpeed=3*(10**8)
WaveLength=LightSpeed/(Fc) #wavelength in meter
FastFadingSampleSize=1 #number of signal measurements per time step

step = 1000  # include the start point at 0 and end point, the space between two sample points is D/(step-1)
D = 10
HeightMapMatrix = np.zeros(shape=(D * step+1, D * step+1))
HeighMapArray = HeightMapMatrix.reshape(1, (D * step+1) ** 2)
#probe = np.zeros([144,5])

class Urban_world(object):
    def __init__(self,GT_loc):
        print('Generate urban environment......')
        #Sample the building
     #   self.Build = np.array([664, 786, 878, 91, 793, 433, 402, 525, 984, 657])
        self.Build_num = 144
        self.BuildMapMatrix = np.zeros(shape=(self.Build_num, 5))
        self.GT_loc = GT_loc
        self.side = 0.0
        # Sample the building Heights
    def Buliding_construct(self):# This part model the distribution of buildings
        ALPHA=0.3
        BETA=1.44
        GAMA=50
        MAXHeight=50
        MINHeight = 10
        #==========================================
        #==Simulate the building locations and building size. Each building is modeled by a square
        D=10 #in km, consider the area of DxD km^2
        N=int(BETA*(D**2)) #the total number of buildings
        A=ALPHA*(D**2)/N #the expected size of each building
        Side=np.sqrt(A)
        self.side = Side
        H_vec=np.random.rayleigh(GAMA,N)
        H_vec=[min(x, MAXHeight) for x in H_vec]   # 建筑物的高度
        H_vec = [max(x, MINHeight) for x in H_vec]  # 建筑物的高度
        Actual_N = 10 #实际建筑物的数量
        #Grid distribution of buildings
        Cluster_per_side=3
        Cluster=Cluster_per_side**2
        N_per_cluster=[np.ceil(N/Cluster) for i in range(Cluster)]
        # ============================
        Road_width = 0.01  # road width in km
        Cluster_size = (D - (Cluster_per_side - 1) * Road_width) / Cluster_per_side
        Cluster_center = np.arange(Cluster_per_side) * (Cluster_size + Road_width) + Cluster_size / 2
        # =====Get the building locations=================
        XLOC = []
        YLOC = []
        for i in range(Cluster_per_side):
            for j in range(Cluster_per_side):
                Idx = i * Cluster_per_side + j
                Buildings = int(N_per_cluster[Idx])
                Center_loc = [Cluster_center[i], Cluster_center[j]]
                Building_per_row = int(np.ceil(np.sqrt(Buildings)))
                Building_dist = (Cluster_size - 2 * Side) / (Building_per_row - 1)
                X_loc = np.linspace((-Cluster_size + 2 * Side) / 2, (Cluster_size - 2 * Side) / 2, Building_per_row)
                Loc_tempX = np.array(list(X_loc) * Building_per_row) + Center_loc[0]
                Loc_tempY = np.repeat(list(X_loc), Building_per_row) + Center_loc[1]
                XLOC.extend(list(Loc_tempX[0:Buildings]))
                YLOC.extend(list(Loc_tempY[0:Buildings]))

        for i in range(N):
            x1=XLOC[i]-Side/2
            x2=XLOC[i]+Side/2
            y1=YLOC[i]-Side/2
            y2=YLOC[i]+Side/2
            HeightMapMatrix[int(np.ceil(x1 * step) - 1):int(np.floor(x2 * step)+1),int(np.ceil(y1 * step) - 1):int(np.floor(y2 * step)+1)] = H_vec[i]/100
            # probe[i] = [np.ceil(x1 * step) - 1,np.floor(x2 * step)+1,np.ceil(y1 * step) - 1,np.floor(y2 * step)+1,H_vec[i]/1000]
            self.BuildMapMatrix[i][0] =x1
            self.BuildMapMatrix[i][1] =x2
            self.BuildMapMatrix[i][2] =y1
            self.BuildMapMatrix[i][3] =y2
            self.BuildMapMatrix[i][4]=H_vec[i]/100
     #   print(self.HeightMapMatrix)

        # ===========View Building and GT distributions
        print('figure')
        plt.figure()
        # Build = np.random.randint(0,N,size=10)
        for i in range(N):
            x1 = XLOC[i] - Side / 2
            x2 = XLOC[i] + Side / 2
            y1 = YLOC[i] - Side / 2
            y2 = YLOC[i] + Side / 2
            XList = [x1, x2, x2, x1, x1]
            YList = [y1, y1, y2, y2, y1]
            plt.plot(XList, YList, 'r-')
            plt.text(XLOC[i],YLOC[i],str(np.int(H_vec[i])))

        plt.plot(self.GT_loc[:, 0], self.GT_loc[:, 1], 'bp', markersize=1)
        for i in range(len(self.GT_loc)):
            plt.text(self.GT_loc[i][0],self.GT_loc[i][1],str(i))
        plt.scatter(6.0,5.8,c='blue',marker='x')
        plt.title('location', fontsize=30)
        plt.xlim((0, D))
        plt.ylim((0, D))
        plt.grid()
        # plt.show()
        plt.savefig('Urban_'+str(len(self.GT_loc))+'.png')
        plt.close()
        return self.BuildMapMatrix
        #=================END of Building distributions================================

    # =========Main Function that determines the best outage from all BS at a given location=======
    # loc_vec: a matrix, nx3, each row is a (x,y,z) location
    # SIR_th: the SIR threshold for determining outage
    def getPointMiniOutage(self,loc_vec):
        numLoc = 1  # 只有一个位置
        for i in range(numLoc):
            PointLoc = loc_vec
            OutageMatrix,LoS_state,SNR_set = self.getPointOutageMatrix(PointLoc, SIR_THRESHOLD)  # 返回与用户的连接情况
          #  print(LoS_state)
        return OutageMatrix,LoS_state,SNR_set

    # For a given location, return the empirical outage probaibility from all sectors of all BSs
    # PointLoc:  the given point location
    # SIR_th: the SIR threshold for defining the outage
    # OutageMatrix: The average outage probability for connecting with each site, obtained by averaging over all the samples
    def getPointOutageMatrix(self,PointLoc, SNR_th):
        numGT = len(self.GT_loc)
        SignalFromUav = []
        LoS_state = []
        TotalPower = 0
        t1 = time.time()
        for i in range(len(self.GT_loc)):
            GT = self.GT_loc[i, :]  # 每一个用户的位置
            LoS = self.checkLoS(PointLoc, GT)  # 判断uav和用户之间是否存在LoS
            LoS_state.append(LoS)
            MeasuredSignal = self.getReceivedPower_RicianAndRayleighFastFading(PointLoc, GT, LoS)  # 每一个用户接收到的功率
            SignalFromUav.append(MeasuredSignal)
            TotalPower = TotalPower + MeasuredSignal
        TotalPowerAllSector = np.sum(TotalPower, axis=1)  # the interference of all power
        t2 = time.time()
     #   print(t2-t1)
     #   print('one')
        CoverMatrix = np.zeros(numGT)  # 不考虑干扰
        SNR_set = np.zeros(numGT)
        for i in range(len(self.GT_loc)):
            SignalTothisGT = SignalFromUav[i]
            SNR = SignalTothisGT / (10 ** (PN / 10))
            SNR_set[i] = SNR
            SNR_dB = 10 * np.log10(SNR)
            if SNR > SNR_th:
                CoverMatrix[i] = 1.0
        #    print(SNR)
        return CoverMatrix,LoS_state,SNR_set



    #This function check whether there is LoS between the GT and the given Loc
    def checkLoS(self,PointLoc,GT):
        SamplePoints=np.linspace(0,1,1000)   #uav与bs之间的距离采样,100个点,采样精度决定了检测Los的准确度
        XSample=GT[0]+SamplePoints*(PointLoc[0]-GT[0]) # x,y,z之间是一一对应的
        YSample=GT[1]+SamplePoints*(PointLoc[1]-GT[1])
        ZSample=GT[2]+SamplePoints*(PointLoc[2]-GT[2])
        XRange = np.int_(np.floor(XSample * (step)))
        YRange = np.int_(np.floor(YSample * (step)))  #
        XRange = [max(x, 0) for x in XRange]  # remove the negative idex
        YRange = [max(x, 0) for x in YRange]  # remove the negative idex
        XRange = [min(x, 10000) for x in XRange]  # remove the negative idex
        YRange = [min(x, 10000) for x in YRange]  # remove the negative idex
     #   Idx_vec = np.int_((np.array(XRange) * (D * step +1)+ np.array(YRange)))
        SelectedHeight = [HeightMapMatrix[XRange[i]][YRange[i]] for i in range(len(XRange))]
      #  SelectedHeight = [HeighMapArray[0, i] for i in Idx_vec]
        if any([x > y for (x, y) in zip(SelectedHeight, ZSample)]):
            return False
        else:
            return True# 所有建筑物和采样点之间都没有遮挡的话，证明存在LoS径


    # Return the received power at a location from all the three sectors of a BS
    # While the large scale path loss power is a constant for given location and site, the fast fading may change very fast.
    # Hence, we return multiple fast fading coefficients. The number of samples is determined by FastFadingSampleSize
    # A simple fast-fading implementation: if LoS, Rician fading with K factor 15 dB; otherwise, Rayleigh fading
    def getReceivedPower_RicianAndRayleighFastFading(self,PointLoc, GT, LoS):
        HorizonDistance = np.sqrt((GT[0] - PointLoc[0]) ** 2 + (GT[1] - PointLoc[1]) ** 2)
        LargeScale = self.getLargeScalePowerFromGT(PointLoc, GT,
                                              LoS)  # large-scale received power based on path loss
        # the random component, which is Rayleigh fading
        RayleighComponent = np.sqrt(0.5) * (
                    np.random.randn() + 1j * np.random.randn())

        if LoS:  # LoS, fast fading is given by Rician fading with K factor 15 dB
            K_R_dB = 15  # Rician K factor in dB
            K_R = 10 ** (K_R_dB / 10)
            threeD_distance = 100 * np.sqrt((GT[0] - PointLoc[0]) ** 2 + (GT[1] - PointLoc[1]) ** 2 + (
                        GT[2] - PointLoc[2]) ** 2)  # 3D distance in meter
          #  DetermComponent = np.exp(-1j * 2 * np.pi * threeD_distance / WaveLength)  # deterministic component
            DetermComponent = 1.0 #不知道为什么有这一项，因此先设为1
            AllFastFadingCoef = np.sqrt(K_R / (K_R + 1)) * DetermComponent + np.sqrt(1 / (K_R + 1)) * RayleighComponent
        else:  # NLoS, fast fading is Rayleigh fading
            AllFastFadingCoef = RayleighComponent

        h_overall = AllFastFadingCoef * np.sqrt(np.tile(LargeScale, (FastFadingSampleSize, 1)))
        PowerInstant = np.abs(h_overall) ** 2  # the instantneous received power in Watt
        return PowerInstant

    def getLargeScalePowerFromGT(self,PointLoc,GT,LoS):  # 路损
        ChGain=1.0 # 假设信道增益为1.0
        Distance=100*np.sqrt((GT[0]-PointLoc[0])**2+(GT[1]-PointLoc[1])**2+(GT[2]-PointLoc[2])**2) #convert to meter
        #We use 3GPP TR36.777 Urban Macro Cell model to generate the path loss
        #UAV height between 22.5m and 300m
        if LoS:
            PathLoss_LoS_dB=0.1+20*np.log10(Distance)+20*np.log10(4*np.pi*Fc/LightSpeed)
            PathLoss_LoS_Linear=10**(-PathLoss_LoS_dB/10)
            Prx=ChGain*PB*PathLoss_LoS_Linear
        else:
            PathLoss_NLoS_dB=21+20*np.log10(Distance)+20*np.log10(4*np.pi*Fc/LightSpeed)
            PathLoss_NLoS_Linear=10**(-PathLoss_NLoS_dB/10)
            Prx=ChGain*PB*PathLoss_NLoS_Linear
        return Prx
"""
world = Urban_world()
world.Buliding_construct()
##============VIew the radio map for given height
UAV_height = 0.1  # UAV height in km
X_vec = 1.0   #无人机的位置
Y_vec = 2.8
numX, numY = np.size(X_vec), np.size(Y_vec)

Loc_vec_All = np.zeros(shape=(numX * numY, 3))

Loc_vec = np.zeros(shape=(1, 3))
Loc_vec[:, 0] = X_vec
Loc_vec[:, 1] = Y_vec
Loc_vec[:, 2] = UAV_height
print(Loc_vec[0,:])
OutageMapActual = world.getPointMiniOutage(Loc_vec[0,:])     #用于检测无人机是否和用户存在有效连接
print(OutageMapActual)

#Outage_vec_All = np.reshape(OutageMapActual, numX * numY)

"""
