
"""
    The environment building is modified based on the following repository:
    reference code link: https://github.com/xuxiaoli-seu/SNARM-UAV-Learning
"""


import numpy as np
import matplotlib.pyplot as plt


SNR_THRESHOLD=1.0 #SIR threshold in Watt for outage
np.random.seed(1)
PB=0.01 #Transmit power in Watt
PN = -105   # dB
Fc=2e9 # Operating Frequency in GHz
LightSpeed=3*(10**8)
WaveLength=LightSpeed/(Fc) #wavelength in meter

step = 1000  # include the start point at 0 and end point, the space between two sample points is D/(step-1)
D = 10
HeightMapMatrix = np.zeros(shape=(D * step+1, D * step+1))
HeighMapArray = HeightMapMatrix.reshape(1, (D * step+1) ** 2)

class Urban_world(object):
    def __init__(self,GT_loc):
        print('Generate urban environment......')
        #Sample the building
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
        D=10
        N=int(BETA*(D**2)) #the total number of buildings
        A=ALPHA*(D**2)/N #the expected size of each building
        Side=np.sqrt(A)
        self.side = Side
        H_vec=np.random.rayleigh(GAMA,N)
        H_vec=[min(x, MAXHeight) for x in H_vec]
        H_vec = [max(x, MINHeight) for x in H_vec]
        #Grid distribution of buildings
        Cluster_per_side=3
        Cluster=Cluster_per_side**2
        N_per_cluster=[np.ceil(N/Cluster) for i in range(Cluster)]
        # ============================
        Road_width = 0.01  # road width
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
            self.BuildMapMatrix[i][0] =x1
            self.BuildMapMatrix[i][1] =x2
            self.BuildMapMatrix[i][2] =y1
            self.BuildMapMatrix[i][3] =y2
            self.BuildMapMatrix[i][4]=H_vec[i]/100

        # ===========View Building and GT distributions
        print('figure')
        plt.figure()
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

    # ================
    def getPointMiniOutage(self,loc_vec):
        PointLoc = loc_vec
        OutageMatrix,LoS_state,SNR_set = self.getPointOutageMatrix(PointLoc, SNR_THRESHOLD)
        return OutageMatrix,LoS_state,SNR_set


    def getPointOutageMatrix(self,PointLoc, SNR_th):
        numGT = len(self.GT_loc)
        SignalFromUav = []
        LoS_state = []
        TotalPower = 0
        for i in range(len(self.GT_loc)):
            GT = self.GT_loc[i, :]
            LoS = self.checkLoS(PointLoc, GT)
            LoS_state.append(LoS)
            MeasuredSignal = self.getReceivedPower_RicianAndRayleighFastFading(PointLoc, GT, LoS)
            SignalFromUav.append(MeasuredSignal)
            TotalPower = TotalPower + MeasuredSignal
        CoverMatrix = np.zeros(numGT)
        SNR_set = np.zeros(numGT)
        for i in range(len(self.GT_loc)):
            SignalTothisGT = SignalFromUav[i]
            SNR = SignalTothisGT / (10 ** (PN / 10))
            SNR_set[i] = SNR
            if SNR > SNR_th:
                CoverMatrix[i] = 1.0
        return CoverMatrix,LoS_state,SNR_set



    #This function check whether there is LoS between the GT and the given Loc

    def checkLoS(self,PointLoc,GT):
        SamplePoints=np.linspace(0,1,1000)
        XSample=GT[0]+SamplePoints*(PointLoc[0]-GT[0])
        YSample=GT[1]+SamplePoints*(PointLoc[1]-GT[1])
        ZSample=GT[2]+SamplePoints*(PointLoc[2]-GT[2])
        XRange = np.int_(np.floor(XSample * (step)))
        YRange = np.int_(np.floor(YSample * (step)))  #
        XRange = [max(x, 0) for x in XRange]  # remove the negative idex
        YRange = [max(x, 0) for x in YRange]  # remove the negative idex
        XRange = [min(x, 10000) for x in XRange]
        YRange = [min(x, 10000) for x in YRange]
        SelectedHeight = [HeightMapMatrix[XRange[i]][YRange[i]] for i in range(len(XRange))]
        if any([x > y for (x, y) in zip(SelectedHeight, ZSample)]):
            return False
        else:
            return True


    # A simple fast-fading implementation: if LoS, Rician fading with K factor 15 dB; otherwise, Rayleigh fading
    def getReceivedPower_RicianAndRayleighFastFading(self,PointLoc, GT, LoS):
        LargeScale = self.getLargeScalePowerFromGT(PointLoc, GT,
                                              LoS)  # large-scale received power based on path loss
        # the random component, which is Rayleigh fading
        RayleighComponent = np.sqrt(0.5) * (
                    np.random.randn() + 1j * np.random.randn())

        if LoS:  # LoS, fast fading is given by Rician fading with K factor 15 dB
            K_R_dB = 15  # Rician K factor in dB
            K_R = 10 ** (K_R_dB / 10)
            AllFastFadingCoef = np.sqrt(K_R / (K_R + 1)) + np.sqrt(1 / (K_R + 1)) * RayleighComponent
        else:  # NLoS, fast fading is Rayleigh fading
            AllFastFadingCoef = RayleighComponent

        h_overall = AllFastFadingCoef * np.sqrt(LargeScale)
        PowerInstant = np.abs(h_overall) ** 2  # the instantneous received power in Watt
        return PowerInstant

    def getLargeScalePowerFromGT(self,PointLoc,GT,LoS):  # pathloss
        ChGain=1.0
        Distance=100*np.sqrt((GT[0]-PointLoc[0])**2+(GT[1]-PointLoc[1])**2+(GT[2]-PointLoc[2])**2) #convert to meter
        if LoS:
            PathLoss_LoS_dB=0.1+20*np.log10(Distance)+20*np.log10(4*np.pi*Fc/LightSpeed)
            PathLoss_LoS_Linear=10**(-PathLoss_LoS_dB/10)
            Prx=ChGain*PB*PathLoss_LoS_Linear
        else:
            PathLoss_NLoS_dB=21+20*np.log10(Distance)+20*np.log10(4*np.pi*Fc/LightSpeed)
            PathLoss_NLoS_Linear=10**(-PathLoss_NLoS_dB/10)
            Prx=ChGain*PB*PathLoss_NLoS_Linear
        return Prx