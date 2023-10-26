import numpy as np
import imageio.v2 as imageio
from scipy import ndimage

# Assignment 3: Image descriptors
# Nome: Pedro Dias Batista
# Nusp: 10769809
# SCC0251 - 2023.01

LuminanceRatio = np.array([0.299, 0.587, 0.144, 0])
KernelX = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
KernelY =np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
conversion_M = np.array([[10, 23, 34], [43, 12, 91], [0, 92, 1]])
magnitude_M = np.array([[1, 2, 3], [3, 2, 1], [1, 2, 3]])
bins_number = 9
bins_edges = np.arange(0, 180, 20)

#Transform RGB image into a Gray one using the Luminance Method (Weights above "LuminanceRatio")
def pre_processing_Luminance(img):
    shape = img.shape
    new_img = np.zeros((shape[0], shape[1]))
    new_img = img[:,:,0]*LuminanceRatio[0] + img[:,:,1]*LuminanceRatio[1] + img[:,:,2]*LuminanceRatio[2]
    return new_img

#Digitize the angles into Bins divided in 9 ranging 20 degrees each from 0 to 180 degrees
def count_bins(M, angle, bins_edges):
    bins = np.digitize(angle, bins_edges) - 1
    dg = np.zeros(bins_number)
    shape = bins.shape
    for i in range (shape[0]):
        for j in range (shape[1]):
            dg[bins[i][j]] += abs(M[i][j])
    return dg

#Calculate the euclidian distance for given point pA and pB For any dimension 
def euclidian_distance(pA, pB):
    return np.sum((pA -  pB)**2)

#Calculate the KNN given a dataTest for N neighbours for a N amount of samples (used two in this case)
def KNN(dataTest, dataSample, N_samples, N):
    distances = []
    for i,samples in enumerate(dataSample):
        for sample in samples:
            distances.append((i, euclidian_distance(sample, dataTest)))

    distances.sort(key=lambda x:x[1])
    neighbors=distances[:3] #Get the N fisrt neighbors

    count = np.zeros((N_samples,2))
    for i in neighbors:
        count[i[0]][0] = i[0]
        count[i[0]][1] += 1
    
    prediction = max(count, key=lambda x:x[1])
    return prediction[0] 

#Histogram oriented Gradients this funcion does all the mathematical, from reading a RGB image into a discrate gradiente array
def HoG(files):
    dg = []
    for file in files:
        img = imageio.imread(file)
        img_gray = pre_processing_Luminance(img)
        img_convX = np.array(ndimage.convolve(img_gray, KernelX), dtype=float)
        img_convY = np.array(ndimage.convolve(img_gray, KernelY), dtype=float)
        M_sum = np.sum(np.sum(np.sqrt((img_convX**2) + (img_convY**2))))
        M  = np.array((np.sqrt((img_convX**2) + (img_convY**2))/M_sum), dtype=float)
        angle = np.array(np.arctan(img_convY/img_convX), dtype=float)
        angle = angle + (np.pi/2)
        angle = np.degrees(angle)
        dg.append(count_bins(M, angle, bins_edges))
    dg = np.array(dg)
    return dg


def main():
    np.seterr(divide="ignore", invalid="ignore") #in case the arctan funtion has to divide a value for 0

    #read input
    file_in_sampleA = (input().split(' '))
    file_in_sampleB = (input().split(' ')) 
    file_in_test = (input().split(' ')) 

    #open image and process
    dg_sampleA = HoG(file_in_sampleA)
    dg_sampleB = HoG(file_in_sampleB)
    dg_Test = HoG(file_in_test)
    
    dg_samples = []
    dg_samples.append(dg_sampleA)
    dg_samples.append(dg_sampleB)

    #KNN results for predictig images
    for dg in dg_Test:
        print((int)(KNN(dg, dg_samples, 2, 3)), end=" ")

if(__name__ == "__main__"):
    main()  