import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set_context("paper")
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


cxtfree = [50, 100, 200, 300]
w2vdkm = np.array([ -0.4302, -0.379, -0.3418, -0.291])
ftdkm = np.array([-0.4622, -0.4442, -0.4420, -0.3819])
glvdkm = np.array([ -0.3446, -0.25443, -0.092, -0.0431])
spdkm = np.array([ -0.3026, 0.047774, 0.158096, 0.18])

bert = [ 50, 100, 200, 300, 500, 768]
btdkm = [-0.033, 0.0526, 0.1391, 0.16519, 0.175265, 0.17566]
btadkm = [ -0.1207, 0.0526, 0.1391, 0.1930, 0.19605, 0.20074]
#btfdkm = [-0.077, 0.0701, 0.1107, 0.1659, 0.1803, 0.1779]



plt.plot(cxtfree, w2vdkm, marker = "o",alpha=0.7, markersize=7, label='word2vec')
plt.plot(cxtfree, ftdkm, marker = "o",alpha=0.7, markersize=7,label='FastText')
plt.plot(cxtfree, glvdkm, marker = "o",alpha=0.7, markersize=7, label='Glove')
plt.plot(cxtfree, spdkm, marker = "o",alpha=0.8,markersize=7,label='Spherical')

plt.plot(bert, btdkm, marker = "o", alpha=0.7,markersize=7,label='BERT Reduced')
plt.plot(bert, btadkm, marker = "o",alpha=0.7, markersize=7,label='BERT Average')
#plt.plot(bert, btfdkm, marker = "o", label='Bert First Word w/ Dup')



plt.legend()
plt.set_cmap('jet')
plt.ylim(-.48, 0.30)
plt.xticks([100, 300, 500, 750])
plt.xlabel("Dimensions")
plt.ylabel("NPMI score")
#plt.axes.set_xscale(1, 'log')
#plt.title("Word2Vec")
plt.savefig('img1.png')
plt.clf()


cxtfree = [ 50, 100, 200, 300]
w2vdgm = np.array([ -0.01849, 0.0108, 0.14008,0.0908])
ftdgm = np.array([ 0.1174, 0.1770, 0.17698, 0.2117])
glvdgm = np.array([0.0724, 0.162636, 0.2061, 0.23268])
spdgm = np.array([0.04839, 0.175428, 0.2463, 0.24962])

bert = [ 50, 100, 200, 300, 500, 768]
btdgm = [ 0.18046, 0.2303,0.2350, 0.24244, 0.2410, 0.2418]
btadgm  = [ 0.1803, 0.23302,0.2444,0.24663, 0.2425, 0.2572]
#btfdgm  = [ 0.1889, 0.239084, 0.2204, 0.2443, 0.2404, 0.2255]


plt.plot(cxtfree, w2vdgm, marker = "o",alpha=0.7,markersize=7, label='word2vec')
plt.plot(cxtfree, ftdgm, marker = "o",alpha=0.7,markersize=7, label='FastText')
plt.plot(cxtfree, glvdgm, marker = "o",alpha=0.7,markersize=7, label='Glove')
plt.plot(cxtfree, spdgm, marker = "o", alpha=0.8,markersize=7,label='Spherical')

plt.plot(bert, btdgm, marker = "o",  alpha=0.7,markersize=7,label='BERT Reduced')
plt.plot(bert, btadgm, marker = "o",  alpha=0.7,markersize=7,label='BERT Average')
#plt.plot(bert, btfdgm, marker = "o", label='Bert First Word w/ Dup')

plt.legend()
plt.ylim(-.48, 0.30)
plt.xticks([100, 300, 500, 750])
plt.xlabel("Dimensions")
plt.ylabel("NPMI score")
#plt.title("Word2Vec")
plt.savefig('img3.png')


plt.clf()


cxtfree = [ 50, 100, 200, 300]
w2vdkmr = np.array([0.1850,0.174033, 0.1470, 0.1657])
ftdkmr = np.array([ 0.2321, 0.2429, 0.2716, 0.2117])
glvdkmr= np.array([ 0.1834, 0.20553, 0.21382, 0.23268])
spdkmr = np.array([ 0.262314, 0.279952,0.273146, 0.282])

bert = [ 50, 100, 200, 300, 500, 768]
btdkmr = [0.2627, 0.2527,0.2572, 0.2493, 0.24862, 0.2504]
btadkmr  = [.2760, 0.2703,0.2603,0.2772,0.2660, 0.25174]
#btfdkmr  = [0.2559, 0.2698, 0.2542,0.2314,0.2618, 0.2579]

#elmo = [20, 50, 100, 200, 300, 500, 700, 1024]
#elkr = [0.1479, 0.17214, 0.16102, 0.139, 0.1250, 0.1243,  0.13243, 0.1394]

plt.plot(cxtfree, w2vdkmr, marker = "o", alpha=0.7,markersize=7, label='word2vec')
plt.plot(cxtfree, ftdkmr, marker = "o", alpha=0.7,markersize=7,label='FastText')
plt.plot(cxtfree, glvdkmr, marker = "o",alpha=0.7,markersize=7, label='Glove')

plt.plot(cxtfree, spdkmr, marker = "o",  alpha=0.8,markersize=7,label='Spherical')

plt.plot(bert, btdkmr, marker = "o",alpha=0.7,markersize=7, label='BERT Reduced')
plt.plot(bert, btadkmr, marker = "o",alpha=0.7, markersize=7,label='BERT Average')

#plt.plot(bert, btfdkmr, marker = "o", label='Bert First Word w/ Dup')

#plt.plot(elmo, elkr, marker = "o", label='Elmo w/ Dup')


plt.legend()
plt.ylim(-.48, 0.30)
plt.xticks([100, 300, 500, 750])
plt.xlabel("Dimensions")
plt.ylabel("NPMI score")
#plt.title("Using Rerank")
plt.savefig('img2.png')
