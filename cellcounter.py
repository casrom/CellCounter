import cv2, sys
import numpy as np
import scipy.ndimage
import matplotlib as mpl
from skimage.measure import label, regionprops
mpl.use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import argparse, os
import sys
import time, math

parser = argparse.ArgumentParser(description='Specifically designed for counting cells in multiple clusters.')
parser.add_argument('f', type=str, help='image file to be processed')
parser.add_argument('-gt', '--global_threshold', type=int, default=50, help='default(50) Global threshold value for detecthing clusters.')
parser.add_argument('-lt', '--local_threshold', type=int, default=60, help='default(60) Local threshold value for cleaning up background noise when detecting cells.')
parser.add_argument('-s', '--sigma', type=int, default=9, help='default(9) Gaussian blur sigma value for uifying within cell peaks.')
parser.add_argument('-mff', '--maximum_filter_size', type=int, default=7, help='default(7) Odd Integer. Radius within which local maxima are considered a single maximum. Smaller radius gives higher sensitivity.')
parser.add_argument('-c', '--channel', type=str, default='b', help='default(b) Channel r, g, or b to be analyzed.')

#Additioanal images
parser.add_argument('-f2', '--additional_file', type=str, default=None, help='Additional overlay image for counting in a different channel.')

parser.add_argument('-o', '--output', type=str, default=None, help='default(output/<inputfilename>.csv/jpg) Output file name.')


args = parser.parse_args()


# Global Threshold
gt = args.global_threshold
# Local Thresholds
t = args.local_threshold #initial image thresholding (remove background noise)
sigma = args.sigma #gaussian blur params (reduce within cell multiple peaks)
mff_size = args.maximum_filter_size #maximum filter size, used in finding local maxima (smaller gives more sensitivity)



start_G = time.time()
##Global Segmentation (finding large blobs)
sys.stdout.write('Performing global segmentation ... ')
sys.stdout.flush()
start = time.time()

img = cv2.imread(args.f)
b,g,r = cv2.split(img)
gimb = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
gimb = cv2.GaussianBlur(eval(args.channel),(99,99),0)
gimb[gimb<gt] = 0
gimb[gimb>gt] = 255
kernel1 = np.ones((5,5),np.uint8)
kernel2 = np.ones((100,100),np.uint8)
kernel3 = np.ones((25,25),np.uint8)


gimb = cv2.erode(gimb,kernel3,iterations = 1)
gimb = cv2.dilate(gimb,kernel2,iterations = 1)
gimb[gimb<0] = 0
gimb[gimb>0] = 255

gimb = cv2.resize(gimb, (img.shape[1], img.shape[0]), fx=2, fy=2,interpolation = cv2.INTER_AREA) 
GL = label(gimb)
Gprops = regionprops(GL)

end = time.time()
sys.stdout.write('Done. ('+str(int(end-start))+'s)\n')
sys.stdout.flush()

##Cell Labeling
def cell_label(img):
    im = cv2.GaussianBlur(img,(sigma,sigma),0)

    im[im<t] = 0
    # ret,th = cv2.threshold(im,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,25,-1)
    k = np.ones((10,10),np.uint8)
    M = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel1)
    M = cv2.erode(M,kernel1,iterations = 1)
    M = cv2.dilate(M,kernel1,iterations = 1)
    M[M!=255] = 0
    M[M==255] = 1
    im = np.multiply(im,M)
    

    im = cv2.GaussianBlur(im,(sigma,sigma),0)
    
    mx = scipy.ndimage.filters.maximum_filter(im,mff_size)
    msk = (im == mx)
    background = (mx==0)
    pks = msk ^ background
    # cv2.imshow('image',np.array(msk, dtype=np.float))
    # cv2.waitKey(0)
    #Cell Labels
    L = label(pks)
    props = regionprops(L)
    return (L, props)



##Plot
# sys.stdout.write('Performing cell segmentation ... ')
# sys.stdout.flush()
# start = time.time()

# L, props = cell_label(eval(args.channel))
# L2 = None
# props2 = None

# end = time.time()
# sys.stdout.write('Done. ('+str(int(end-start))+'s)\n')
# sys.stdout.flush()



if args.additional_file:
    sys.stdout.write('Overlay image specified, loading ... ')
    sys.stdout.flush()

    start = time.time()
    img2 = cv2.imread(args.additional_file)
    b2,g2,r2 = cv2.split(img2)
    end = time.time()

    sys.stdout.write('Done. ('+str(int(end-start))+'s)\n')
    sys.stdout.flush()




# from matplotlib.pyplot import figure
# plt.rc('font', size=4)  

# fig,ax = plt.subplots(1)
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# ax.imshow(gimb, alpha=0.1)

log = []
total_cluster = GL.max()
counter = 0

print('Found ' +str(total_cluster)+ ' potential clusters.')


# for each potential cluster (each global label)
for i in range(1,GL.max()):
    bbox = Gprops[i].bbox
    if bbox[0] > 10 and bbox[1] > 10 and bbox[2] < GL.shape[0]- 10 and bbox[3] < GL.shape[1]-10:
        # rect = patches.Rectangle((bbox[1],bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0], linewidth=0.5, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)
        patch_M = GL[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        patch_M[patch_M != Gprops[i].label] = 0
        patch_M[patch_M == Gprops[i].label] = 1
        patch = b[bbox[0]:bbox[2],bbox[1]:bbox[3]]
        patch = np.multiply(patch,patch_M)
        l,props = cell_label(np.array(patch, dtype=np.uint8))

        log_entry = [counter+1]
        # get cell points
        #l = np.multiply(L,(GL==i))
        #N = 1
        #props = regionprops(l)

        # plot cell points
        for r in props:
            y = int(r.centroid[0] + bbox[0])
            x = int(r.centroid[1] + bbox[1])
            img[y-1:y+1,x-1:x+1,:] = [0, 0, 255]
            #plt.plot(r.centroid[1] + bbox[1] ,r.centroid[0] + bbox[0] ,'y.', markersize=0.05)

        # Draw Boxes!
            cv2.rectangle(img,(bbox[1],bbox[0]), (bbox[3], bbox[2]),(0,255,255),4)
            

        # box = [int(v) for v in regionprops(label(GL==i))[0].bbox]
        log_entry.append(len(props))

        if args.additional_file:
            patch = b2[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            patch = np.multiply(patch,patch_M)
            l,props = cell_label(np.array(patch, dtype=np.uint8))
            for r in props:
                y = int(r.centroid[0] + bbox[0])
                x = int(r.centroid[1] + bbox[1])
                img[y-3:y+3,x-3:x+3,:] = [0, 0, 255]

            log_entry.append(len(props))


        #     # box = [int(v) for v in regionprops(label(GL==i))[0].bbox]
        #     # plt.text((box[3]+box[1])/2,box[2]+50,'['+str(i)+']'+str(len(props))+'|'+str(len(props2)),color='red')
        # else:
            # plt.text((box[3]+box[1])/2,box[2]+50,'['+str(i)+']'+str(len(props)),color='red')
        cv2.putText(img,str(log_entry[0])+' '+str(log_entry[1:]),(bbox[1],bbox[2]-3), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2)
        log.append(log_entry)
        counter += 1
    loading_bar = '['+int(1.0*i/total_cluster*10)*'>'+(10-int(1.0*i/total_cluster*10))*'-'+']'
    sys.stdout.write('\r' + '>>> Analyzing ... '+loading_bar+'('+str(i)+'/'+str(total_cluster)+')')
    sys.stdout.flush()
sys.stdout.write('\r' + '>>> Analyzing ... '+'[>>>>>>>>>>]'+'('+str(total_cluster)+'/'+str(total_cluster)+')')
sys.stdout.flush()
time.sleep(0.3)
print('\rFound ' +str(counter)+ ' clusters.                    ')

#plt.plot([200,300,400],[100,200,300],'c', linewidth=5)
#ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
sys.stdout.write('Saving data ... ')
sys.stdout.flush()
start = time.time()

try: 
    os.makedirs('output')
except OSError:
    if not os.path.isdir('output'):
        raise

outfname = os.path.basename(args.f).split('.')[0]
if args.output:
    outfname = str(eval(args.output))


with open('output/'+outfname+'.csv','w') as output:
    for l in log:
        output.write(','.join([str(x) for x in l])+'\n')
cv2.imwrite('output/'+outfname+'_labeled.jpg',img)
#plt.show()
#plt.tight_layout()
#fig.savefig(os.path.basename(args.f).split('.')[0]+".svg")
#plt.close()

end = time.time()
sys.stdout.write('Done. ('+str(int(end-start))+'s)\n')
end_G = time.time()
sys.stdout.write('Total time used '+str(int(end_G-start_G))+'s.\n')
sys.stdout.flush()
print('Labeled image saved to: ' + 'output/'+outfname+'_labeled.jpg')
print('Counting results saved to: ' + 'output/'+outfname+'.csv')



