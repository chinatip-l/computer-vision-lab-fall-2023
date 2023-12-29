import matplotlib.pyplot as plt
import cv2
import numpy as np
import re
import glob
FILE_PATH="output"
files=[f for f in glob.glob(f"{FILE_PATH}/*.png") if re.findall(r'his', f)==[]]

for f in files:
    
    n=re.split("\.|/|_|K|P",f)
    print(n)
    if len(n)==7:
        q= "Mean" if n[2]=='q1' else "Median" 
        k=int(n[4])
        p=int(n[5])
    else:
        q="Original"
    image = cv2.imread(f)
    print(type(image))
    print(image.dtype)
    print(image.shape)
    b=[0]*256
    for i in image:
        for j in i:
            b[j[0]]+=1

    
    plt.figure(figsize=(16, 5), dpi=100)
    plt.xlim([-1,256])
    if max(b)<5000:
        plt.ylim([0,5000])
    else:
        plt.ylim([0,22000])
    sn=re.split("\.|/",f)
    print(sn)
    if(q!="Original"):
        plt.title(f"{n[1]}_{n[2]}.png {q} Filter Kernel={k}x{k} Padding={p} Stride=1", fontsize=20)
    else:
        plt.title(f"{n[1]}_{n[2]}.png {q}", fontsize=20)
        if max(b)<8000:
            plt.ylim([0,7000])
        else:
            plt.ylim([0,max(b)])
    plt.bar(range(0,256),b,width=4, color='#121212')
    
    plt.savefig(f"{sn[0]}/{sn[1]}_his.{sn[2]}",bbox_inches='tight')

