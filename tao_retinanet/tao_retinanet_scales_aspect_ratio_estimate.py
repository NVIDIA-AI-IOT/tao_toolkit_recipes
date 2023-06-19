# Copyright (c) 2023 NVIDIA CORPORATION. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import numpy as np
from sklearn.cluster import KMeans

num_scales_retinanet=6
num_ars_retinanet=3
limit_max_ar=4

#file 004156.jpg
# 004156.jpg: JPEG image data, JFIF standard 1.01, aspect ratio, density 1x1, segment length 16, baseline, precision 8, 1242x375, frames 3
shorter_length_of_image = 375

folder="/home/user/tlt-experiments/data/training/label_2/"
widths=[]
heights=[]
files=[]

for r, d, f in os.walk(folder):
    for file in f:
        if file.endswith(".txt"):
            file1 = open(folder+file, 'r')
            lines=file1.readlines()
                
            for line in lines:
                line_split=line.split(" ")
                cls=line_split[0]
                xl=float(line_split[4])
                yl=float(line_split[5])
                xr=float(line_split[6])
                yr=float(line_split[7])
                
                width=xr-xl
                height=yr-yl
                
                if cls != 'DontCare' and width>=0 and height >= 0:
                    widths.append(width)
                    heights.append(height)
                    files.append(file1)
            file1.close()

scales=[]
aspect_ratios=[]
for i in range(len(widths)):
    w=widths[i]
    h=heights[i]
    if w<h:
        scale=w/shorter_length_of_image
    else:
        scale=h/shorter_length_of_image
        
    scales.append(scale)
    
    ar=h/w
    if ar<limit_max_ar:
        aspect_ratios.append(ar)

x=np.array(scales)
x=x.reshape(x.shape[0], 1)
kmeans = KMeans(n_clusters=num_scales_retinanet, random_state=0, n_init="auto").fit(x)
centers=kmeans.cluster_centers_
centers=np.squeeze(centers, axis=1)
print("scales: ", np.sort(centers))

x=np.array(aspect_ratios)
x=x.reshape(x.shape[0], 1)
kmeans = KMeans(n_clusters=num_ars_retinanet, random_state=0, n_init="auto").fit(x)
centers=kmeans.cluster_centers_
centers=np.squeeze(centers, axis=1)
print("aspect ratios: ", np.sort(centers))
