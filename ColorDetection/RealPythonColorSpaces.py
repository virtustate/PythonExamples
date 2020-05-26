import matplotlib.pyplot as plt
import numpy
import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score

picture = cv2.imread('several100.png')
picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
pixel_colors = picture.reshape((numpy.shape(picture)[0] * numpy.shape(picture)[1], 3))
norm = colors.Normalize(vmin=-1., vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
hsv_picture = cv2.cvtColor(picture, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_picture)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()

# normalize hsv,x,y values [0,1] and put in 5-d array for clustering
h_n = h.flatten() / 180.0
s_n = s.flatten() / 256.0
v_n = v.flatten() / 256.0
n_y = hsv_picture.shape[0]
n_x = hsv_picture.shape[1]
x_row = numpy.array(range(0, n_x)) / float(n_x - 1)
y_row = numpy.ones([n_x])
x = x_row
y = y_row * 0
for i in range(1, n_y):
    x = numpy.append(x, x_row)
    y = numpy.append(y, y_row * float(i / (n_y - 1)))
d5 = numpy.transpose(numpy.vstack((h_n, s_n, v_n)))
# identify clusters
kmeans = KMeans(n_clusters=8).fit(d5)
# construct image of clusters
cluster_centers = kmeans.cluster_centers_
cluster_centers[:, 0] = (cluster_centers[:, 0] * 180.).astype(numpy.uint8)
cluster_centers[:, 1] = (cluster_centers[:, 1] * 256.).astype(numpy.uint8)
cluster_centers[:, 2] = (cluster_centers[:, 2] * 256.).astype(numpy.uint8)
# nasty bug when an replaced label value equals a subsequent label in replace loop
labels = 1000 * (kmeans.labels_ + 1)
h_new = labels.copy()
s_new = labels.copy()
v_new = labels.copy()
i = 0
swatch = list()
stats = list()
for center in cluster_centers:
    label = 1000 * (i + 1)
    h_new[h_new == label] = center[0]
    s_new[s_new == label] = center[1]
    v_new[v_new == label] = center[2]
    s_h = (numpy.ones((5, 10)) * center[0]).astype(numpy.uint8)
    s_s = (numpy.ones((5, 10)) * center[1]).astype(numpy.uint8)
    s_v = (numpy.ones((5, 10)) * center[2]).astype(numpy.uint8)
    swatch.append(cv2.cvtColor(cv2.merge((s_h, s_s, s_v)), cv2.COLOR_HSV2RGB))
    i = i + 1
    #stats.append([center[3] * float(n_x), center[4] * float(n_y), len(h_new[h_new == i])])
h_new = h_new.reshape(n_y, n_x).astype(numpy.uint8)
s_new = s_new.reshape(n_y, n_x).astype(numpy.uint8)
v_new = v_new.reshape(n_y, n_x).astype(numpy.uint8)
# for a_stat in stats:
#     x = int(a_stat[0])
#     y = int(a_stat[1])
#     h_new[y, x] = 0
#     s_new[y, x] = 0
#     v_new[y, x] = 0
new = cv2.cvtColor(cv2.merge((h_new, s_new, v_new)), cv2.COLOR_HSV2RGB)
plt.imshow(numpy.vstack(swatch))
plt.show()
plt.imshow(new)
plt.show()
print(f'Davies Bouldin: {davies_bouldin_score(d5, kmeans.labels_)}')
print(f'Silhouette: {silhouette_score(d5, kmeans.labels_, metric="euclidean")}')

print(f'Colinski Harabasz: {calinski_harabasz_score(d5, kmeans.labels_)}')
# TODO: calculate the RMS color difference between original and reconstructed image
