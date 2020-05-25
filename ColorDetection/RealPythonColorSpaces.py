import matplotlib.pyplot as plt
import numpy
import cv2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score

picture = cv2.imread('several.png')
picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
# plt.imshow(picture)
# plt.show()
# r, g, b = cv2.split(picture)
# fig = plt.figure()
# axis = fig.add_subplot(1, 1, 1, projection="3d")
pixel_colors = picture.reshape((numpy.shape(picture)[0] * numpy.shape(picture)[1], 3))
norm = colors.Normalize(vmin=-1., vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
# axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
# axis.set_xlabel("Red")
# axis.set_ylabel("Green")
# axis.set_zlabel("Blue")
# plt.show()
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
d5 = numpy.transpose(numpy.vstack((h_n, s_n, v_n,x,y)))
# identify clusters
kmeans = KMeans(n_clusters=10, random_state=0, n_init=100).fit(d5)
# construct image of clusters
cluster_centers = kmeans.cluster_centers_
cluster_centers[:, 0] = (cluster_centers[:, 0] * 180.).astype(numpy.uint8)
cluster_centers[:, 1] = (cluster_centers[:, 1] * 256.).astype(numpy.uint8)
cluster_centers[:, 2] = (cluster_centers[:, 2] * 256.).astype(numpy.uint8)
h_new = kmeans.labels_.astype(numpy.uint8)
s_new = kmeans.labels_.astype(numpy.uint8)
v_new = kmeans.labels_.astype(numpy.uint8)
i = 0
swatch = list()
stats = list()
for center in cluster_centers:
    h_new[h_new == i] = center[0]
    s_new[s_new == i] = center[1]
    v_new[v_new == i] = center[2]
    i = i + 1
    s_h = (numpy.ones((5, 10)) * center[0]).astype(numpy.uint8)
    s_s = (numpy.ones((5, 10)) * center[1]).astype(numpy.uint8)
    s_v = (numpy.ones((5, 10)) * center[2]).astype(numpy.uint8)
    swatch.append(cv2.cvtColor(cv2.merge((s_h, s_s, s_v)), cv2.COLOR_HSV2RGB))
    #stats.append([center[3] * float(n_x), center[4] * float(n_y), len(h_new[h_new == i])])
h_new = h_new.reshape(n_y, n_x)
s_new = s_new.reshape(n_y, n_x)
v_new = v_new.reshape(n_y, n_x)
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
