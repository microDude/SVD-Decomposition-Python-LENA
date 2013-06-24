'''
Created on Feb 14, 2013
@author: gutshall
Main Module to test Unitary Matices with Lena
'''

# System Library imports
from numpy import float32,around,diag
from numpy.linalg import svd
from matplotlib.pyplot import *
import scipy.misc as misc

# Load the Lena Image
image = misc.lena().astype(float32)

# Display the original
fig1 = figure(1)
ax = fig1.add_subplot(1,1,1)
ax.imshow(image,cmap = 'bone')
fig1.show()

# Find the highest 'n' eigenvales of the image
[U,s,V] = svd(image) # Note, svd() returns already V = V.conj().T
print('U = ',U.shape)
print('s = ',s.shape)
print('V = ',V.shape)

# Plot the eigenvalues of SVD
fig2 = figure(2)
ax = fig2.add_subplot(1,1,1)
ax.loglog(s)
ax.grid(True)
ax.set_title('Eigenvalues of Lena')
ax.set_xlabel('index [k]')
ax.set_ylabel('amplitude')
fig2.show()

# Debug, reconstruct orginal image
#===================================================================================================
# sr = diag(s)
# print('size of s = ',sr.shape)
# B = U.dot(sr).dot(V) # Note, svd() returns already V = V.conj().T.  So, no need to take conjagate transpose of V
# B = around(B)
# print('size of B = ',B.shape)
#===================================================================================================

# Reconstruct the image with a subset of the orthonormal basis
numEigs = 20
Ur = U[:,0:numEigs]
Vr = V[0:numEigs,:]
sr = diag(s[0:numEigs])
print('Ur = ', Ur.shape)
print('sr = ', sr.shape)
print('Vr = ', Vr.shape)

B = Ur.dot(sr).dot(Vr) # Note, svd() returns already V = V.conj().T.  So, no need to take conjagate transpose of V
B = around(B)

fig4 = figure(4)
ax = fig4.add_subplot(1,1,1)
ax.imshow(B,cmap = 'bone')
[m,n] = image.shape
compressionRatio = around(( ((numEigs*(m + n) + numEigs) / (m*n)) * 100),decimals=1)
title('Original = ' + str(m) + '*' + str(n) + '=' + str(m*n) + ' bytes \n' +
          'Compressed = ' + str(numEigs) + '*' + str(m) + '+' + str(numEigs) + '*' + str(n) + '+' + str(numEigs) +
          '=' + str(numEigs*m + numEigs*n + numEigs) + ' bytes \n' +
          'Compression Ratio = ' +  str(compressionRatio) + ' %')
show()
