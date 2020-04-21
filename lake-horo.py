#!/bin/python

#Import libraries for simulation
import tensorflow as tf
import numpy as np
import sys
import time
import horovod.tensorflow as hvd

hvd.init()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'

sess = tf.InteractiveSession() #CPU version
#sess = tf.InteractiveSession(config=config) #Use only for capability 3.0 GPU

#Imports for visualization
import PIL.Image

def DisplayArray(a, fmt='jpeg', rng=[0,1]):
  """Display an array as a picture."""
  rank = hvd.rank()
  jpg_file_name = "lake_py_" + str(rank) + ".jpg"
  dat_file_name = "lake_c_" + str(rank) +".dat"

  global N
  h = float(1)/N
  with open(dat_file_name,'w') as f:
      for i in range(len(a)):
        for j in range(len(a[i])):
          f.write(str(i*h)+" "+str(j*h)+" "+str(a[i][j])+'\n')

  a = (a - rng[0])/float(rng[1] - rng[0])*255
  a = np.uint8(np.clip(a, 0, 255))

  with open(jpg_file_name,"w") as f:
      PIL.Image.fromarray(a).save(f, "jpeg")

sess = tf.InteractiveSession()

# Computational Convenience Functions
def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

def simple_conv(x, k):
  """A simplified 2D convolution operation"""
  x = tf.expand_dims(tf.expand_dims(x, 0), -1)
  y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
  return y[0, :, :, 0]

def laplace(x):
  """Compute the 2D laplacian of an array"""
#  5 point stencil #
  five_point = [[0.0, 1.0, 0.0],
                [1.0, -4., 1.0],
                [0.0, 1.0, 0.0]]

#  9 point stencil #
  nine_point = [[0.25, 1.0, 0.25],
                [1.00, -5., 1.00],
                [0.25, 1.0, 0.25]]

  thirteeen_point = [[0.000,0.000,0.000,0.125,0.000,0.000,0.000],
                    [0.000,0.000,0.000,0.250,0.000,0.000,0.000],
                    [0.000,0.000,0.000,1.000,0.000,0.000,0.000],
                    [0.125,0.250,1.000,-5.50,1.000,0.250,0.125],
                    [0.000,0.000,0.000,1.000,0.000,0.000,0.000],
                    [0.000,0.000,0.000,0.250,0.000,0.000,0.000],
                     [0.000,0.000,0.000,0.125,0.000,0.000,0.000]]
   
  laplace_k = make_kernel(thirteeen_point)
  return simple_conv(x, laplace_k)

# Define the PDE
if len(sys.argv) != 4:
  print "Usage:", sys.argv[0], "N npebs num_iter"
  sys.exit()
  
N = int(sys.argv[1])
npebs = int(sys.argv[2])
num_iter = int(sys.argv[3])

# Initial Conditions -- some rain drops hit a pond

# Set everything to zero
u_init  = np.zeros([N/2, N], dtype=np.float32)
ut_init = np.zeros([N/2, N], dtype=np.float32)

# Some rain drops hit a pond at random points
for n in range(npebs):
  a = np.random.randint(0, N/2)
  b = np.random.randint(0, N)
  u_init[a,b] = np.random.uniform()

# Parameters:
# eps -- time resolution
# damping -- wave damping
eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())

# Create variables for simulation state
U  = tf.Variable(u_init)
Ut = tf.Variable(ut_init)

#The complete 3N/2 * N matrices on which we'll perform calculations
U_full = tf.Variable(np.zeros([(N/2) + 3, N], dtype=np.float32))          
Ut_full = tf.Variable(np.zeros([(N/2) + 3, N], dtype=np.float32))


rank_bcast = tf.group(
  tf.assign(U_full[N/2:], hvd.broadcast(U[:3], 1)),  #Sending first 3 rows of rank 1 to rank 0 for U
  tf.assign(Ut_full[N/2:], hvd.broadcast(Ut[:3], 1)),  #Sending first 3 rows of rank 1 to rank 0 for Ut
  tf.assign(U_full[:3], hvd.broadcast(U[-3:], 0)),  #Sending last 3 rows of rank 0 to rank 1 for U
  tf.assign(Ut_full[:3], hvd.broadcast(Ut[-3:], 0)))  #Sending 2nd half of rank 0 to rank 1 for Ut

#Copy the rest of U and Ut for rank 0
U_full_rank0_group = tf.group(                         
    U_full[:N/2].assign(U),
    Ut_full[:N/2].assign(Ut))

#Copy the rest of U and Ut for rank 1
U_full_rank1_group = tf.group(
      U_full[3:].assign(U),
      Ut_full[3:].assign(Ut))

# Discretized PDE update rules
U_ = U_full + eps * Ut_full
Ut_ = Ut_full + eps * (laplace(U_full) - damping * Ut_full)


# Operation to update the state
step = tf.group(
  U_full.assign(U_),
  Ut_full.assign(Ut_))


#Get the matrix U and Ut from U_full and Ut_full for rank 0
rank0_extract = tf.group(
  U.assign(tf.slice(U_full,[0,0],[N/2,N])),
  Ut.assign(tf.slice(Ut_full,[0,0],[N/2,N])))

#Get the matrix U and Ut from U_full and Ut_full for rank 1
rank1_extract = tf.group(
  U.assign(tf.slice(U_full,[3,0],[N/2,N])),
  Ut.assign(tf.slice(Ut_full,[3,0],[N/2,N])))

# Initialize state to initial conditions
tf.global_variables_initializer().run()

# Run num_iter steps of PDE
start = time.time()
for i in range(num_iter):
  # Step simulation
  rank_bcast.run()      #Send the upper and lower values for rank 1 and 0 respectively for matrix U and Ut

  if hvd.rank() == 0:      # Fill out the rest of the values for U_full and Ut_full
      U_full_rank0_group.run()    
  else:
      U_full_rank1_group.run()

  step.run({eps: 0.06, damping: 0.03})    

  if hvd.rank() == 0:           #Extract U and Ut from Ufull and Utfull
      rank0_extract.run()
  else:
      rank1_extract.run()

end = time.time()


print('Elapsed time: {} seconds'.format(end - start))

# from here we send our data to the other node

DisplayArray(U.eval(), rng=[-0.1, 0.1])
