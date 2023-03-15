import numpy as np
import matplotlib.pyplot as plt
from sympy.matrices import Matrix
import cv2 as cv
import imutils
import pprint
import array
#from sympy import symbols, Eq, solve
from sympy.solvers import solve
from sympy import Symbol

def eqParabola(coefficients, x_pts):
  print("The equation of parabola is :")
  print('y = %.5f * x^2 %.5f * x + %.5f' % (coefficients[0], coefficients[1], coefficients[2]))

  y = (coefficients[0]*x_pts*x_pts) + (coefficients[1]*x_pts) + coefficients[2]
  plt.title("Ball Trajectory")
  plt.xlabel("x co-ordinate of the center of the ball (pixels)")
  plt.ylabel("y co-ordinate of the center of the ball (pixels)")
  plt.plot(x_pts,y, "-r", label="Best fit curve")
  plt.plot(X, Y, "-b", label="Plotting with points identified")
  plt.legend(loc="upper right")
  print("The final landing y-coordinate is :")
  y_initial = (coefficients[0]*x_pts[0]*x_pts[0]) + (coefficients[1]*x_pts[0]) + coefficients[2]
  y_final = y_initial + 300
  print(y_final)
  x = Symbol('x')
  print("The landing x-cooridnate of the ball is :")
  solution = solve((coefficients[0]*x**2) + (coefficients[1]*x) + coefficients[2] - y_final)
  for val in solution:
    if val[x] > 0:
      print(val)


def ballTraj(X_pts, Y_pts):
  X_pts = np.array(X_pts)
  Y_pts = np.array(Y_pts)
  A_t=np.vstack((X_pts*X_pts,X_pts,np.ones(X_pts.shape)))
  #print(A_t)
  y=Y_pts
  y=y[np.newaxis]
  y_t = y.T
  A_t_B = np.matmul(A_t,y_t)
  A_t_A = np.matmul(A_t,A_t.transpose())
  coefficients = np.matmul(np.linalg.inv(A_t_A),A_t_B)
  return coefficients


X=[]
Y=[]
cap = cv.VideoCapture('/home/vignesh/Desktop/SEM2/Perception/Project1/ball.mov')
#cap = cv.VideoCapture('ball.mov')
pts=()
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("The file cannot be opened")

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
  # resize the frame and convert it to the HSV
    #frame = cv.resize(frame, (frame.shape[1]//2, frame.shape[0]//2), interpolation=cv.INTER_AREA)
    hsv=cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
   # Creating Mask to distinguish only the ball (in HSV)

    ## old values
    lower_red=(0,170,133)
    upper_red=(20,255,255)
    mask = cv.inRange(hsv, lower_red, upper_red)
    img=mask.copy()

    ### Finding Contours
    # find contours in the mask and initialize the current
	  # (x, y) center of the ball
    pts = np.where(img != 0)
    x_pts = pts[1]
    y_pts = pts[0]
    print("x points are :")
    print(x_pts)
    print("y points are :")
    print(y_pts)

    try:
      if abs(max(x_pts)-min(x_pts)) < 20:
        x_center = (max(x_pts)+min(x_pts))//2
      if abs(max(y_pts)-min(y_pts)) < 20:
        y_center = (max(y_pts)+min(y_pts))//2
      plt.scatter(x_center, y_center)

      if x_center != None:
        X.append(x_center)
        Y.append(y_center)
      
      #for pt in len(x_pts):
      cv.circle(frame, (x_pts,y_pts), 5, (0, 0, 255), 2)
      
    except:
      print("Value not found")
    
    if len(X) > 0:
      for pt in range(len(X)):
        #cv.circle(img, (X[pt],Y[pt]), 5, (255, 255, 255), -1)
        cv.circle(frame, (X[pt],Y[pt]), 2, (0, 255, 0), -1)

    #Displaying the image frame by frame with detection of ball center
    cv.imshow("Image", frame)

    # Press Q on keyboard to  exit
    if cv.waitKey(20) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything is done, release the video capture object
cap.release()
 
# Closes all the frames
cv.destroyAllWindows()

coefficients = ballTraj(X,Y)

eqParabola(coefficients, X)

plt.show()
