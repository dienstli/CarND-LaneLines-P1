# **Finding Lane Lines on the Road** 

[//]: # (Image References)

[image1]: ./examples/solid_white_right_result.png "Image result after pipeline"

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.
I've developed a simple pipeline consisting of six steps:
  - mask yellow colours to white, and this is how we can better detect yellow lane colour. I tried an RGB version from [link](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html)
  - transform the image to greyscale
  - apply Gaussian blur for noise reduction
  - detect edges using Canny edge detection
  - select the region of interest defined as a set of vertices
  - find lines using Hough's algorithm

find the pipeline code below ...
```python 
kernel_size = 5
canny_low_threshold = 50
canny_high_threshold = 150
rho = 1
theta = np.pi/180
# threshold, ow many points we need to agree it is a line, 
# here we get rid fo some noisy lines by making this number higher
threshold = 15
min_line_len = 5
max_line_gap = 150
alpha = 0.8
beta = 1.0
lamb = 0.0

# define the set of interest area vertices
vertices = np.array([[(0, image.shape[0]),
                      (450, 320), 
                      (490, 320), 
                      (image.shape[1], image.shape[0])]], 
                    dtype=np.int32)

# copy image
img = np.copy(image)
# yellow to white
img = mask_white_yellow_rgb(img)
# grayscale
img = grayscale(img)
# blur
img = gaussian_blur(img, kernel_size)
# canny before are of interest
img = canny(img, canny_low_threshold, canny_high_threshold)
# extract the area of interest
img = region_of_interest(img, vertices)
#hough lines
lines = hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap)
#merge with original image
weighted_img(lines, image, alpha, beta, lamb)
```
As part of the project, we also have to modify the `draw_lines` function. 
I've tried a couple of different approaches. 
In the end, I settle for one I found to be the most elegant (not the fastest). <br>
I took the idea from [link](https://peteris.rocks/blog/extrapolate-lines-with-numpy-polyfit/). 
You will find one of my previous approaches (faster!) as commented code in the notebook.<br>
The algorithm is as the one in the functions notes. 
Then we constructed the lines using `polyfit` and `poly1d` numpy function and extrapolated them in the picture.
```python
# https://peteris.rocks/blog/extrapolate-lines-with-numpy-polyfit/
def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    x_right = []
    y_right = []
    x_left = []
    y_left = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            if y2 == y1 and x2 == x1:
                continue
        
            slope = (y2 - y1)/(x2 - x1)
            
            # hoping to get rid of noisy lines, not sure this is the right approach
            if (slope > -0.4 and slope < 0.4) or slope < -0.8 or slope > 0.8:
                continue
            
            if slope < 0:
                x_right += [x1, x2]
                y_right += [y1, y2]
            else:
                x_left += [x1, x2]
                y_left += [y1, y2]
                
    # commented code also works! I choose this other approach to get better quality in the videos
    if len(x_left) != 0 and len(y_left) != 0:
        # Calculating average slope and b from all the points on left - here slope will be negative
        z = np.polyfit(x_left, y_left, 1)
        f = np.poly1d(z)
        # we could use min and max to make shorter lines
        min_x = int(min(x_left))
        max_x = int(max(x_left))
        cv2.line(img, (min_x, int(f(min_x))), (max_x, int(f(max_x))), color, thickness)
        

    if len(x_right) != 0 and len(y_right) != 0:
        # Calculating average slope and b from all the points on right - here slope will be positive
        z = np.polyfit(x_right, y_right, 1)
        f = np.poly1d(z)
        # we could use min and max to make shorter lines
        min_x = int(min(x_right))
        max_x = int(max(x_right))
        cv2.line(img, (min_x, int(f(min_x))), (max_x, int(f(max_x))), color, thickness)
```

Find next and image after applying the pipeline:<br> 
![image result after the pipeline][image1]


### 2. Identify potential shortcomings with your current pipeline



One of the problems is the narrow curves. We can see in the challenger video that we produce more noisy lanes when 
the right turn happens. One possible solution would be to reduce the area of interest in the image and change 
some of the Hough algorithm parameters (e.g. threshold).<br> 
Our dataset does not contain images on the middle lane or very close by objects that end up in our interest area, 
lousy weather images, or night shots. I would say that so far, we are overfitting to the current environment.<br>
I find the pipeline slow, but there is room for improvements in the code.

### 3. Suggest possible improvements to your pipeline

  - One of the improvements, ``draw_lines`` should not build lines but a polynomial curve. 
It introduces getting a new (x,y) coordinate in the equation only. <br>
  - Another improvement should be the normalization of the image to tackle any colour. 
What happens when the camera gets broken, and the image is not captured in accurate colours?<br>