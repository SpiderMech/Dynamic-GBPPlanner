import numpy as np
import cv2
from create_distance_field import *

OBSTACLE_COLOR = (0,0,0)
img = 255*np.ones((1000, 1000, 3), dtype=np.uint8)

cv2.rectangle(img, (475, 460), (525, 510), OBSTACLE_COLOR, -1)

blur = create_distance_field(img)

# cv2.imshow("Result", blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('playground.png', 255 * blur)