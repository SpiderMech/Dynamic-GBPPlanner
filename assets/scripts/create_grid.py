import cv2
import numpy as np

im = cv2.imread("../imgs/junction_twoway/junction_twoway.png")
print(im.shape)

grid_size = 1000 // 100
for i in range(0, 1001, grid_size):
    color = (200, 200, 200) if (i % 50) else (0, 0, 200)
    if i == 500:
        color = (200, 0, 0)
    cv2.line(im, (i, 0), (i, 1000), color, 1)
    cv2.line(im, (0, i), (1000, i), color, 1)

# cv2.imshow("grid img", im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite("../imgs/junction_twoway/junction_twoway_grid.png", im)
