import cv2
from create_distance_field import *
import numpy as np

im = cv2.imread("../imgs/magic_roundabout/magic_roundabout_pre.png")
blur = create_distance_field(im, 2)

print(blur.shape)

# Create a copy of the blurred image scaled to 0-255 and 3 channels for drawing
overlay = (blur * 255).astype(np.uint8)
if len(overlay.shape) == 2:
    overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

# Grid size
grid_size = 1000 // 100
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.3
thickness = 1
text_color = (0, 0, 255)

# for i in range(100):
#     for j in range(100):
#         x = j * grid_size
#         y = i * grid_size
#         coord_x = j - 50
#         coord_y = i - 50
#         text = f"({coord_x},{coord_y})"
#         cv2.putText(overlay, text, (x + 2, y + grid_size - 2), font, font_scale, text_color, thickness)

# Draw grid lines
for i in range(0, 1001, grid_size):
    color = (200, 200, 200) if (i % 50) else (0, 0, 200)
    if i == 500:
        color = (200, 0, 0)
    cv2.line(overlay, (i, 0), (i, 1000), color, 1)
    cv2.line(overlay, (0, i), (1000, i), color, 1)

cv2.imwrite('../imgs/magic_roundabout/magic_roundabout.png', (255 * blur).astype(np.uint8))
cv2.imwrite('../imgs/magic_roundabout/magic_roundabout_grid.png', overlay)
# cv2.imshow("Grid Overlay", overlay)
# cv2.waitKey(0)
# cv2.destroyAllWindows()