import numpy as np

TWO_CLOUDS_2D = np.array([[3, -1],
                         [4, 0],
                         [5, -2],
                         [4, 6],
                         [5, 7],
                         [5.5, 6]])
TWO_CLOUDS_2D_LABELS = np.array([0, 0, 0, 1, 1, 1])

THREE_CLOUDS_3D = np.array([[3, -1, -1],
                            [4, 0, 0],
                            [5, -2, 1],
                            [4, 6, -1],
                            [5, 7, 0],
                            [5.5, 6, 1],
                            [10, 2, -1],
                            [11, 4, 0],
                            [12, 3, 1]])
THREE_CLOUDS_3D_LABELS = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])