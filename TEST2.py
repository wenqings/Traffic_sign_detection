import numpy as np
classes = np.array([[1,1]])
s = np.array([[0.99,0.4]])
boxes = np.array([[[1,2,3,4],[5,6,7,8]]])

new_boxes = np.empty((0,4),int)
new_scores = np.empty((0),float)
new_classes = np.array([])

for i in range(2):
    if np.squeeze(s)[i] >= 0.6:
        if np.squeeze(classes).astype(np.int32)[i] in [1, 2, 3, 4, 5]:
            if boxes[0][i][2] - boxes[0][i][0] >= 0.01 and boxes[0][i][3] - boxes[0][i][1] >= 0.01:
                a = [[boxes[0][i][0], boxes[0][i][1], boxes[0][i][2], boxes[0][i][3]]]
                print(a)
                new_boxes = np.append(new_boxes, a, axis=0)

                # new_scores = np.append(new_scores, [s[0][i]],axis=0)
                new_scores = np.append(new_scores, s[0][i])
                # np.concatenate(new_classes, classes[0][i])
                # print(classes[0][i])
                print(new_boxes.shape)
                print(new_scores.shape)
                print('new_scores:',new_scores)