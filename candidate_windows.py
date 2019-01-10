import cv2
import numpy as np
a = cv2.ximgproc.createEdgeBoxes()


def get_edge_boxes(image, size):
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection('data/edge_detection_model/model.yml')
    rgb_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv2.ximgproc.createEdgeBoxes(minBoxArea=25000)
    edge_boxes.setMaxBoxes(size)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)

    return boxes, edges


def extract_windows(image, boxes):
    windows = []
    for box in boxes:
        x, y, w, h = box
        windows.append(image[y:y + h, x:x + w])

    return windows


def visualize_boxes_and_edges(image, boxes, edges):

    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("edges", edges)
    cv2.imshow("edgeboxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#im = cv2.imread('data/test/images/6.JPEG')
#b, e = get_edge_boxes(im, 1)
#cv2.imshow("window", extract_windows(im, b)[0])
#visualize_boxes_and_edges(im, b, e)
