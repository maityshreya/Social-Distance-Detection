import cv2
import numpy as np

class VideoCapture:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
    
    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self):
        self.cap.release()

class YOLOv3Detector:
    def __init__(self, weights_path, config_path):
        self.net = cv2.dnn.readNet(weights_path, config_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect_objects(self, frame):
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        boxes, confidences, class_ids, centroids = [], [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if class_id == 0 and confidence > 0.5:  # Object is a person
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    centroids.append((center_x, center_y))

        return boxes, confidences, class_ids, centroids

class DistanceCalculator:
    @staticmethod
    def calculate_distance(point1, point2):
        return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

class FrameAnnotator:
    @staticmethod
    def draw_bounding_box(frame, box, color):
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
    @staticmethod
    def draw_text(frame, text, position, color=(0, 0, 255), font_scale=0.5, thickness=2):
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


class FrameDisplay:
    @staticmethod
    def show_frame(window_name, frame):
        cv2.imshow(window_name, frame)
    
    @staticmethod
    def wait_key(delay=1):
        return cv2.waitKey(delay)
    
    @staticmethod
    def destroy_all_windows():
        cv2.destroyAllWindows()

# Parameters
min_distance = 50

# Main execution
video_path = 'pedestrian.mp4'  # Change this to the path of your video file
video_capture = VideoCapture(video_path)
detector = YOLOv3Detector('yolov3.weights', 'yolov3.cfg')

while True:
    ret, frame = video_capture.read_frame()
    if not ret:
        break

    boxes, confidences, class_ids, centroids = detector.detect_objects(frame)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    violation_count = 0
    for i in range(len(centroids)):
        if i in indices:
            x1, y1 = centroids[i]
            for j in range(i + 1, len(centroids)):
                if j in indices:
                    x2, y2 = centroids[j]
                    distance = DistanceCalculator.calculate_distance((x1, y1), (x2, y2))
                    if distance < min_distance:
                        violation_count += 1
                        FrameAnnotator.draw_bounding_box(frame, boxes[i], (0, 0, 255))
                        FrameAnnotator.draw_bounding_box(frame, boxes[j], (0, 0, 255))
                    else:
                        FrameAnnotator.draw_bounding_box(frame, boxes[i], (0, 255, 0))
                        FrameAnnotator.draw_bounding_box(frame, boxes[j], (0, 255, 0))
                   

    # Display violation count
    FrameAnnotator.draw_text(frame, f'Violations: {violation_count}', (10, 20))


    FrameDisplay.show_frame('Image', frame)
    if FrameDisplay.wait_key(1) == ord('q'):
        break

video_capture.release()
FrameDisplay.destroy_all_windows()
