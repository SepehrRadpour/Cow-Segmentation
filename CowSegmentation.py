import cv2
import torch


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_cows(model, video_path, output_path):

    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = video.read()

        if not ret:
            break

        results = model(frame)

        cows = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'] == 'cow']

        for _, cow in cows.iterrows():
            xmin, ymin, xmax, ymax = cow[['xmin', 'ymin', 'xmax', 'ymax']]
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

        out.write(frame)

         cv2.imshow("Object Detection", frame)

    video.release()
    out.release()

    cv2.destroyAllWindows()

    print(f"Detection saved to {output_path}")

detect_cows(model, 'input_video.mp4', 'output_video.mp4')
