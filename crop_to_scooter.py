import cv2
from utils.datasets import LoadImages


vid_path = "/home/sibi/Downloads/cycle_videos/rec_3.mp4"
data = LoadImages(path=vid_path)

vid_name = vid_path.split('/')[-1]
save_path = f"/runs/detect/exp100/{vid_name}"
print(save_path)

video_writer = None

for path, img, img0, vid_cap, s in data:
    cv2.imshow("video", img0)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
    if not video_writer:
        print("creating vid writer")
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        h, w, _ = img0.shape
        # print(fps, w, h)
        
        video_writer = cv2.VideoWriter(
            "output.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (int(w), int(h)), True
        )
        
    video_writer.write(img0)

# cleanup
video_writer.release()
cv2.destroyAllWindows()
    