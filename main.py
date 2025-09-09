import cv2
import numpy as np

def detect_traffic_lights(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red_lower1 = np.array([0, 70, 70])
    red_upper1 = np.array([15, 255, 255])
    red_lower2 = np.array([160, 70, 70])
    red_upper2 = np.array([180, 255, 255])
    yellow_lower = np.array([15, 150, 150])
    yellow_upper = np.array([35, 255, 255])
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([90, 255, 255])

    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_green = cv2.inRange(hsv, green_lower, green_upper)

    def find_and_draw(mask, label, color):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                if 0.8 < aspect_ratio < 1.2:
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    find_and_draw(mask_red, "RED", (0, 0, 255))
    find_and_draw(mask_yellow, "YELLOW", (0, 255, 255))
    find_and_draw(mask_green, "GREEN", (0, 255, 0))
    return image, mask_red, mask_yellow, mask_green, cv2.split(hsv)

def overlay_masks_and_hsv(frame, mask_r, mask_y, mask_g, hsv_channels):
    thumb_height = 60
    pad = 5

    def resize_and_label(img, label, box_color):
        thumb = cv2.resize(img, (int(img.shape[1]*thumb_height/img.shape[0]), thumb_height))
        cv2.rectangle(thumb, (0,0), (thumb.shape[1]-1, thumb.shape[0]-1), box_color, 2)
        cv2.putText(thumb, label, (5, thumb.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, box_color, 1, cv2.LINE_AA)
        return thumb

    r_thumb = resize_and_label(cv2.cvtColor(mask_r, cv2.COLOR_GRAY2BGR), "RED", (0,0,255))
    y_thumb = resize_and_label(cv2.cvtColor(mask_y, cv2.COLOR_GRAY2BGR), "YELLOW", (0,255,255))
    g_thumb = resize_and_label(cv2.cvtColor(mask_g, cv2.COLOR_GRAY2BGR), "GREEN", (0,255,0))

    h, s, v = hsv_channels
    h_col = cv2.applyColorMap(cv2.convertScaleAbs(h, alpha=2), cv2.COLORMAP_HSV)
    s_col = cv2.merge([s,s,s])
    v_col = cv2.merge([v,v,v])
    h_thumb = resize_and_label(h_col, "HUE", (200,0,200))
    s_thumb = resize_and_label(s_col, "SAT", (200,200,0))
    v_thumb = resize_and_label(v_col, "VAL", (100,100,100))

    masks_row = np.hstack([r_thumb, y_thumb, g_thumb])
    hsv_row = np.hstack([h_thumb, s_thumb, v_thumb])
    dashboard = np.vstack([masks_row, hsv_row])

    dash_height, dash_width = dashboard.shape[:2]

    frame_disp = cv2.resize(frame, (dash_width, int(frame.shape[0] * dash_width // frame.shape[1])))

    combined = np.zeros((frame_disp.shape[0]+dash_height+pad*2, dash_width, 3), dtype=np.uint8)
    combined[:frame_disp.shape[0], :dash_width] = frame_disp
    combined[frame_disp.shape[0]+pad:frame_disp.shape[0]+pad+dash_height, :dash_width] = dashboard
    return combined

if __name__ == "__main__":
    video_path = "traffic.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    ret, frame = cap.read()
    if not ret:
        print("Error: No frames in video.")
        cap.release()
        exit()
    output_img, mask_r, mask_y, mask_g, hsv_channels = detect_traffic_lights(frame)
    output_combined = overlay_masks_and_hsv(output_img, mask_r, mask_y, mask_g, hsv_channels)
    out_height, out_width = output_combined.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 20  
    out = cv2.VideoWriter('traffic_output.avi', fourcc, fps, (out_width, out_height))
    out.write(output_combined)
    cv2.imshow('Traffic Light Detection • Masks • HSV Channels', output_combined)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_img, mask_r, mask_y, mask_g, hsv_channels = detect_traffic_lights(frame)
        output_combined = overlay_masks_and_hsv(output_img, mask_r, mask_y, mask_g, hsv_channels)
        cv2.imshow('Traffic Light Detection • Masks • HSV Channels', output_combined)
        out.write(output_combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
