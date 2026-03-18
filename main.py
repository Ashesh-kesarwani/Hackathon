import cv2
import numpy as np
import os
import time
import sys
import serial

def load_yolo():
    files = ["yolov3-tiny.weights", "yolov3-tiny.cfg", "coco.names"]
    for f in files:
        if not os.path.exists(f):
            print(f"Error: Missing {f}")
            sys.exit(1)

    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    
    try:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except Exception:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
            
    return net, classes, output_layers

class CameraState:
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.mode = "COUNT"
        self.prev_centers = []
        self.crossing_times = []
        
        self.display_img = None
        self.count = 0
        self.flow_rate = 0.0
        self.emergency = False
        self.req_time = 0

class TrafficController:
    def __init__(self, port="COM9", baud=9600):
        self.active_cam = 1
        self.state = "GREEN"
        self.timer_end = time.time()
        self.last_sent_cmds = {}
        
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)
        except Exception as e:
            self.ser = None
            print(f"Warning: Could not connect to Arduino on {port}: {e}")
            
        # Initial state payload
        self.send_cmd("C1_GREEN")
        self.send_cmd("C2_RED")

    def send_cmd(self, cmd):
        prefix = cmd[:2]
        if self.ser and self.ser.is_open:
            if self.last_sent_cmds.get(prefix) != cmd:
                try:
                    self.ser.write((cmd + '\n').encode())
                    self.last_sent_cmds[prefix] = cmd
                except Exception:
                    pass

    def update(self, s1: CameraState, s2: CameraState):
        current_time = time.time()
        
        if current_time >= self.timer_end:
            if self.state == "GREEN":
                self.state = "YELLOW"
                self.timer_end = current_time + 3
                if self.active_cam == 1:
                    self.send_cmd("C1_YELLOW")
                    self.send_cmd("C2_RED")
                else:
                    self.send_cmd("C1_RED")
                    self.send_cmd("C2_YELLOW")
            
            elif self.state == "YELLOW":
                self.state = "GREEN"
                if self.active_cam == 1:
                    if s2.emergency and not s1.emergency:
                        self.active_cam = 2
                    elif s2.req_time > s1.req_time:
                        self.active_cam = 2
                else:
                    if s1.emergency and not s2.emergency:
                        self.active_cam = 1
                    elif s1.req_time > s2.req_time:
                        self.active_cam = 1
                
                allocated_time = s1.req_time if self.active_cam == 1 else s2.req_time
                if allocated_time == 0: allocated_time = 10
                self.timer_end = current_time + allocated_time
                
                if self.active_cam == 1:
                    self.send_cmd("C1_GREEN")
                    self.send_cmd("C2_RED")
                else:
                    self.send_cmd("C1_RED")
                    self.send_cmd("C2_GREEN")

        elif self.state == "GREEN":
            # Emergency override preemption
            if self.active_cam == 1 and s2.emergency and not s1.emergency:
                self.state = "YELLOW"
                self.timer_end = current_time + 3
                self.send_cmd("C1_YELLOW")
                self.send_cmd("C2_RED")
            elif self.active_cam == 2 and s1.emergency and not s2.emergency:
                self.state = "YELLOW"
                self.timer_end = current_time + 3
                self.send_cmd("C1_RED")
                self.send_cmd("C2_YELLOW")

        # Set status strings for display
        remaining = int(max(0, self.timer_end - current_time))
        if self.active_cam == 1:
            if self.state == "GREEN":
                s1_stat, s2_stat = "GREEN", "RED"
            else:
                s1_stat, s2_stat = "YELLOW", "RED"
        else:
            if self.state == "GREEN":
                s1_stat, s2_stat = "RED", "GREEN"
            else:
                s1_stat, s2_stat = "RED", "YELLOW"
                
        return s1_stat, remaining, s2_stat, remaining

    def close(self):
        if self.ser and self.ser.is_open:
            self.send_cmd("C1_RED")
            self.send_cmd("C2_RED")
            self.ser.close()

def get_required_time(mode, count, flow, emergency):
    if emergency:
        return 120
    if mode == "COUNT":
        if count < 15: return 30
        elif count <= 29: return 45
        else: return 60
    else:
        if flow < 0.5: return 30
        elif flow < 2.0: return 45
        else: return 60

def process_camera(frame, net, output_layers, classes, state: CameraState):
    height, width, _ = frame.shape
    line_y = height * 2 // 3

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                label = classes[class_id]
                if label in ["car", "motorbike", "bus", "truck"]:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    
    current_centers = []
    emergency_detected = False
    vehicle_count = 0

    if len(indices) > 0:
        flattened_indices = indices.flatten()
        vehicle_count = len(flattened_indices)
        for i in flattened_indices:
            box = boxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            label = classes[class_ids[i]]
            
            cx, cy = x + w // 2, y + h // 2
            current_centers.append((cx, cy))
            
            if label in ["bus", "truck"]:
                emergency_detected = True
                color = (0, 255, 0)
                text = f"EMERGENCY ({label})"
            else:
                color = (0, 0, 255)
                text = label
                
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if vehicle_count > 40:
        state.mode = "FLOW"
    elif vehicle_count < 20:
        state.mode = "COUNT"

    crossings = 0
    new_prev_centers = []
    matched_curr = set()
    
    for px, py in state.prev_centers:
        best_d = float('inf')
        best_c = None
        for i, (cx, cy) in enumerate(current_centers):
            if i in matched_curr: continue
            d = (px - cx)**2 + (py - cy)**2
            if d < 2500:
                best_d = d
                best_c = i
        if best_c is not None:
            cx, cy = current_centers[best_c]
            matched_curr.add(best_c)
            if (py < line_y and cy >= line_y) or (py >= line_y and cy < line_y):
                crossings += 1
            new_prev_centers.append((cx, cy))

    for i, (cx, cy) in enumerate(current_centers):
        if i not in matched_curr:
            new_prev_centers.append((cx, cy))

    state.prev_centers = new_prev_centers
    
    current_time = time.time()
    for _ in range(crossings):
        state.crossing_times.append(current_time)

    state.crossing_times = [t for t in state.crossing_times if current_time - t <= 5.0]
    flow_rate = len(state.crossing_times) / 5.0

    state.count = vehicle_count
    state.flow_rate = flow_rate
    state.emergency = emergency_detected
    state.req_time = get_required_time(state.mode, state.count, state.flow_rate, state.emergency)

    if state.mode == "FLOW":
        cv2.line(frame, (0, line_y), (width, line_y), (255, 255, 0), 2)
        
    state.display_img = frame

def draw_overlay(frame, state, sig_stat, timer):
    cv2.putText(frame, f"Cam {state.cam_id}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Mode: {state.mode}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if state.mode == "FLOW":
        cv2.putText(frame, f"Flow: {state.flow_rate:.1f}/s", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        cv2.putText(frame, f"Vehicles: {state.count}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    if sig_stat == "GREEN": color = (0, 255, 0)
    elif sig_stat == "YELLOW": color = (0, 255, 255)
    else: color = (0, 0, 255)
    
    cv2.putText(frame, f"Status: {sig_stat}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Timer: {timer}s", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    if state.emergency:
        cv2.putText(frame, "EMERGENCY OVERRIDE", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def main():
    net, classes, output_layers = load_yolo()

    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)

    state1 = CameraState(1)
    state2 = CameraState(2)
    
    controller = TrafficController("COM9", 9600)

    frame_count = 0
    last_print_time = time.time()

    last_sig1, last_sig2, last_timer = "WAIT", "WAIT", 0

    while True:
        ret1, frame1 = cap1.read() if cap1.isOpened() else (False, None)
        ret2, frame2 = cap2.read() if cap2.isOpened() else (False, None)

        if not ret1 and not ret2:
            print("No video feed from either camera.")
            break

        if frame_count % 2 == 0:
            if ret1: process_camera(frame1, net, output_layers, classes, state1)
            if ret2: process_camera(frame2, net, output_layers, classes, state2)
            
            # Update traffic light controller logic
            sig1, timer1, sig2, timer2 = controller.update(state1, state2)
            last_sig1, last_sig2, last_timer = sig1, sig2, timer1

            if ret1 and state1.display_img is not None:
                draw_overlay(state1.display_img, state1, sig1, timer1)
            if ret2 and state2.display_img is not None:
                draw_overlay(state2.display_img, state2, sig2, timer1)

        if state1.display_img is not None:
            cv2.imshow("Intersection 1", state1.display_img)
        if state2.display_img is not None:
            cv2.imshow("Intersection 2", state2.display_img)

        current_time = time.time()
        if current_time - last_print_time >= 1.5:
            log1 = ""
            log2 = ""
            if state1.display_img is not None:
                val = state1.count if state1.mode == "COUNT" else f"{state1.flow_rate:.1f}/sec"
                lbl = "Vehicles" if state1.mode == "COUNT" else "Flow"
                log1 = f"[Cam1] Mode: {state1.mode} | {lbl}: {val} | {last_sig1} ({last_timer}s)"
                    
            if state2.display_img is not None:
                val = state2.count if state2.mode == "COUNT" else f"{state2.flow_rate:.1f}/sec"
                lbl = "Vehicles" if state2.mode == "COUNT" else "Flow"
                log2 = f"[Cam2] Mode: {state2.mode} | {lbl}: {val} | {last_sig2} ({last_timer}s)"
                    
            if log1 and log2:
                print(f"{log1}   ||   {log2}")
            elif log1:
                print(log1)
            elif log2:
                print(log2)
                
            last_print_time = current_time

        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_count += 1

    if cap1.isOpened(): cap1.release()
    if cap2.isOpened(): cap2.release()
    controller.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
