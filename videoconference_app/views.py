from django.shortcuts import render, redirect
from .forms import RegisterForm
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse, JsonResponse
'''import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image'''

# Create your views here.

def index(request):
    return render(request, 'index.html')


def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            return render(request, 'login.html', {'success': "Registration successful. Please login."})
        else:
            error_message = form.errors.as_text()
            return render(request, 'register.html', {'error': error_message})

    return render(request, 'register.html')


def login_view(request):
    if request.method=="POST":
        email = request.POST.get('email')
        password = request.POST.get('password')
        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)
            return redirect("/dashboard")
        else:
            return render(request, 'login.html', {'error': "Invalid credentials. Please try again."})

    return render(request, 'login.html')

@login_required
def dashboard(request):
    return render(request, 'dashboard.html', {'name': request.user.first_name})

@login_required
def videocall(request):
    return render(request, 'videocall.html', {'name': request.user.first_name + " " + request.user.last_name})

@login_required
def logout_view(request):
    logout(request)
    return redirect("/login")

@login_required
def join_room(request):
    if request.method == 'POST':
        roomID = request.POST['roomID']
        return redirect("/meeting?roomID=" + roomID)
    return render(request, 'joinroom.html')

'''@login_required
def gesture_view(request):
    return render(request, 'meeting/gesture.html')

# Initialize gemini
genai.configure(api_key="AIzaSyDIEJpltQNTGhO2ChKPqTs5VEtaI0abFJQ")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the HandDetector class with the given parameters
detector = HandDetector(detectionCon=0.75, maxHands=1)

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Error: Could not open webcam.")

def initialize_canvas(frame):
    return np.zeros_like(frame)

def process_hand(hand):
    lmList = hand["lmList"]  # List of 21 landmarks for the hand
    bbox = hand["bbox"]  # Bounding box around the hand (x,y,w,h coordinates)
    center = hand['center']  # Center coordinates of the hand
    handType = hand["type"]  # Type of the hand ("Left" or "Right")
    fingers = detector.fingersUp(hand)  # Count the number of fingers up
    return lmList, bbox, center, handType, fingers

def weighted_average(current, previous, alpha=0.5):
    return alpha * current + (1 - alpha) * previous

response_text = None

def send_to_ai(model, canvas, fingers):
    global response_text
    if fingers[4] == 1:
        image = Image.fromarray(canvas)
        response = model.generate_content(["solve this math problem", image])
        response_text = response.text if response else None

# Initialize variables
prev_pos = None
drawing = False
points = []  # Store points for drawing
smooth_points = None  # Smoothed position

# Initialize canvas
_, frame = cap.read()
canvas = initialize_canvas(frame)

def video_stream():
    global prev_pos, drawing, points, smooth_points, canvas

    # Initialize the webcam stream
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Error: Could not open webcam.")

    while True:
        # Capture each frame from the webcam
        success, img = cap.read()

        if not success:
            print("Failed to capture image")
            break

        # Flip the image horizontally for a later selfie-view display
        img = cv2.flip(img, 1)

        hands, img = detector.findHands(img, draw=True, flipType=True)

        if hands:
            hand = hands[0]
            lmList, bbox, center, handType, fingers = process_hand(hand)

            # Get the positions of the index and middle finger tips
            index_tip = lmList[8]
            thumb_tip = lmList[4]

            # Determine drawing state based on fingers up
            if fingers[1] == 1 and fingers[2] == 0:  # Only index finger is up
                current_pos = np.array([index_tip[0], index_tip[1]])
                if smooth_points is None:
                    smooth_points = current_pos
                else:
                    smooth_points = weighted_average(current_pos, smooth_points)
                smoothed_pos = tuple(smooth_points.astype(int))

                if drawing:  # Only add to points if already drawing
                    points.append(smoothed_pos)
                prev_pos = smoothed_pos
                drawing = True
            elif fingers[1] == 1 and fingers[2] == 1:  # Both index and middle fingers are up
                drawing = False
                prev_pos = None
                points = []  # Clear points to avoid connection
                smooth_points = None
            elif fingers[0] == 1:  # Thumb is up
                canvas = initialize_canvas(img)
                points = []
                drawing = False
                prev_pos = None
                smooth_points = None
            elif fingers[4] == 1:
                send_to_ai(model, canvas, fingers)

        # Draw polyline on the canvas
        if len(points) > 1 and drawing:
            cv2.polylines(canvas, [np.array(points)], isClosed=False, color=(0, 0, 255), thickness=5)

        # Combine the image and canvas
        img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Draw polyline on the canvas
        if len(points) > 1 and drawing:
            cv2.polylines(canvas, [np.array(points)], isClosed=False, color=(0, 0, 255), thickness=5)

        # Combine the image and canvas
        img = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@login_required
def video_feed(request):
    return StreamingHttpResponse(video_stream(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_response(request):
    global response_text
    return JsonResponse({'response': response_text})'''


import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from django.views.decorators.csrf import csrf_exempt

# Initialize MediaPipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize deque for each color
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# Color indices and setup
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # Blue, Green, Red, Yellow
colorIndex = 0

# Paint window setup
paint_window = np.zeros((471, 636, 3)) + 255
paint_window = cv2.rectangle(paint_window, (40, 1), (140, 65), (0, 0, 0), 2)  # Clear
paint_window = cv2.rectangle(paint_window, (160, 1), (255, 65), (255, 0, 0), 2)  # Blue
paint_window = cv2.rectangle(paint_window, (275, 1), (370, 65), (0, 255, 0), 2)  # Green
paint_window = cv2.rectangle(paint_window, (390, 1), (485, 65), (0, 0, 255), 2)  # Red
paint_window = cv2.rectangle(paint_window, (505, 1), (600, 65), (0, 255, 255), 2)  # Yellow


paint_window=cv2.putText(paint_window, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
paint_window=cv2.putText(paint_window, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
paint_window=cv2.putText(paint_window, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
paint_window=cv2.putText(paint_window, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
paint_window=cv2.putText(paint_window, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
paint_window = cv2.putText(paint_window, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
paint_window = cv2.putText(paint_window, "CLOSE", (520, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
paint_window = cv2.putText(paint_window, "Press Q to close", (40,450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

@login_required
@csrf_exempt
def start_drawing(request):
    global bpoints, gpoints, rpoints, ypoints
    global blue_index, green_index, red_index, yellow_index
    global paint_window, colorIndex

    if 'colorIndex' not in globals():
        colorIndex = 0  # Initialize colorIndex if not set

    if request.method == 'POST':
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                return JsonResponse({'error': 'Failed to capture frame from camera'}, status=500)

            # Flip the frame for a mirror effect
            frame = cv2.flip(frame, 1)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Drawing the buttons on the frame
            frame = cv2.rectangle(frame, (40,1), (140,65), (0,0,0), 2)
            frame = cv2.rectangle(frame, (160,1), (255,65), (255,0,0), 2)
            frame = cv2.rectangle(frame, (275,1), (370,65), (0,255,0), 2)
            frame = cv2.rectangle(frame, (390,1), (485,65), (0,0,255), 2)
            frame = cv2.rectangle(frame, (505,1), (600,65), (0,255,255), 2)

            # Drawing the CLOSE button on the frame
            frame = cv2.rectangle(frame, (500,400), (620,465), (0,0,0), 2)

            cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "CLOSE", (520, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

            # Hand detection
            result = hands.process(framergb)

            # Process hand landmarks
            if result.multi_hand_landmarks:
                landmarks = []
                for hand_lms in result.multi_hand_landmarks:
                    for lm in hand_lms.landmark:
                        lmx = int(lm.x * 640)
                        lmy = int(lm.y * 480)
                        landmarks.append([lmx, lmy])

                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                fore_finger = (landmarks[8][0], landmarks[8][1])
                center = fore_finger
                thumb = (landmarks[4][0], landmarks[4][1])
                cv2.circle(frame, center, 3, (0,255,0),-1)

                # Check for "CLEAR" or "CLOSE" button presses
                

                if 505 <= fore_finger[0] <= 600 and 400 <= center[1] <= 465:
                    # Close button
                    break

                if (thumb[1] - center[1] < 30):
                    bpoints.append(deque(maxlen=512))
                    blue_index += 1
                    gpoints.append(deque(maxlen=512))
                    green_index += 1
                    rpoints.append(deque(maxlen=512))
                    red_index += 1
                    ypoints.append(deque(maxlen=512))
                    yellow_index += 1

                elif center[1] <= 65:
                    if 40 <= center[0] <= 140: # Clear Button
                        bpoints = [deque(maxlen=512)]
                        gpoints = [deque(maxlen=512)]
                        rpoints = [deque(maxlen=512)]
                        ypoints = [deque(maxlen=512)]

                        blue_index = 0
                        green_index = 0
                        red_index = 0
                        yellow_index = 0
                        paint_window[67:,:,:] = 255
                    elif 160 <= center[0] <= 255:
                        colorIndex = 0 # Blue
                    elif 275 <= center[0] <= 370:
                            colorIndex = 1 # Green
                    elif 390 <= center[0] <= 485:
                            colorIndex = 2 # Red
                    elif 505 <= center[0] <= 600:
                            colorIndex = 3 # Yellow
                else :
                    if colorIndex == 0:
                        bpoints[blue_index].appendleft(center)
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft(center)
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft(center)
                    elif colorIndex == 3:
                        ypoints[yellow_index].appendleft(center)

            # Append the next deques when nothing is detected to avoid messing up
            else:
                bpoints.append(deque(maxlen=512))
                blue_index += 1
                gpoints.append(deque(maxlen=512))
                green_index += 1
                rpoints.append(deque(maxlen=512))
                red_index += 1
                ypoints.append(deque(maxlen=512))
                yellow_index += 1

            # Draw lines of all the colors on the canvas and frame
            points = [bpoints, gpoints, rpoints, ypoints]
            for i in range(len(points)):
                for j in range(len(points[i])):
                    for k in range(1, len(points[i][j])):
                        if points[i][j][k - 1] is None or points[i][j][k] is None:
                            continue
                        cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                        cv2.line(paint_window, points[i][j][k - 1], points[i][j][k], colors[i], 2)

            # Display the windows
            cv2.imshow("Paint", paint_window)
            cv2.imshow("Output", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        return JsonResponse({'message': 'Drawing session completed successfully!'})

    return JsonResponse({'error': 'Invalid request'}, status=400)

def get_response(request):
    global response_text
    return JsonResponse({'response': response_text})
def video_feed(request):
    return StreamingHttpResponse(start_drawing(), content_type='multipart/x-mixed-replace; boundary=frame')
