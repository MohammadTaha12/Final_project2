import cv2
import time
from time import sleep
from ultralytics import YOLO
from gpiozero import LED, Button, MotionSensor
import threading
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub
import queue
import requests
import json
import re
from datetime import datetime
import io
import sys
from concurrent.futures import ProcessPoolExecutor

# --- Firebase Configuration ---
FIREBASE_URL = "https://smart-traffic-sys-default-rtdb.europe-west1.firebasedatabase.app/"
firebase_logs = []
firebase_logs_lock = threading.Lock()
firebase_upload_interval = 60  # Upload logs every 60 seconds

# Custom print function to capture logs for Firebase
original_print = print
def firebase_print(*args, **kwargs):
    # Call the original print function
    original_print(*args, **kwargs)
    
    # Convert args to string
    output = io.StringIO()
    original_print(*args, file=output, **kwargs)
    log_text = output.getvalue().strip()
    
    # Skip detection output lines and vehicle count lines
    detection_pattern = r'^\d+: \d+x\d+ .*(ms)$'
    vehicle_count_pattern = r'^🚘 Street \d+: \d+ vehicles, wait: \d+ seconds$'
    
    if re.match(detection_pattern, log_text) or re.match(vehicle_count_pattern, log_text):
        return
    
    # Add to Firebase logs
    with firebase_logs_lock:
        firebase_logs.append(log_text)

# Replace the built-in print function with our custom function
print = firebase_print

# Function to upload logs to Firebase
def upload_logs_to_firebase():
    global firebase_logs
    
    while not stop_event.is_set():
        time.sleep(firebase_upload_interval)
        
        with firebase_logs_lock:
            if not firebase_logs:
                continue
            
            logs_to_upload = firebase_logs.copy()
            firebase_logs = []
        
        # Create a timestamp for the log entry
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Organize data structure
        data = {
            "timestamp": timestamp,
            "logs": {}
        }
        
        # Add each statement with an index
        for i, log in enumerate(logs_to_upload):
            data["logs"][f"log_{i}"] = log
        
        # Upload to Firebase
        upload_url = f"{FIREBASE_URL}/traffic_system_logs/{timestamp}.json"
        
        try:
            response = requests.put(upload_url, data=json.dumps(data))
            response.raise_for_status()  # Raise an exception for HTTP errors
            print(f"✅ Successfully uploaded {len(logs_to_upload)} log statements to Firebase at {timestamp}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Error uploading to Firebase: {e}")
            # Put logs back in the queue
            with firebase_logs_lock:
                firebase_logs = logs_to_upload + firebase_logs

# --- Configuration Constants ---
# YAMNet and Emergency Detection Configuration
# Global state for emergency mode
emergency_mode_active = False
emergency_street_index = -1  # 0, 1, or 2
emergency_start_time = 0
emergency_duration = 60  # seconds
emergency_lock = threading.Lock()
pedestrian_lock = threading.Lock()  # Separate lock for pedestrian mode
stop_event = threading.Event() # For stopping threads gracefully

# Variable to save green signal before emergency
pre_emergency_green = None

# YAMNet Model and Audio Settings
YAMNET_MODEL_HANDLE = "https://tfhub.dev/google/yamnet/1"
yamnet_model = None
yamnet_class_names = []
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
# Duration of audio fed to YAMNet (0.96s for 16kHz)
YAMNET_EXPECTED_WAVEFORM_LENGTH = int(AUDIO_SAMPLE_RATE * 0.96)

# User-configurable microphone device indices for 3 streets
MIC_DEVICE_INDICES = [1, 0, 2] # Default, user should change this
EMERGENCY_SOUND_KEYWORDS = ["siren", "ambulance", "emergency", "police", "fire truck", "alarm"]
EMERGENCY_CONFIDENCE_THRESHOLD = 0.2 # Minimum score to consider a sound an emergency

# Pedestrian Mode Configuration
WAIT_AFTER_BUTTON = 5             # Delay before activating pedestrian mode
MOTION_TIMEOUT = 3                # No motion time to stop pedestrian green light
MAX_PEDESTRIAN_TIME = 40          # Maximum time for pedestrian green light
PEDESTRIAN_COOLDOWN_TIME = 60     # Cooldown before allowing next pedestrian cycle per crossing
PEDESTRIAN_GRACE_PERIOD = 10      # Grace period after activation before checking for motion (seconds)

# Traffic Signal Configuration
MIN_GREEN_TIME = 20               # Minimum time for green signal (seconds)
MAX_GREEN_TIME = 60               # Maximum time for green signal when other streets have vehicles
LONG_WAIT_THRESHOLD = 50          # Time in seconds to consider a wait as "long"
LONG_WAIT_GREEN_TIME = 30         # Green time given to streets with long wait
EMPTY_STREET_TIMEOUT = 2          # Time in seconds to consider a street as "empty" and change signal

# Vehicle Weight Configuration
CAR_WEIGHT = 1                    # Weight for cars
BUS_WEIGHT = 3                    # Weight for buses
TRUCK_WEIGHT = 2                  # Weight for trucks

# --- Global State Variables ---
# Pedestrian Mode
traffic_locked = False
pedestrian_mode_active = False
pedestrian_mode_start_time = 0
last_motion_time = 0
motion_detected_flags = [False] * 3
active_pedestrian_crossing = -1
pedestrian_cooldown_until = [0] * 3

# Traffic Light State
previous_green = None
green_start_time = 0
green_reason = None
green_lock_until = 0
all_streets_empty = False

# Traffic Analysis State
last_process_time = 0
last_wait_update = 0
wait_times = [0, 0, 0]
wait_start_time = [0, 0, 0]
previous_counts = [0, 0, 0]
empty_since = [0, 0, 0]

# Data Queues
audio_data_queue = queue.Queue()

# --- Pedestrian Mode Functions ---
def activate_pedestrian_mode(crossing_index):
    global traffic_locked, pedestrian_mode_active, pedestrian_mode_start_time, last_motion_time, active_pedestrian_crossing
    global wait_times, wait_start_time, previous_counts, empty_since, motion_detected_flags
    current_time = time.time()
    
    with pedestrian_lock:
        # Check for emergency mode safely
        emergency_active = False
        with emergency_lock:
            emergency_active = emergency_mode_active
            
        if emergency_active:
            print(f"🚫 [Pedestrian {crossing_index + 1}] Cannot activate: Emergency Mode is active.")
            return
        if pedestrian_mode_active:
            print(f"🚫 [Pedestrian {crossing_index + 1}] Cannot activate: Already active.")
            return
        if current_time < pedestrian_cooldown_until[crossing_index]:
            remaining_cooldown = int(pedestrian_cooldown_until[crossing_index] - current_time)
            print(f"🚫 [Pedestrian {crossing_index + 1}] Cannot activate: Still in cooldown for {remaining_cooldown} seconds.")
            return
        print(f"🚶‍♂️ Activating pedestrian mode triggered by crossing {crossing_index + 1}")
        
        # Step 1: Turn off all traffic signals (all red)
        for i in range(3):
            signals[i]["green"].off()
            signals[i]["red"].on()
            
        # Step 2: Wait before activating pedestrian signals
        time.sleep(WAIT_AFTER_BUTTON)
        
        # Check for emergency mode again after waiting
        with emergency_lock:
            if emergency_mode_active:
                print(f"🚫 [Pedestrian {crossing_index + 1}] Cannot activate: Emergency Mode became active during wait.")
                return
        
        # Step 3: Turn on all pedestrian green signals
        for i in range(3):
            pedestrian_signals[i]["red"].off()
            pedestrian_signals[i]["green"].on()
            
        # Step 4: Update global state
        pedestrian_mode_active = True
        pedestrian_mode_start_time = current_time
        last_motion_time = current_time  # Initialize last motion time to current time
        active_pedestrian_crossing = crossing_index
        traffic_locked = True
        
        # Initialize motion flag for active crossing to True to give grace period
        motion_detected_flags[crossing_index] = True
        print(f"✅ Pedestrian motion detection initialized for crossing {crossing_index + 1}")

def deactivate_pedestrian_mode(triggered_by_emergency=False):
    global pedestrian_mode_active, active_pedestrian_crossing, pedestrian_cooldown_until, traffic_locked
    global previous_green, green_start_time, green_reason, motion_detected_flags
    
    with pedestrian_lock:
        if not pedestrian_mode_active:
            return
        print(f"🛑 Deactivating pedestrian mode.")
        
        # Step 1: Turn off all pedestrian green lights and turn on red immediately
        for i in range(3):
            pedestrian_signals[i]["green"].off()
            pedestrian_signals[i]["red"].on()
            
        # Step 2: Update global state
        pedestrian_mode_active = False
        
        # Reset motion detection flags
        if active_pedestrian_crossing >= 0:
            motion_detected_flags[active_pedestrian_crossing] = False
        
        # Set cooldown for this crossing
        if active_pedestrian_crossing >= 0:
            pedestrian_cooldown_until[active_pedestrian_crossing] = time.time() + PEDESTRIAN_COOLDOWN_TIME
            
        active_pedestrian_crossing = -1
        traffic_locked = False
        
        # Step 3: Wait before restoring traffic signals (unless triggered by emergency)
        if not triggered_by_emergency:
            print(f"⏳ Waiting {WAIT_AFTER_BUTTON} seconds before restoring traffic signals...")
            time.sleep(WAIT_AFTER_BUTTON)
            
        # Step 4: Reset all traffic signals to red
        for i in range(3):
            signals[i]["red"].on()
            signals[i]["green"].off()
            
        # Step 5: Determine which street should get green signal
        # Find street with highest vehicle count
        current_time = time.time()
        highest_count = 0
        highest_count_street = -1
        
        for i in range(3):
            if previous_counts[i] > highest_count:
                highest_count = previous_counts[i]
                highest_count_street = i
                
        # If there are vehicles, activate green for that street
        if highest_count_street >= 0 and highest_count > 0:
            print(f"🚦 Restoring traffic flow: Street {highest_count_street + 1} gets green signal with {highest_count} vehicles")
            signals[highest_count_street]["red"].off()
            signals[highest_count_street]["green"].on()
            previous_green = highest_count_street
            green_start_time = current_time
            green_reason = "post_pedestrian"
            
            # Reset wait time for the street that gets green signal
            wait_times[highest_count_street] = 0
            wait_start_time[highest_count_street] = 0
            print(f"⏱️ Reset wait timer for Street {highest_count_street+1} (got green signal)")
        else:
            print("🚦 No vehicles detected on any street, all signals remain red")
            previous_green = None

def monitor_pir_sensor(index):
    pir = pedestrian_signals[index]["pir"]
    def motion_detected():
        # Only update motion flag if pedestrian mode is active and this is the active crossing
        with pedestrian_lock:
            if pedestrian_mode_active and active_pedestrian_crossing == index:
                motion_detected_flags[index] = True
                print(f"👀 Motion detected at crossing {index+1}")
    def motion_stopped():
        with pedestrian_lock:
            if pedestrian_mode_active and active_pedestrian_crossing == index:
                print(f"🔍 Motion stopped at crossing {index+1}")
        motion_detected_flags[index] = False
    pir.when_motion = motion_detected
    pir.when_no_motion = motion_stopped
    print(f"✅ Started monitoring PIR sensor {index+1}")
    stop_event.wait()
    print(f"🛑 Stopped monitoring PIR sensor {index+1}")

def monitor_pedestrian_activity():
    global last_motion_time, pedestrian_mode_active, active_pedestrian_crossing
    while not stop_event.is_set():
        # Use pedestrian lock to check state
        is_pedestrian_active = False
        active_crossing = -1
        ped_start_time = 0
        
        with pedestrian_lock:
            is_pedestrian_active = pedestrian_mode_active
            active_crossing = active_pedestrian_crossing
            ped_start_time = pedestrian_mode_start_time
        
        if is_pedestrian_active and active_crossing != -1:
            # Check for emergency mode safely
            emergency_active = False
            with emergency_lock:
                emergency_active = emergency_mode_active
                
            if emergency_active:
                print(f"🚨 Emergency mode detected while pedestrian mode is active. Deactivating pedestrian mode.")
                deactivate_pedestrian_mode(triggered_by_emergency=True)
                time.sleep(0.5)
                continue
            
            current_time = time.time()
            
            # Use pedestrian lock to access pedestrian mode variables
            with pedestrian_lock:
                elapsed_time = current_time - ped_start_time
                if elapsed_time >= MAX_PEDESTRIAN_TIME:
                    print(f"⏱️ Pedestrian time expired ({MAX_PEDESTRIAN_TIME} seconds). Stopping pedestrian mode.")
            
            if elapsed_time >= MAX_PEDESTRIAN_TIME:
                deactivate_pedestrian_mode()
                continue
            
            # Check if we're still in the grace period
            grace_period_active = elapsed_time < PEDESTRIAN_GRACE_PERIOD
            if grace_period_active:
                # During grace period, always update last_motion_time
                last_motion_time = current_time
                continue
                
            motion_at_active_crossing = motion_detected_flags[active_crossing]
            
            if motion_at_active_crossing:
                last_motion_time = current_time
            elif current_time - last_motion_time >= MOTION_TIMEOUT:
                print(f"🛑 No pedestrian motion for {MOTION_TIMEOUT} seconds at crossing {active_crossing+1}. Stopping pedestrian mode.")
                deactivate_pedestrian_mode()
        time.sleep(0.5)

def handle_pedestrian_button(index):
    button = pedestrian_signals[index]["button"]
    while not stop_event.is_set():
        try:
            button.wait_for_press(timeout=None)
            if stop_event.is_set():
                break
            print(f"🔘 [Pedestrian Signal {index + 1}] Button pressed, waiting {WAIT_AFTER_BUTTON} seconds...")
            wait_end_time = time.time() + WAIT_AFTER_BUTTON
            while time.time() < wait_end_time and not stop_event.is_set():
                time.sleep(0.1)
            if stop_event.is_set():
                break
            
            # Check for emergency mode before activating pedestrian mode
            emergency_active = False
            with emergency_lock:
                emergency_active = emergency_mode_active
                
            if not emergency_active:
                activate_pedestrian_mode(index)
            else:
                print(f"🚫 [Pedestrian {index + 1}] Cannot activate: Emergency Mode is active.")
                continue
                
            # Use local variables to check state
            while True:
                is_pedestrian_active = False
                is_emergency_active = False
                
                with pedestrian_lock:
                    is_pedestrian_active = pedestrian_mode_active
                    
                with emergency_lock:
                    is_emergency_active = emergency_mode_active
                    
                if not is_pedestrian_active or is_emergency_active or stop_event.is_set():
                    break
                    
                time.sleep(0.5)
                
            if stop_event.is_set():
                break
        except Exception as e:
            if stop_event.is_set():
                break
            print(f"❌ Error in handle_pedestrian_button {index+1}: {e}")
            time.sleep(1)

# --- End of Pedestrian Mode Functions ---
# --- Emergency Detection Functions ---
def load_yamnet_model():
    global yamnet_model, yamnet_class_names
    try:
        print("⏳ Loading YAMNet model...")
        yamnet_model = hub.load(YAMNET_MODEL_HANDLE)
        # Get class names
        class_map_path = yamnet_model.class_map_path().numpy().decode("utf-8")
        with open(class_map_path) as f:
            yamnet_class_names = [line.strip() for line in f.readlines()]
        print("✅ YAMNet model loaded successfully.")
        return True
    except Exception as e:
        print(f"❌ Error loading YAMNet model: {e}")
        return False

def preprocess_waveform(waveform):
    # Ensure waveform is mono
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    # Ensure waveform is float32 in range [-1.0, 1.0]
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)
        # Assuming input might be int16 from sounddevice, scale it
        if np.max(np.abs(waveform)) > 1.0:
             waveform /= 32768.0
    # Ensure correct length for YAMNet
    if len(waveform) < YAMNET_EXPECTED_WAVEFORM_LENGTH:
        waveform = np.pad(waveform, (0, YAMNET_EXPECTED_WAVEFORM_LENGTH - len(waveform)), "constant")
    elif len(waveform) > YAMNET_EXPECTED_WAVEFORM_LENGTH:
        waveform = waveform[:YAMNET_EXPECTED_WAVEFORM_LENGTH]
    return waveform

def analyze_audio_chunk(waveform, street_idx):
    global emergency_mode_active, emergency_street_index, emergency_start_time, pre_emergency_green
    global wait_times, wait_start_time
    
    if yamnet_model is None:
        return
    processed_waveform = preprocess_waveform(waveform)
    scores, embeddings, log_mel_spectrogram = yamnet_model(processed_waveform)
    scores_np = scores.numpy()[0] # Get scores for the single frame
    detected_emergency_sound = False
    highest_score_for_emergency = 0
    detected_sound_name = ""
    for i, score in enumerate(scores_np):
        class_name = yamnet_class_names[i].lower()
        if any(keyword in class_name for keyword in EMERGENCY_SOUND_KEYWORDS):
            if score > EMERGENCY_CONFIDENCE_THRESHOLD and score > highest_score_for_emergency:
                highest_score_for_emergency = score
                detected_emergency_sound = True
                detected_sound_name = yamnet_class_names[i]
    if detected_emergency_sound:
        with emergency_lock:
            # Only trigger if not already in emergency mode for a *different* street, 
            # or if the current emergency is for this street (to potentially refresh timer or log)
            if not emergency_mode_active or (emergency_mode_active and emergency_street_index == street_idx):
                if not emergency_mode_active: # New emergency
                    print(f"🚨 EMERGENCY DETECTED on Street {street_idx + 1}! Sound: {detected_sound_name} (Score: {highest_score_for_emergency:.2f})")
                    
                    # Save previous green signal before emergency
                    pre_emergency_green = previous_green
                    print(f"💾 Saved previous green signal: Street {pre_emergency_green + 1 if pre_emergency_green is not None else 'None'}")
                    
                    # Check for pedestrian mode safely
                    is_pedestrian_active = False
                    with pedestrian_lock:
                        is_pedestrian_active = pedestrian_mode_active
                    
                    # Set emergency mode first to prevent pedestrian activation
                    emergency_mode_active = True  
                    emergency_street_index = street_idx
                    emergency_start_time = time.time()
                    
                    # Reset wait time for the emergency street
                    wait_times[street_idx] = 0
                    wait_start_time[street_idx] = 0
                    print(f"⏱️ Reset wait timer for Street {street_idx+1} (emergency detected)")
                    
                    # Deactivate pedestrian mode if active
                    if is_pedestrian_active:
                        print(f"🚨 Emergency detected while pedestrian mode is active. Forcing deactivation.")
                        deactivate_pedestrian_mode(triggered_by_emergency=True)
                    
                    # Reset all signals
                    reset_signals()
                    
                    # Activate green signal for emergency street
                    for i in range(3):
                        if i == street_idx:
                            signals[i]["red"].off()
                            signals[i]["green"].on()
                        else:
                            signals[i]["red"].on()
                            signals[i]["green"].off()
                    
                elif emergency_street_index == street_idx: # Re-detected for the same street
                    print(f"🚨 Emergency re-detected on Street {street_idx + 1}. Sound: {detected_sound_name} (Score: {highest_score_for_emergency:.2f})")
            elif emergency_mode_active and emergency_street_index != street_idx:
                 print(f"ℹ️ Emergency sound ({detected_sound_name}) on Street {street_idx + 1} ignored, emergency already active for Street {emergency_street_index + 1}.")

# Callback for sounddevice InputStream
def audio_callback(indata, frames, time_info, status):
    pass

def listen_for_emergency(street_idx, device_idx, stop_event_ref):
    print(f"🎤 Starting emergency sound detection for Street {street_idx + 1} (Mic Index: {device_idx})")
    stream = None
    try:
        def callback_wrapper(indata, frames, time_info, status):
            if status:
                print(f"🔊 Audio Status (Street {street_idx+1}, Mic {device_idx}): {status}", flush=True)
            analyze_audio_chunk(indata[:, 0], street_idx)
        stream = sd.InputStream(
            device=device_idx,
            channels=AUDIO_CHANNELS,
            samplerate=AUDIO_SAMPLE_RATE,
            callback=callback_wrapper,
            blocksize=YAMNET_EXPECTED_WAVEFORM_LENGTH, # Process audio in chunks expected by YAMNet
            dtype='float32' # YAMNet expects float32
        )
        stream.start()
        stop_event_ref.wait() # Keep thread alive until stop_event is set
    except Exception as e:
        print(f"❌ Error in audio thread for Street {street_idx + 1} (Mic {device_idx}): {e}")
    finally:
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as e_close:
                print(f"❌ Error closing stream for Street {street_idx + 1} (Mic {device_idx}): {e_close}")
        print(f"🛑 Stopped audio thread for Street {street_idx + 1} (Mic {device_idx})")

# --- End of Emergency Detection Functions ---
# Load YOLO model

model = YOLO("yolov8n.pt")

# Setup cameras
cam1 = cv2.VideoCapture("/dev/video2")
cam2 = cv2.VideoCapture("/dev/video0")
cam3 = cv2.VideoCapture("/dev/video4")
if not (cam1.isOpened() and cam2.isOpened() and cam3.isOpened()):
    print("⚠️ Problem starting cameras")
    exit()
print("✅ Cameras are working")

# Setup traffic signals
signals = [
    {"red": LED(23), "green": LED(24)},  # Street 1
    {"red": LED(15), "green": LED(14)},  # Street 2
    {"red": LED(21), "green": LED(20)}   # Street 3
]

# Setup pedestrian signals
pedestrian_signals = [
    {"red": LED(26), "green": LED(25), "button": Button(11, pull_up=False), "pir": MotionSensor(22, queue_len=1, sample_rate=10, threshold=0.2)}, # Street 1
    {"red": LED(16), "green": LED(12), "button": Button(9, pull_up=False), "pir": MotionSensor(27, queue_len=1, sample_rate=10, threshold=0.2)},  # Street 2
    {"red": LED(6),  "green": LED(5),  "button": Button(10, pull_up=False), "pir": MotionSensor(17, queue_len=1, sample_rate=10, threshold=0.2)}  # Street 3
]

def reset_signals():
    for sig in signals:
        sig["red"].on()
        sig["green"].off()
    for ped in pedestrian_signals:
        ped["red"].on()
        ped["green"].off()
reset_signals()

# Define function to count vehicles (car, bus, truck)
def count_cars(results):
    count = 0
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            if cls == 2:  # car
                count += CAR_WEIGHT
            elif cls == 5:  # bus
                count += BUS_WEIGHT
            elif cls == 7:  # truck
                count += TRUCK_WEIGHT
    return count

# Helper function to find street with highest vehicle count
def find_street_with_highest_count(counts, exclude_street=None):
    highest_count = 0
    highest_count_street = -1
    
    for i in range(3):
        if i != exclude_street and counts[i] > highest_count:
            highest_count = counts[i]
            highest_count_street = i
            
    return highest_count_street, highest_count

# Helper function to activate green signal for a street
def activate_green_signal(street_index, reason, current_time):
    global previous_green, green_start_time, green_reason, wait_times, wait_start_time
    
    if street_index < 0 or street_index > 2:
        print("⚠️ Invalid street index for green signal activation")
        return False
        
    # Turn off all signals first
    for i in range(3):
        signals[i]["green"].off()
        signals[i]["red"].on()
    
    # Turn on green for selected street
    signals[street_index]["red"].off()
    signals[street_index]["green"].on()
    
    # Update state
    previous_green = street_index
    green_start_time = current_time
    green_reason = reason
    
    # Reset wait time for the street that gets green signal
    wait_times[street_index] = 0
    wait_start_time[street_index] = 0
    print(f"⏱️ Reset wait timer for Street {street_index+1} (got green signal)")
    
    return True

try:
    print("🚀 Starting Smart Traffic System with Firebase integration")
    
    # Start Firebase upload thread
    firebase_thread = threading.Thread(target=upload_logs_to_firebase, daemon=True)
    firebase_thread.start()
    print("📤 Firebase upload thread started")
    
    if not load_yamnet_model():
        print("❌ Exiting due to YAMNet model loading failure.")
        exit(1)
    # Start audio listening threads for emergency detection
    audio_threads = []
    for i in range(len(MIC_DEVICE_INDICES)):
        # Check if a valid device index is provided for the street
        if i < len(MIC_DEVICE_INDICES) and MIC_DEVICE_INDICES[i] is not None:
            thread = threading.Thread(target=listen_for_emergency, args=(i, MIC_DEVICE_INDICES[i], stop_event), daemon=True)
            audio_threads.append(thread)
            thread.start()
        else:
            print(f"⚠️ Warning: No microphone device index configured for Street {i+1}. Emergency detection will not work for this street.")
    # Start pedestrian monitoring threads
    pedestrian_threads = []
    # Start PIR sensor monitoring threads
    for i in range(3):
        thread = threading.Thread(target=monitor_pir_sensor, args=(i,), daemon=True)
        pedestrian_threads.append(thread)
        thread.start()
    # Start pedestrian activity monitoring thread
    pedestrian_activity_thread = threading.Thread(target=monitor_pedestrian_activity, daemon=True)
    pedestrian_threads.append(pedestrian_activity_thread)
    pedestrian_activity_thread.start()
    # Start pedestrian button monitoring threads
    for i in range(3):
        thread = threading.Thread(target=handle_pedestrian_button, args=(i,), daemon=True)
        pedestrian_threads.append(thread)
        thread.start()
        
    while True:
        current_time = time.time()
        
        # --- Emergency Mode Check and Control ---
        with emergency_lock:
            if emergency_mode_active:
                # Check for pedestrian mode safely
                is_pedestrian_active = False
                with pedestrian_lock:
                    is_pedestrian_active = pedestrian_mode_active
                
                # Ensure pedestrian mode is deactivated if active
                if is_pedestrian_active:
                    print(f"🚨 Emergency mode active but pedestrian mode still active. Forcing deactivation.")
                
                # Release emergency lock before calling deactivate_pedestrian_mode
                emergency_active = emergency_mode_active
                emergency_idx = emergency_street_index
                emergency_time = emergency_start_time
                
                if is_pedestrian_active:
                    # Temporarily release emergency lock
                    with emergency_lock:
                        pass
                    
                    # Deactivate pedestrian mode
                    deactivate_pedestrian_mode(triggered_by_emergency=True)
                    
                    # Reacquire emergency lock
                    with emergency_lock:
                        # Ensure emergency mode is still active
                        if not emergency_mode_active:
                            emergency_mode_active = emergency_active
                            emergency_street_index = emergency_idx
                            emergency_start_time = emergency_time
                
                if current_time - emergency_start_time >= emergency_duration:
                    print(f"⏳ Emergency mode for Street {emergency_street_index + 1} ended.")
                    
                    # Store emergency street index before clearing it
                    ended_emergency_street = emergency_street_index
                    
                    # Clear emergency mode
                    emergency_mode_active = False
                    emergency_street_index = -1
                    
                    # Reset wait time for the street that had emergency
                    wait_times[ended_emergency_street] = 0
                    wait_start_time[ended_emergency_street] = 0
                    print(f"⏱️ Reset wait timer for Street {ended_emergency_street+1} (emergency ended)")
                    
                    # Restore previous green signal after emergency
                    reset_signals() # First, reset all signals to red
                    
                    if pre_emergency_green is not None:
                        print(f"🔄 Restoring previous green signal for Street {pre_emergency_green + 1}")
                        # Activate green signal for street that was active before emergency
                        activate_green_signal(pre_emergency_green, "post_emergency_restore", current_time)
                        
                        # Reset pre-emergency green variable
                        pre_emergency_green = None
                    else:
                        print("ℹ️ No previous green signal to restore.")
                        
                        # Find street with highest vehicle count to give green signal
                        highest_count_street, highest_count = find_street_with_highest_count(previous_counts)
                                
                        # If there are vehicles, activate green for that street
                        if highest_count_street >= 0 and highest_count > 0:
                            print(f"🚦 Giving green signal to Street {highest_count_street + 1} with {highest_count} vehicles")
                            activate_green_signal(highest_count_street, "post_emergency_highest_count", current_time)
                else:
                    # Emergency mode is active, control signals accordingly
                    for i, sig in enumerate(signals):
                        if i == emergency_street_index:
                            sig["green"].on()
                            sig["red"].off()
                        else:
                            sig["green"].off()
                            sig["red"].on()
                    # Ensure all pedestrian signals are red
                    for ped in pedestrian_signals:
                        ped["green"].off()
                        ped["red"].on()
                    # Skip normal traffic logic while emergency mode is active
                    sleep(0.1) 
                    continue # Skip the rest of the loop and re-evaluate emergency status
        # --- End of Emergency Mode Check and Control ---
        
        # --- Pedestrian Mode Check ---
        # Use pedestrian lock to check state
        is_pedestrian_active = False
        with pedestrian_lock:
            is_pedestrian_active = pedestrian_mode_active
            
        if is_pedestrian_active:
            # Check for emergency mode safely
            emergency_active = False
            with emergency_lock:
                emergency_active = emergency_mode_active
                
            if emergency_active:
                print(f"🚨 Emergency mode detected in pedestrian check. Deactivating pedestrian mode.")
                deactivate_pedestrian_mode(triggered_by_emergency=True)
                continue
            # Skip normal traffic logic while pedestrian mode is active
            sleep(0.1)
            continue
        # --- End of Pedestrian Mode Check ---
        
        # If we reach here, both emergency and pedestrian modes are inactive
        # Proceed with normal traffic control logic
        
        # Update wait times for each street
        if current_time - last_wait_update >= 1:
            for i in range(3):
                if previous_green != i:  # Only update wait times for streets that don't have green signal
                    if previous_counts[i] > 0 and wait_start_time[i] == 0:
                        wait_start_time[i] = current_time
                        wait_times[i] = 0
                        print(f"⏱️ Started wait timer for Street {i+1}")
                    elif previous_counts[i] > 0 and wait_start_time[i] > 0:
                        wait_times[i] = int(current_time - wait_start_time[i])
                    elif previous_counts[i] == 0 and wait_start_time[i] > 0:
                        wait_times[i] = 0
                        wait_start_time[i] = 0
                        print(f"⏱️ Stopped wait timer for Street {i+1} (empty)")
            last_wait_update = current_time
        
        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        ret3, frame3 = cam3.read()
        if not (ret1 and ret2 and ret3):
            print("⚠️ Problem reading cameras")
            continue
        
        # Process images every 4 seconds
        if current_time - last_process_time >= 1:
            last_process_time = current_time
            results1 = model(frame1, imgsz=320)
            results2 = model(frame2, imgsz=320)
            results3 = model(frame3, imgsz=320)
            # Call count_cars to get the counts
            car_count1 = count_cars(results1)
            car_count2 = count_cars(results2)
            car_count3 = count_cars(results3)
            counts = [car_count1, car_count2, car_count3]
            print(f"🚘 Street 1: {counts[0]} vehicles, wait: {wait_times[0]} seconds")
            print(f"🚘 Street 2: {counts[1]} vehicles, wait: {wait_times[1]} seconds")
            print(f"🚘 Street 3: {counts[2]} vehicles, wait: {wait_times[2]} seconds")
            
            # Check for streets that suddenly became empty
            for i in range(3):
                if previous_counts[i] > 0 and counts[i] == 0:
                    if empty_since[i] == 0:
                        empty_since[i] = current_time
                        print(f"⚠️ Street {i+1} became empty, starting verification")
                    if wait_start_time[i] > 0:
                        wait_times[i] = 0
                        wait_start_time[i] = 0
                        print(f"⏱️ Stopped wait timer for Street {i+1} (became empty)")
                elif counts[i] > 0:
                    empty_since[i] = 0
                    
            previous_counts = counts.copy()
            
            # --- COMPLETELY REWRITTEN TRAFFIC CONTROL ALGORITHM ---
            
            # Check if all streets are empty
            if all(count == 0 for count in counts):
                if not all_streets_empty:
                    print("⚠️ All streets empty, all signals red")
                    all_streets_empty = True
                    if previous_green is not None:
                        signals[previous_green]["green"].off()
                        signals[previous_green]["red"].on()
                        previous_green = None
                        green_reason = None
                        green_lock_until = 0
                continue
            else:
                all_streets_empty = False
                
                # If no street has green signal but there are vehicles, we need to select one
                if previous_green is None:
                    # Find street with highest vehicle count
                    highest_count_street, highest_count = find_street_with_highest_count(counts)
                    
                    # If we found a street with vehicles, give it green signal
                    if highest_count_street >= 0 and highest_count > 0:
                        print(f"🚦 No street has green signal. Giving green to Street {highest_count_street + 1} with {highest_count} vehicles")
                        activate_green_signal(highest_count_street, "initial_highest_count", current_time)
                        
                        # Skip the rest of the algorithm since we just set a green signal
                        continue
            
            # Now we have at least one street with vehicles and a street with green signal
            # Let's determine if we need to change the green signal
            
            # Step 1: Check if current green signal can be changed
            can_change = True
            
            # Check minimum green time
            if previous_green is not None:
                green_time = current_time - green_start_time
                if green_time < MIN_GREEN_TIME:
                    can_change = False
                    remaining_min_time = int(MIN_GREEN_TIME - green_time)
                    print(f"⏱️ Minimum green time not reached, {remaining_min_time} seconds remaining")
            
            # Step 2: Check if current green street is empty
            if previous_green is not None and can_change:
                if counts[previous_green] == 0 and empty_since[previous_green] > 0:
                    empty_time = current_time - empty_since[previous_green]
                    if empty_time >= EMPTY_STREET_TIMEOUT:
                        print(f"🚫 Street {previous_green+1} empty for {int(empty_time)} seconds, changing signal")
                        
                        # Find street with highest vehicle count
                        next_street, next_count = find_street_with_highest_count(counts, previous_green)
                        
                        # If we found a street with vehicles, give it green signal
                        if next_street >= 0 and next_count > 0:
                            # Turn off current green and activate new green
                            print(f"🟢 Changed signal from Street {previous_green+1} (empty) to Street {next_street+1} with {next_count} vehicles")
                            activate_green_signal(next_street, "empty_street_switch", current_time)
                        else:
                            # No other street has vehicles, turn off all signals
                            print("⚠️ No other street has vehicles, all signals red")
                            signals[previous_green]["green"].off()
                            signals[previous_green]["red"].on()
                            previous_green = None
                            green_reason = None
                            all_streets_empty = True
                        
                        # Skip the rest of the algorithm since we just changed the signal
                        continue
            
            # Step 3: Check for long wait priority
            if can_change:
                # Find streets with long wait times
                long_wait_streets = []
                for i in range(3):
                    if i != previous_green and counts[i] > 0 and wait_times[i] >= LONG_WAIT_THRESHOLD:
                        long_wait_streets.append(i)
                
                if long_wait_streets:
                    # Find street with longest wait time
                    longest_wait_street = max(long_wait_streets, key=lambda i: wait_times[i])
                    longest_wait_time = wait_times[longest_wait_street]
                    
                    print(f"🕒 Street {longest_wait_street+1} has waited for {longest_wait_time} seconds (threshold: {LONG_WAIT_THRESHOLD})")
                    
                    # Turn off current green and activate new green
                    print(f"🟢 Changed signal from Street {previous_green+1} to Street {longest_wait_street+1} due to long wait")
                    activate_green_signal(longest_wait_street, "long_wait", current_time)
                    
                    # Skip the rest of the algorithm since we just changed the signal
                    continue
            
            # Step 4: Check if another street has significantly more vehicles
            if can_change:
                # Find street with highest vehicle count (excluding current green)
                highest_count_street, highest_count = find_street_with_highest_count(counts, previous_green)
                
                # Only change if the highest count street has more vehicles than current green
                if highest_count_street >= 0 and highest_count > 0 and counts[highest_count_street] > counts[previous_green]:
                    print(f"🚗 Street {highest_count_street+1} has {counts[highest_count_street]} vehicles, more than Street {previous_green+1} with {counts[previous_green]} vehicles")
                    
                    # Turn off current green and activate new green
                    print(f"🟢 Changed signal from Street {previous_green+1} to Street {highest_count_street+1} due to higher vehicle count")
                    activate_green_signal(highest_count_street, "higher_count", current_time)
            
            # --- END OF COMPLETELY REWRITTEN TRAFFIC CONTROL ALGORITHM ---
                
        time.sleep(0.1)
except KeyboardInterrupt:
    print("👋 Program stopped by user")
    # Force one final upload of logs before exiting
    with firebase_logs_lock:
        if firebase_logs:
            # Create a timestamp for the log entry
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            # Organize data structure
            data = {
                "timestamp": timestamp,
                "logs": {}
            }
            
            # Add each statement with an index
            for i, log in enumerate(firebase_logs):
                data["logs"][f"log_{i}"] = log
            
            # Upload to Firebase
            upload_url = f"{FIREBASE_URL}/traffic_system_logs/{timestamp}.json"
            
            try:
                response = requests.put(upload_url, data=json.dumps(data))
                response.raise_for_status()
                print(f"✅ Final upload: Successfully uploaded {len(firebase_logs)} log statements to Firebase")
            except requests.exceptions.RequestException as e:
                print(f"❌ Error in final upload to Firebase: {e}")
except Exception as e:
    print(f"❌ Error occurred: {e}")
finally:
    # Clean up resources
    stop_event.set()
    print("🧹 Cleaning up resources...")
    
    # Stop cameras
    cam1.release()
    cam2.release()
    cam3.release()
    
    # Reset all signals
    reset_signals()
    
    # Wait for threads to finish
    time.sleep(1)
    print("✅ Program stopped successfully")
