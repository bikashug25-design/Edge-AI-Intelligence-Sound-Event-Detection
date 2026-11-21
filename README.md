# Edge-AI-Based Intelligent Sound Event Detection and Alert System

## Overview
An Edge-AI-based sound event detection and alert system designed to identify important environmental sound events using an ESP32-S3 microcontroller.

The system uses Machine Learning to classify incoming audio into three categories:
- **ALARM**
- **BACKGROUND**
- **IMPACT**

When an alarm or impact event is detected, the system activates a buzzer and LED to provide an alert.

## Key Features
- Real-time sound event detection
- MFCC-based audio feature extraction
- Support Vector Machine (SVM) classification
- Local Edge-AI inference on ESP32-S3
- Temporal voting to improve prediction stability
- Arduino Uno-based buzzer and LED alert controller
- Wi-Fi-based audio streaming from an Android smartphone

## System Architecture
1. Android smartphone captures audio.
2. Audio is streamed to ESP32-S3 over Wi-Fi.
3. ESP32-S3 processes audio and extracts MFCC features.
4. SVM classifier predicts the sound category.
5. Confirmed ALARM or IMPACT events trigger the alert controller.
6. Arduino Uno activates the buzzer and LED.

## Machine Learning Details
- **Learning type:** Supervised Machine Learning
- **Problem type:** Multi-class classification
- **Feature extraction:** Mel-Frequency Cepstral Coefficients (MFCC)
- **Classifier:** Support Vector Machine (SVM) with RBF kernel
- **Sampling rate:** 16 kHz
- **Audio window:** 1 second
- **Number of MFCC coefficients:** 20
- **Number of extracted features:** 40

### Model Performance
The SVM model achieved **87.50% mean accuracy** in 5-fold cross-validation on the selected dataset.

This is cross-validation performance, not a guarantee of real-world detection accuracy.

## Hardware Components
- ESP32-S3 development board
- Arduino Uno
- Android smartphone
- Active buzzer
- LED
- BC547 NPN transistor
- Resistors, breadboard and jumper wires

## Repository Structure
```text
Edge-AI-Sound-Event-Detection/
├── ESP32_S3/
├── Arduino_Uno/
├── ML_Model/
├── Documentation/
├── Images/
├── README.md
└── .gitignore
```

## Dataset
The model was developed using a selected subset of the ESC-50 environmental sound dataset.

Original audio recordings are not included in this repository. Obtain the dataset separately and follow its license and usage conditions.

## Requirements
Python dependencies and setup instructions will be provided in the `ML_Model` folder.

## Limitations
- The model currently recognizes three sound classes.
- Detection performance depends on microphone quality and environmental noise.
- Wi-Fi connectivity is required for smartphone-to-ESP32 audio streaming.
- Further testing is required in real-world environments.

## Future Improvements
- Expand the number of detectable sound classes.
- Improve robustness in noisy environments.
- Optimize model inference and memory usage.
- Develop a dedicated Android application.
- Conduct extensive real-world testing.

## Disclaimer
This is an educational prototype and is not a certified industrial or life-safety monitoring system.

## Contributors
Group 17 – Department of Electronics and Instrumentation Engineering  
National Institute of Technology Silchar

## Mentor
Dr. Vipin Chandra Pal  
Assistant Professor, Electronics and Instrumentation Engineering

## License
License information will be added after confirmation by the project contributors.
