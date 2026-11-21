
#include "mfcc.h"

#include <Arduino.h>
#include <math.h>
#include <complex>
#include <vector>
#include <algorithm>
#include <limits>

using namespace std;

static const double MY_PI = 3.14159265358979323846;
static const float EPSILON = 1e-10f;

static double hzToMel(double hz) {
    const double f_sp = 200.0 / 3.0;

    if (hz < 1000.0) return hz / f_sp;

    const double min_log_mel = 1000.0 / f_sp;
    const double logstep = log(6.4) / 27.0;

    return min_log_mel + log(hz / 1000.0) / logstep;
}

static double melToHz(double mel) {
    const double f_sp = 200.0 / 3.0;
    const double min_log_mel = 1000.0 / f_sp;
    const double logstep = log(6.4) / 27.0;

    if (mel < min_log_mel) return mel * f_sp;

    return 1000.0 * exp(logstep * (mel - min_log_mel));
}

static void fft(vector<complex<float>>& data) {
    const int n = data.size();
    int j = 0;

    for (int i = 1; i < n; i++) {
        int bit = n >> 1;

        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }

        j ^= bit;

        if (i < j) swap(data[i], data[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        float angle = -2.0f * (float)MY_PI / len;

        complex<float> wlen(cosf(angle), sinf(angle));

        for (int i = 0; i < n; i += len) {
            complex<float> w(1.0f, 0.0f);

            for (int k = 0; k < len / 2; k++) {
                complex<float> u = data[i + k];
                complex<float> v = data[i + k + len / 2] * w;

                data[i + k] = u + v;
                data[i + k + len / 2] = u - v;

                w *= wlen;
            }
        }
    }
}

static vector<vector<float>> makeMelFilters() {
    vector<vector<float>> filters(
        MFCC_NUM_MELS,
        vector<float>(MFCC_FFT_SIZE / 2 + 1, 0.0f)
    );

    double min_mel = hzToMel(0.0);
    double max_mel = hzToMel(AUDIO_SAMPLE_RATE / 2.0);

    vector<double> mel_points(MFCC_NUM_MELS + 2);

    for (int i = 0; i < MFCC_NUM_MELS + 2; i++) {
        mel_points[i] = min_mel +
            (max_mel - min_mel) * i / (MFCC_NUM_MELS + 1);
    }

    vector<double> hz_points(MFCC_NUM_MELS + 2);

    for (int i = 0; i < MFCC_NUM_MELS + 2; i++) {
        hz_points[i] = melToHz(mel_points[i]);
    }

    for (int m = 0; m < MFCC_NUM_MELS; m++) {
        double left = hz_points[m];
        double center = hz_points[m + 1];
        double right = hz_points[m + 2];

        double enorm = 2.0 / (right - left);

        for (int k = 0; k <= MFCC_FFT_SIZE / 2; k++) {
            double freq = (double)k * AUDIO_SAMPLE_RATE / MFCC_FFT_SIZE;

            double weight = 0.0;

            if (freq >= left && freq <= center) {
                weight = (freq - left) / (center - left);
            } else if (freq > center && freq <= right) {
                weight = (right - freq) / (right - center);
            }

            filters[m][k] = (float)(weight * enorm);
        }
    }

    return filters;
}

bool extractMFCCFeatures(const float* audio, float* features) {
    if (audio == nullptr || features == nullptr) return false;

    const int frame_count = MFCC_NUM_FRAMES;
    const int bins = MFCC_FFT_SIZE / 2 + 1;

    vector<vector<float>> filters = makeMelFilters();

    vector<vector<float>> mel_power(
        MFCC_NUM_MELS, vector<float>(frame_count, 0.0f)
    );

    vector<complex<float>> fft_data(MFCC_FFT_SIZE);

    // STFT and mel power
    for (int frame = 0; frame < frame_count; frame++) {
        int frame_start = frame * MFCC_HOP_LENGTH - MFCC_FFT_SIZE / 2;

        for (int i = 0; i < MFCC_FFT_SIZE; i++) {
            int index = frame_start + i;
            float sample = 0.0f;

            if (index >= 0 && index < AUDIO_WINDOW_SIZE) {
                sample = audio[index];
            }

            float window = 0.5f - 0.5f * cosf(
                2.0f * (float)MY_PI * i / MFCC_FFT_SIZE
            );

            fft_data[i] = complex<float>(sample * window, 0.0f);
        }

        fft(fft_data);

        float power[bins];

        for (int k = 0; k < bins; k++) {
            float real = fft_data[k].real();
            float imag = fft_data[k].imag();
            power[k] = real * real + imag * imag;
        }

        for (int m = 0; m < MFCC_NUM_MELS; m++) {
            float energy = 0.0f;

            for (int k = 0; k < bins; k++) {
                energy += filters[m][k] * power[k];
            }

            mel_power[m][frame] = energy;
        }
    }

    // Convert mel power to dB and find GLOBAL maximum
    vector<vector<float>> log_mel(
        MFCC_NUM_MELS, vector<float>(frame_count, 0.0f)
    );

    float max_db = -INFINITY;

    for (int m = 0; m < MFCC_NUM_MELS; m++) {
        for (int t = 0; t < frame_count; t++) {
            float energy = max(EPSILON, mel_power[m][t]);

            float db = 10.0f * log10f(energy);

            log_mel[m][t] = db;

            if (db > max_db) max_db = db;
        }
    }

    // Apply global top_db=80 clipping BEFORE DCT
    float floor_db = max_db - 80.0f;

    for (int m = 0; m < MFCC_NUM_MELS; m++) {
        for (int t = 0; t < frame_count; t++) {
            log_mel[m][t] = max(floor_db, log_mel[m][t]);
        }
    }

    // Orthonormal DCT-II
    vector<vector<float>> mfcc_values(
        MFCC_NUM_COEFFICIENTS,
        vector<float>(frame_count, 0.0f)
    );

    for (int c = 0; c < MFCC_NUM_COEFFICIENTS; c++) {
        double scale = (c == 0)
            ? sqrt(1.0 / MFCC_NUM_MELS)
            : sqrt(2.0 / MFCC_NUM_MELS);

        for (int t = 0; t < frame_count; t++) {
            double sum = 0.0;

            for (int m = 0; m < MFCC_NUM_MELS; m++) {
                sum += log_mel[m][t] *
                    cos(MY_PI * c * (m + 0.5) / MFCC_NUM_MELS);
            }

            mfcc_values[c][t] = (float)(sum * scale);
        }
    }

    // 20 means + 20 population standard deviations
    for (int c = 0; c < MFCC_NUM_COEFFICIENTS; c++) {
        float mean = 0.0f;

        for (int t = 0; t < frame_count; t++) {
            mean += mfcc_values[c][t];
        }

        mean /= frame_count;

        float variance = 0.0f;

        for (int t = 0; t < frame_count; t++) {
            float difference = mfcc_values[c][t] - mean;
            variance += difference * difference;
        }

        variance /= frame_count;

        features[c] = mean;
        features[c + MFCC_NUM_COEFFICIENTS] = sqrtf(variance);
    }

    return true;
}
