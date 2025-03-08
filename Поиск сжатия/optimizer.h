#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>
#include <filesystem>
#include <random>
#include <numeric>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <fstream>

#include "learning.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

struct Image {
    int width, height;
    std::vector<unsigned char> data;
    Image(int w, int h) : width(w), height(h), data(w * h * 3) {}
    Image(int w, int h, const unsigned char* d) : width(w), height(h), data(d, d + w * h * 3) {}
};

class ImageOptimizer {
private:
    std::mutex mtx;
    std::atomic<bool> found_better;
    LearningState learning_state;

    double calculate_entropy(const Image& image) const {
        alignas(16) int histogram[3][256] = {{0}, {0}, {0}};
        size_t pixel_count = image.width * image.height;
        double gradient_entropy = 0.0;

        #pragma omp parallel for reduction(+:gradient_entropy)
        for (int y = 1; y < image.height - 1; ++y) {
            for (int x = 1; x < image.width - 1; ++x) {
                size_t idx = (y * image.width + x) * 3;
                float gray = (image.data[idx] + image.data[idx + 1] + image.data[idx + 2]) / 3.0f;
                float gx = (image.data[((y + 1) * image.width + x) * 3] - image.data[((y - 1) * image.width + x) * 3]) / 3.0f;
                float gy = (image.data[(y * image.width + x + 1) * 3] - image.data[(y * image.width + x - 1) * 3]) / 3.0f;
                gradient_entropy += std::sqrt(gx * gx + gy * gy);
                histogram[0][image.data[idx]]++;
                histogram[1][image.data[idx + 1]]++;
                histogram[2][image.data[idx + 2]]++;
            }
        }
        gradient_entropy /= pixel_count * 255.0;

        double entropy = 0.0;
        double max_entropy = std::log2(256.0);
        for (int channel = 0; channel < 3; ++channel) {
            double channel_entropy = 0.0;
            for (int i = 0; i < 256; ++i) {
                if (histogram[channel][i] > 0) {
                    double p = static_cast<double>(histogram[channel][i]) / pixel_count;
                    channel_entropy -= p * std::log2(p);
                }
            }
            entropy += channel_entropy / max_entropy * (1.0 + gradient_entropy * 1.5);
        }
        return entropy / 3.0;
    }

    void compute_laplacian(const Image& image, std::vector<float>& laplacian, float noise_reduction) const {
        laplacian.resize(image.width * image.height);
        float noise_threshold = noise_reduction * 20.0f;

        #pragma omp parallel for collapse(2)
        for (int y = 1; y < image.height - 1; y++) {
            for (int x = 1; x < image.width - 1; x++) {
                size_t idx = (y * image.width + x) * 3;
                float gray = (image.data[idx] + image.data[idx + 1] + image.data[idx + 2]) / 3.0f;
                float sum = -8 * gray;
                float max_diff = 0.0f;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dy == 0 && dx == 0) continue;
                        size_t n_idx = ((y + dy) * image.width + (x + dx)) * 3;
                        float n_gray = (image.data[n_idx] + image.data[n_idx + 1] + image.data[n_idx + 2]) / 3.0f;
                        float diff = std::abs(n_gray - gray);
                        sum += (diff > noise_threshold) ? n_gray : gray;
                        max_diff = std::max(max_diff, diff);
                    }
                }
                laplacian[y * image.width + x] = std::abs(sum) * (1.0f + std::log1p(max_diff) / 5.0f);
            }
        }
    }

    std::vector<float> frequency_analysis(const Image& image, float freq_factor) const {
        std::vector<float> freq(image.width * image.height);
        float threshold = 10.0f * freq_factor;

        #pragma omp parallel for
        for (size_t i = 0; i < image.width * image.height; ++i) {
            size_t idx = i * 3;
            float r = image.data[idx], g = image.data[idx + 1], b = image.data[idx + 2];
            float mag = std::sqrt(r * r + g * g + b * b);
            freq[i] = mag > threshold ? std::pow(std::log1p(mag), 2.5f) * (mag / 255.0f) : 0;
        }
        return freq;
    }

    std::vector<unsigned char> wavelet_transform(const Image& image, float wave_factor, float detail_preservation) const {
        std::vector<unsigned char> coeffs(image.width * image.height / 4);
        std::vector<float> details(image.width * image.height);
        float scale = wave_factor * calculate_entropy(image);

        #pragma omp parallel for
        for (size_t i = 0; i < image.width * image.height; ++i) {
            details[i] = (image.data[i * 3] + image.data[i * 3 + 1] + image.data[i * 3 + 2]) / 3.0f;
        }

        #pragma omp parallel for collapse(2)
        for (int y = 0; y < image.height; y += 2) {
            for (int x = 0; x < image.width; x += 2) {
                size_t idx = y * image.width + x;
                float avg = (details[idx] + details[idx + 1] +
                            details[idx + image.width] + details[idx + image.width + 1]) / 4.0f;
                float detail = std::max({std::abs(details[idx] - avg), std::abs(details[idx + 1] - avg),
                                        std::abs(details[idx + image.width] - avg),
                                        std::abs(details[idx + image.width + 1] - avg)});
                coeffs[(y / 2) * (image.width / 2) + (x / 2)] =
                    static_cast<unsigned char>(std::min(255.0f, avg + detail * scale * detail_preservation * 1.5f));
            }
        }
        return coeffs;
    }

    double calculate_psnr(const Image& original, const Image& compressed, float contrast_boost) const {
        double mse[3] = {0.0, 0.0, 0.0};
        size_t pixel_count = original.width * original.height;
        double contrast = 0.0;

        #pragma omp parallel for reduction(+:contrast)
        for (size_t i = 0; i < pixel_count; ++i) {
            size_t idx = i * 3;
            float luma = 0.299f * original.data[idx] + 0.587f * original.data[idx + 1] + 0.114f * original.data[idx + 2];
            contrast += std::pow(luma, 1.8f);
        }
        contrast = std::pow(contrast / pixel_count, 1.0 / 1.8) / 255.0 * contrast_boost;

        #pragma omp parallel for
        for (size_t i = 0; i < pixel_count; ++i) {
            size_t idx = i * 3;
            for (int c = 0; c < 3; ++c) {
                int diff = original.data[idx + c] - compressed.data[idx + c];
                float weight = (original.data[idx + c] > 128 ? 2.8f : 1.0f) * (1.0f + contrast);
                mse[c] += diff * diff * weight;
            }
        }

        double total_mse = 0.0;
        for (int c = 0; c < 3; ++c) {
            mse[c] /= pixel_count;
            total_mse += mse[c] * (c == 1 ? 0.5 : 0.25);
        }
        return total_mse == 0 ? 100.0 : 20 * std::log10(255.0 / std::sqrt(total_mse));
    }

    std::vector<unsigned char> adaptive_quantization(const Image& image, float factor, const std::vector<float>& laplacian) const {
        std::vector<unsigned char> quant(image.width * image.height);
        float max_grad = *std::max_element(laplacian.begin(), laplacian.end());

        #pragma omp parallel for
        for (size_t i = 0; i < image.width * image.height; ++i) {
            size_t idx = i * 3;
            float gray = (image.data[idx] + image.data[idx + 1] + image.data[idx + 2]) / 3.0f;
            float level = factor * std::pow(1.0f + laplacian[i] / max_grad, 4.5f) * (gray > 128.0f ? 2.2f : 1.0f);
            quant[i] = static_cast<unsigned char>(std::round(gray / level) * level);
        }
        return quant;
    }

    std::vector<unsigned char> edge_preserving_filter(const Image& image, int radius, float edge_weight, const std::vector<float>& laplacian, float noise_reduction) const {
        std::vector<unsigned char> filtered(image.data.size());
        float max_grad = *std::max_element(laplacian.begin(), laplacian.end());
        float noise_threshold = noise_reduction * 30.0f;

        #pragma omp parallel for collapse(2)
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                size_t idx = (y * image.width + x) * 3;
                float r = 0, g = 0, b = 0, w_sum = 0;
                float center_grad = laplacian[y * image.width + x];
                int adaptive_radius = radius * (1 + static_cast<int>(center_grad / max_grad * 8));
                for (int dy = -adaptive_radius; dy <= adaptive_radius; ++dy) {
                    for (int dx = -adaptive_radius; dx <= adaptive_radius; ++dx) {
                        int ny = std::clamp(y + dy, 0, image.height - 1);
                        int nx = std::clamp(x + dx, 0, image.width - 1);
                        size_t n_idx = (ny * image.width + nx) * 3;
                        float grad_diff = std::abs(center_grad - laplacian[ny * image.width + nx]);
                        if (grad_diff < noise_threshold) continue;
                        float spatial_w = std::exp(-static_cast<float>(dx * dx + dy * dy) / (2.0f * adaptive_radius * adaptive_radius));
                        float range_w = std::exp(-grad_diff / (edge_weight + 0.2f));
                        float w = spatial_w * range_w;
                        r += image.data[n_idx] * w;
                        g += image.data[n_idx + 1] * w;
                        b += image.data[n_idx + 2] * w;
                        w_sum += w;
                    }
                }
                filtered[idx] = static_cast<unsigned char>(std::clamp(r / w_sum, 0.0f, 255.0f));
                filtered[idx + 1] = static_cast<unsigned char>(std::clamp(g / w_sum, 0.0f, 255.0f));
                filtered[idx + 2] = static_cast<unsigned char>(std::clamp(b / w_sum, 0.0f, 255.0f));
            }
        }
        return filtered;
    }

    std::vector<unsigned char> frequency_threshold_compression(const Image& image, float freq_factor) const {
        std::vector<unsigned char> compressed(image.width * image.height);
        std::vector<float> freq = frequency_analysis(image, freq_factor);

        #pragma omp parallel for
        for (size_t i = 0; i < image.width * image.height; ++i) {
            size_t idx = i * 3;
            float gray = (image.data[idx] + image.data[idx + 1] + image.data[idx + 2]) / 3.0f;
            compressed[i] = gray > freq[i] * 255.0f * 1.2f ? 255 : 0;
        }
        return compressed;
    }

    std::vector<unsigned char> local_entropy_compression(const Image& image, float entropy_factor) const {
        std::vector<unsigned char> compressed(image.width * image.height);
        std::vector<float> local_entropy(image.width * image.height);

        #pragma omp parallel for collapse(2)
        for (int y = 1; y < image.height - 1; ++y) {
            for (int x = 1; x < image.width - 1; ++x) {
                int hist[256] = {0};
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        size_t idx = ((y + dy) * image.width + (x + dx)) * 3;
                        hist[(image.data[idx] + image.data[idx + 1] + image.data[idx + 2]) / 3]++;
                    }
                }
                double ent = 0.0;
                for (int i = 0; i < 256; ++i) {
                    if (hist[i] > 0) {
                        double p = hist[i] / 9.0;
                        ent -= p * std::log2(p);
                    }
                }
                local_entropy[y * image.width + x] = ent * entropy_factor * 1.5f;
            }
        }

        #pragma omp parallel for
        for (size_t i = 0; i < image.width * image.height; ++i) {
            size_t idx = i * 3;
            float gray = (image.data[idx] + image.data[idx + 1] + image.data[idx + 2]) / 3.0f;
            compressed[i] = gray > local_entropy[i] * 255.0f / std::log2(256.0) ? 255 : 0;
        }
        return compressed;
    }

    std::vector<unsigned char> custom_compression(const Image& image, int threshold, int filter_size,
                                                float edge_factor, float quant_factor, float entropy_weight,
                                                float texture_factor, float freq_factor, float local_entropy_factor,
                                                float noise_reduction, float contrast_boost, float detail_preservation) const {
        std::vector<float> laplacian(image.width * image.height);
        compute_laplacian(image, laplacian, noise_reduction);
        Image filtered(image.width, image.height);
        filtered.data = edge_preserving_filter(image, filter_size, edge_factor, laplacian, noise_reduction);
        auto quant = adaptive_quantization(filtered, quant_factor, laplacian);
        auto texture = frequency_threshold_compression(filtered, texture_factor);
        auto local_ent = local_entropy_compression(filtered, local_entropy_factor);
        float max_grad = *std::max_element(laplacian.begin(), laplacian.end());
        float entropy = static_cast<float>(calculate_entropy(image));

        std::vector<unsigned char> compressed(image.width * image.height);
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < image.width; ++x) {
                size_t i = y * image.width + x;
                float predicted = 0.0f;
                if (x > 0 && y > 0) {
                    predicted = (quant[i - 1] + quant[i - image.width] + quant[i - image.width - 1]) / 3.0f;
                } else if (x > 0) {
                    predicted = quant[i - 1];
                } else if (y > 0) {
                    predicted = quant[i - image.width];
                }
                float thresh = threshold * (1.0f + edge_factor * laplacian[i] / max_grad + entropy_weight * entropy + freq_factor * contrast_boost * 1.2f);
                compressed[i] = (texture[i] == 255 || local_ent[i] == 255 || (quant[i] - predicted) > thresh) ? 255 : 0;
            }
        }

        std::vector<unsigned char> rle;
        rle.reserve(compressed.size() / 25);
        unsigned char current = compressed[0];
        int count = 1;
        for (size_t i = 1; i < compressed.size(); ++i) {
            if (compressed[i] == current && count < 255) {
                ++count;
            } else {
                rle.push_back(current);
                rle.push_back(static_cast<unsigned char>(count));
                current = compressed[i];
                count = 1;
            }
        }
        rle.push_back(current);
        rle.push_back(static_cast<unsigned char>(count));
        return rle;
    }

    Image custom_decompression(const std::vector<unsigned char>& compressed, int width, int height, float detail_preservation) const {
        std::vector<unsigned char> decompressed(width * height);
        size_t pos = 0;
        for (size_t i = 0; i < compressed.size(); i += 2) {
            std::fill_n(decompressed.begin() + pos, compressed[i + 1], compressed[i]);
            pos += compressed[i + 1];
        }

        Image restored(width, height);
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                size_t i = y * width + x;
                size_t idx = i * 3;
                float predicted = 0.0f;
                int neighbors = 0;
                if (x > 0) { predicted += restored.data[(i - 1) * 3]; neighbors++; }
                if (y > 0) { predicted += restored.data[(i - width) * 3]; neighbors++; }
                if (x < width - 1) { predicted += restored.data[(i + 1) * 3]; neighbors++; }
                if (y < height - 1) { predicted += restored.data[(i + width) * 3]; neighbors++; }
                predicted = neighbors > 0 ? predicted / neighbors : 0;
                float value = decompressed[i] == 255 ? std::min(255.0f, predicted + 120.0f * detail_preservation) : predicted * 0.85f;
                restored.data[idx] = restored.data[idx + 1] = restored.data[idx + 2] = static_cast<unsigned char>(value);
            }
        }
        return restored;
    }

    struct CompressionResult {
        size_t size;
        int threshold, filter_size;
        float edge_factor, quant_factor, entropy_weight, texture_factor, freq_factor, local_entropy_factor;
        float noise_reduction, contrast_boost, detail_preservation;
        std::vector<unsigned char> compressed_data;
        Image restored_image;
        double psnr;
        size_t iteration;
        CompressionResult(int w, int h) : size(std::numeric_limits<size_t>::max()), threshold(0), filter_size(0),
                                        edge_factor(0), quant_factor(0), entropy_weight(0), texture_factor(0),
                                        freq_factor(0), local_entropy_factor(0), noise_reduction(0), contrast_boost(0),
                                        detail_preservation(0), restored_image(w, h), psnr(0), iteration(0) {}
    };

    void save_state(const CompressionResult& result) const {
        std::ofstream file("optimizer_state.txt");
        if (file.is_open()) {
            file << "# Compressed Size (bytes)\n" << result.size << "\n"
                 << "# Threshold (base quantization threshold, 8-192)\n" << result.threshold << "\n"
                 << "# Filter Size (edge preserving filter radius, 0-7)\n" << result.filter_size << "\n"
                 << "# Edge Factor (edge enhancement weight, 0-3)\n" << result.edge_factor << "\n"
                 << "# Quant Factor (quantization factor, 1-5)\n" << result.quant_factor << "\n"
                 << "# Entropy Weight (entropy influence factor, 0-0.5)\n" << result.entropy_weight << "\n"
                 << "# Texture Factor (frequency threshold factor, 0.5-2)\n" << result.texture_factor << "\n"
                 << "# Frequency Factor (frequency analysis factor, 0.5-2)\n" << result.freq_factor << "\n"
                 << "# Local Entropy Factor (local entropy compression factor, 0.5-2)\n" << result.local_entropy_factor << "\n"
                 << "# Noise Reduction (noise suppression factor, 0-1)\n" << result.noise_reduction << "\n"
                 << "# Contrast Boost (contrast enhancement factor, 0.5-2)\n" << result.contrast_boost << "\n"
                 << "# Detail Preservation (detail retention factor, 0.5-2)\n" << result.detail_preservation << "\n"
                 << "# PSNR (peak signal-to-noise ratio)\n" << result.psnr << "\n"
                 << "# Iteration (current iteration number)\n" << result.iteration << "\n";
            file.close();
        } else {
            std::cerr << "Failed to save state to optimizer_state.txt\n";
        }
    }

    CompressionResult load_state(int width, int height) const {
        CompressionResult result(width, height);
        std::ifstream file("optimizer_state.txt");
        if (file.is_open()) {
            std::string line;
            std::getline(file, line); file >> result.size;
            std::getline(file, line); std::getline(file, line); file >> result.threshold;
            std::getline(file, line); std::getline(file, line); file >> result.filter_size;
            std::getline(file, line); std::getline(file, line); file >> result.edge_factor;
            std::getline(file, line); std::getline(file, line); file >> result.quant_factor;
            std::getline(file, line); std::getline(file, line); file >> result.entropy_weight;
            std::getline(file, line); std::getline(file, line); file >> result.texture_factor;
            std::getline(file, line); std::getline(file, line); file >> result.freq_factor;
            std::getline(file, line); std::getline(file, line); file >> result.local_entropy_factor;
            std::getline(file, line); std::getline(file, line); file >> result.noise_reduction;
            std::getline(file, line); std::getline(file, line); file >> result.contrast_boost;
            std::getline(file, line); std::getline(file, line); file >> result.detail_preservation;
            std::getline(file, line); std::getline(file, line); file >> result.psnr;
            std::getline(file, line); std::getline(file, line); file >> result.iteration;
            file.close();
        }
        return result;
    }

    void log_attempt(int t, int f, float e, float q, float ew, float tf, float ff, float lef, float nr, float cb, float dp, size_t size, double psnr) const {
        std::ofstream file("optimizer_log.txt", std::ios::app);
        if (file.is_open()) {
            file << "T=" << t << " F=" << f << " E=" << e << " Q=" << q << " EW=" << ew << " TF=" << tf
                 << " FF=" << ff << " LEF=" << lef << " NR=" << nr << " CB=" << cb << " DP=" << dp
                 << " Size=" << size << " PSNR=" << psnr << "\n";
            file.close();
        }
    }

    void optimize_thread(const Image& image, CompressionResult& best_result, size_t& iteration_counter, size_t iterations_per_thread, size_t total_iterations) {
        std::random_device rd;
        std::mt19937 gen(rd());

        for (size_t i = 0; i < iterations_per_thread && iteration_counter < total_iterations; ++i) {
            int t = static_cast<int>(learning_state.threshold.sample(gen));
            int f = static_cast<int>(learning_state.filter_size.sample(gen));
            float e = learning_state.edge_factor.sample(gen);
            float q = learning_state.quant_factor.sample(gen);
            float ew = learning_state.entropy_weight.sample(gen);
            float tf = learning_state.texture_factor.sample(gen);
            float ff = learning_state.freq_factor.sample(gen);
            float lef = learning_state.local_entropy_factor.sample(gen);
            float nr = learning_state.noise_reduction.sample(gen);
            float cb = learning_state.contrast_boost.sample(gen);
            float dp = learning_state.detail_preservation.sample(gen);

            auto compressed = custom_compression(image, t, f, e, q, ew, tf, ff, lef, nr, cb, dp);
            Image restored = custom_decompression(compressed, image.width, image.height, dp);
            double psnr = calculate_psnr(image, restored, cb);
            size_t size = compressed.size();

            {
                std::lock_guard<std::mutex> lock(mtx);
                std::cout << "Attempt " << iteration_counter + 1 << "/" << total_iterations << ": T=" << t << " F=" << f << " E=" << e
                          << " Q=" << q << " EW=" << ew << " TF=" << tf << " FF=" << ff << " LEF=" << lef
                          << " NR=" << nr << " CB=" << cb << " DP=" << dp << " Size=" << size << " PSNR=" << psnr << "\n";
                log_attempt(t, f, e, q, ew, tf, ff, lef, nr, cb, dp, size, psnr);

                double base_psnr = best_result.psnr > 0 ? best_result.psnr : 30.0;
                float size_ratio = static_cast<float>(image.data.size()) / size; // Вычисляем степень сжатия

                // Обновляем обучение с учетом size_ratio
                learning_state.threshold.update(t, psnr, base_psnr, size_ratio);
                learning_state.filter_size.update(f, psnr, base_psnr, size_ratio);
                learning_state.edge_factor.update(e, psnr, base_psnr, size_ratio);
                learning_state.quant_factor.update(q, psnr, base_psnr, size_ratio);
                learning_state.entropy_weight.update(ew, psnr, base_psnr, size_ratio);
                learning_state.texture_factor.update(tf, psnr, base_psnr, size_ratio);
                learning_state.freq_factor.update(ff, psnr, base_psnr, size_ratio);
                learning_state.local_entropy_factor.update(lef, psnr, base_psnr, size_ratio);
                learning_state.noise_reduction.update(nr, psnr, base_psnr, size_ratio);
                learning_state.contrast_boost.update(cb, psnr, base_psnr, size_ratio);
                learning_state.detail_preservation.update(dp, psnr, base_psnr, size_ratio);

                if (size < best_result.size && psnr >= 30.0) {
                    best_result.size = size;
                    best_result.threshold = t;
                    best_result.filter_size = f;
                    best_result.edge_factor = e;
                    best_result.quant_factor = q;
                    best_result.entropy_weight = ew;
                    best_result.texture_factor = tf;
                    best_result.freq_factor = ff;
                    best_result.local_entropy_factor = lef;
                    best_result.noise_reduction = nr;
                    best_result.contrast_boost = cb;
                    best_result.detail_preservation = dp;
                    best_result.compressed_data = std::move(compressed);
                    best_result.restored_image = std::move(restored);
                    best_result.psnr = psnr;
                    best_result.iteration = iteration_counter;
                    save_state(best_result);
                    std::cout << "New best: Size=" << size << " PSNR=" << psnr << " Iteration=" << iteration_counter << "\n";
                }
                iteration_counter++;
            }
        }
    }

    CompressionResult find_best_parameters(const Image& image, size_t total_iterations) {
        learning_state.load();
        CompressionResult best_result = load_state(image.width, image.height);
        size_t original_size = image.data.size();
        const int num_threads = std::thread::hardware_concurrency();
        size_t iterations_per_thread = total_iterations / num_threads;
        size_t iteration_counter = 0;

        std::cout << "Starting optimization...\n";
        std::cout << "Original size: " << original_size << " bytes\n";
        if (best_result.size != std::numeric_limits<size_t>::max()) {
            std::cout << "Loaded best: Size=" << best_result.size << " T=" << best_result.threshold
                      << " F=" << best_result.filter_size << " E=" << best_result.edge_factor
                      << " Q=" << best_result.quant_factor << " EW=" << best_result.entropy_weight
                      << " TF=" << best_result.texture_factor << " FF=" << best_result.freq_factor
                      << " LEF=" << best_result.local_entropy_factor << " NR=" << best_result.noise_reduction
                      << " CB=" << best_result.contrast_boost << " DP=" << best_result.detail_preservation
                      << " PSNR=" << best_result.psnr << " Iteration=" << best_result.iteration << "\n";
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i) {
            size_t iters = (i == num_threads - 1) ? (total_iterations - iteration_counter) : iterations_per_thread;
            threads.emplace_back(&ImageOptimizer::optimize_thread, this, std::ref(image), std::ref(best_result),
                                 std::ref(iteration_counter), iters, total_iterations);
        }

        for (auto& th : threads) th.join();
        learning_state.save();

        std::cout << "Optimization completed at iteration " << iteration_counter << "\n";
        if (best_result.size != std::numeric_limits<size_t>::max()) {
            double ratio = static_cast<double>(original_size) / best_result.size;
            std::cout << "Final best result: Size=" << best_result.size << " bytes, "
                      << "Ratio=" << ratio << "x, T=" << best_result.threshold
                      << " F=" << best_result.filter_size << " E=" << best_result.edge_factor
                      << " Q=" << best_result.quant_factor << " EW=" << best_result.entropy_weight
                      << " TF=" << best_result.texture_factor << " FF=" << best_result.freq_factor
                      << " LEF=" << best_result.local_entropy_factor << " NR=" << best_result.noise_reduction
                      << " CB=" << best_result.contrast_boost << " DP=" << best_result.detail_preservation
                      << " PSNR=" << best_result.psnr << " Iteration=" << best_result.iteration << "\n";
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - start_time).count();
        std::cout << "Total time: " << elapsed << "s\n";
        return best_result;
    }

    Image load_image(const std::string& path) {
        int width, height, channels;
        unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
        if (!data) {
            std::cerr << "Failed to load image: " << path << "\n";
            return Image(0, 0);
        }
        Image img(width, height, data);
        stbi_image_free(data);
        return img;
    }

    Image create_test_image() const {
        Image img(200, 200);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> noise(0, 40);
        std::uniform_int_distribution<> object_pos(20, 180);
        std::uniform_real_distribution<float> texture(0.5f, 1.5f);

        #pragma omp parallel for collapse(2)
        for (int y = 0; y < img.height; ++y) {
            for (int x = 0; x < img.width; ++x) {
                size_t idx = (y * img.width + x) * 3;
                float t = texture(gen);
                float grad = std::sin(x * 0.1f) * std::cos(y * 0.1f);
                img.data[idx] = std::min(255, static_cast<int>(t * (grad + 1.0f) * 127.5f + noise(gen)));
                img.data[idx + 1] = std::min(255, static_cast<int>(t * (grad + 1.0f) * 127.5f + noise(gen)));
                img.data[idx + 2] = std::min(255, static_cast<int>(t * 128 + noise(gen)));
            }
        }

        int obj_x = object_pos(gen), obj_y = object_pos(gen);
        for (int y = obj_y; y < std::min(obj_y + 40, img.height); ++y) {
            for (int x = obj_x; x < std::min(obj_x + 40, img.width); ++x) {
                size_t idx = (y * img.width + x) * 3;
                img.data[idx] = 255;
                img.data[idx + 1] = std::min(255, static_cast<int>(img.data[idx + 1] * 1.8f));
                img.data[idx + 2] = std::max(0, static_cast<int>(img.data[idx + 2] - 60));
            }
        }
        return img;
    }

public:
    ImageOptimizer() {
        learning_state.load();
    }

    void run_on_test_image(size_t iterations) {
        Image image = create_test_image();
        if (image.width == 0) return;
        find_best_parameters(image, iterations);
    }

    void run_on_file(const std::string& path, size_t iterations) {
        Image image = load_image(path);
        if (image.width == 0) return;
        find_best_parameters(image, iterations);
    }
};

#endif