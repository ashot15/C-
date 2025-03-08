#include <iostream>
#include <string>
#include "optimizer.h"

#ifdef _WIN32
#include <windows.h>
#endif

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
#endif

    std::cout << "Super Advanced Image Optimizer\n";
    std::cout << "Enter number of iterations (e.g., 100):\n> ";
    std::string iter_input;
    std::getline(std::cin, iter_input);
    size_t iterations = iter_input.empty() ? 100 : std::stoul(iter_input);

    std::cout << "Enter image path (or press Enter for test image):\n> ";
    std::string path;
    std::getline(std::cin, path);

    ImageOptimizer optimizer;
    if (path.empty()) {
        std::cout << "Generating test image...\n";
        optimizer.run_on_test_image(iterations);
    } else {
        std::cout << "Loading image from " << path << "...\n";
        optimizer.run_on_file(path, iterations);
    }

    std::cout << "\nPress Enter to exit...\n";
    std::cin.get();
    return 0;
}