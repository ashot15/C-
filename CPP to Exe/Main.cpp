#include <windows.h>
#include <string>
#include <vector>

#pragma comment(linker, "/SUBSYSTEM:WINDOWS")
#pragma comment(lib, "gdi32.lib") // Для MSVC, для g++ используйте -lgdi32

std::string sourceFile, outputFile;
std::vector<bool> flagsSelected = {false, false, false, false, false, false, false, false, false, false};
const char* flags[] = {"-O0", "-O1", "-O2", "-O3", "-g", "-Wall", "-Wextra", "-pedantic", "-std=c++17", "-static"};

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    static HWND hEditSource, hEditOutput, hCompileBtn, hClearBtn;
    static HWND hCheckFlags[10], hGroupBox;
    static HFONT hFont;

    switch (msg) {
        case WM_CREATE: {
            hFont = CreateFont(18, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_SWISS, "Arial");

            HWND hTitle = CreateWindow("STATIC", "C++ Compiler Pro", WS_VISIBLE | WS_CHILD, 20, 10, 250, 30, hwnd, nullptr, nullptr, nullptr);
            SendMessage(hTitle, WM_SETFONT, reinterpret_cast<WPARAM>(hFont), TRUE);

            CreateWindow("STATIC", "Source .cpp (type path):", WS_VISIBLE | WS_CHILD, 20, 50, 200, 20, hwnd, nullptr, nullptr, nullptr);
            hEditSource = CreateWindow("EDIT", "", WS_VISIBLE | WS_CHILD | WS_BORDER | ES_AUTOHSCROLL, 230, 50, 450, 30, hwnd, nullptr, nullptr, nullptr);

            CreateWindow("STATIC", "Output .exe:", WS_VISIBLE | WS_CHILD, 20, 100, 200, 20, hwnd, nullptr, nullptr, nullptr);
            hEditOutput = CreateWindow("EDIT", "", WS_VISIBLE | WS_CHILD | WS_BORDER | ES_AUTOHSCROLL, 230, 100, 450, 30, hwnd, nullptr, nullptr, nullptr);

            hGroupBox = CreateWindow("BUTTON", "Compiler Flags", WS_VISIBLE | WS_CHILD | BS_GROUPBOX, 20, 150, 660, 250, hwnd, nullptr, nullptr, nullptr);
            SendMessage(hGroupBox, WM_SETFONT, reinterpret_cast<WPARAM>(hFont), TRUE);

            for (int i = 0; i < 10; ++i) {
                hCheckFlags[i] = CreateWindow("BUTTON", flags[i], WS_VISIBLE | WS_CHILD | BS_CHECKBOX, 40 + (i % 2) * 300, 180 + (i / 2) * 40, 250, 30, hwnd, reinterpret_cast<HMENU>(static_cast<intptr_t>(10 + i)), nullptr, nullptr);
                SendMessage(hCheckFlags[i], WM_SETFONT, reinterpret_cast<WPARAM>(hFont), TRUE);
            }

            hCompileBtn = CreateWindow("BUTTON", "Compile", WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON, 480, 430, 100, 40, hwnd, reinterpret_cast<HMENU>(2), nullptr, nullptr);
            hClearBtn = CreateWindow("BUTTON", "Clear", WS_VISIBLE | WS_CHILD, 590, 430, 100, 40, hwnd, reinterpret_cast<HMENU>(3), nullptr, nullptr);
            SendMessage(hCompileBtn, WM_SETFONT, reinterpret_cast<WPARAM>(hFont), TRUE);
            SendMessage(hClearBtn, WM_SETFONT, reinterpret_cast<WPARAM>(hFont), TRUE);
            EnableWindow(hCompileBtn, TRUE);
            break;
        }
        case WM_CTLCOLORSTATIC: {
            HDC hdc = reinterpret_cast<HDC>(wParam);
            SetTextColor(hdc, RGB(0, 0, 255));
            SetBkColor(hdc, RGB(240, 240, 240));
            return reinterpret_cast<LRESULT>(CreateSolidBrush(RGB(240, 240, 240)));
        }
        case WM_COMMAND: {
            if (HIWORD(wParam) == BN_CLICKED) {
                if (LOWORD(wParam) >= 10 && LOWORD(wParam) < 20) {
                    int index = LOWORD(wParam) - 10;
                    flagsSelected[index] = !flagsSelected[index];
                    SendMessage(hCheckFlags[index], BM_SETCHECK, flagsSelected[index] ? BST_CHECKED : BST_UNCHECKED, 0);
                }
                else if (LOWORD(wParam) == 2) {
                    char sourceBuffer[260], outputBuffer[260];
                    GetWindowText(hEditSource, sourceBuffer, 260);
                    GetWindowText(hEditOutput, outputBuffer, 260);
                    sourceFile = sourceBuffer;
                    outputFile = outputBuffer;

                    std::string command = "g++ \"" + sourceFile + "\" -o \"" + outputFile + "\"";
                    for (int i = 0; i < 10; ++i) {
                        if (flagsSelected[i]) command += " " + std::string(flags[i]);
                    }

                    int result = system(command.c_str());
                    MessageBox(hwnd, result == 0 ? "Compilation successful!" : "Compilation failed!", "Result", MB_OK | (result == 0 ? MB_ICONINFORMATION : MB_ICONERROR));
                }
                else if (LOWORD(wParam) == 3) {
                    SetWindowText(hEditSource, "");
                    SetWindowText(hEditOutput, "");
                    for (int i = 0; i < 10; ++i) {
                        flagsSelected[i] = false;
                        SendMessage(hCheckFlags[i], BM_SETCHECK, BST_UNCHECKED, 0);
                    }
                }
            }
            break;
        }
        case WM_DESTROY:
            DeleteObject(hFont);
            PostQuitMessage(0);
            break;
        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
    }
    return 0;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow) {
    WNDCLASS wc = {0};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = "CompilerWindow";
    RegisterClass(&wc);

    HWND hwnd = CreateWindow("CompilerWindow", "C++ Compiler Pro", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 700, 500, nullptr, nullptr, hInstance, nullptr);
    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);

    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return 0;
}
