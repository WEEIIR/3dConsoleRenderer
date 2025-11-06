/*
 * BİRLEŞTİRİLMİŞ 3D KONSOL UYGULAMASI
 * * Bu dosya, aşağıdaki dosyaların içeriğini birleştirir:
 * - load.hpp
 * - mathh.hpp
 * - 3dConsole.cpp
 *
 * Derlemek için 'json.hpp' dosyasının aynı dizinde olması gerekir.
 */

 // --- Başlangıç: Gerekli Include Dosyaları ---
#include <windows.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <immintrin.h>
#include <codecvt>
#include <locale>
#include <string>
#include <chrono>
#include <thread>
#include <cwchar>
#include <cmath>
#include <array> 
#include <algorithm>
#include <omp.h> // paralel for

// json.hpp hariç tutuldu, ancak projenin çalışması için gereklidir
#include "json.hpp" 
// --- Bitiş: Gerekli Include Dosyaları ---


// --- Başlangıç: load.hpp içeriği ---

using json = nlohmann::json;
using namespace std;

string wstring_to_utf8(const wstring& wstr) {
    wstring_convert<codecvt_utf8<wchar_t>> conv;
    return conv.to_bytes(wstr);
}

wstring utf8_to_wstring(const string& str) {
    wstring_convert<codecvt_utf8<wchar_t>> conv;
    return conv.from_bytes(str);
}

void load_json_to_m256_w(
    const wstring& filename,
    vector<wstring>& keys,
    vector<__m256>& vectors)
{
    ifstream file(wstring_to_utf8(filename));
    if (!file.is_open()) {
        wcerr << L"Dosya açılamadı: " << filename << L"\n";
        return;
    }

    json j;
    file >> j;

    for (auto& el : j.items()) {
        keys.push_back(utf8_to_wstring(el.key()));

        vector<float> tmp = el.value().get<vector<float>>();

        // DÜZELTME: JSON'da 18 float var, 8'den az olup olmadığını kontrol et
        if (tmp.size() < 8) continue; // 8'den azsa atla

        // Sadece ilk 8 float'u al
        __m256 vec = _mm256_set_ps(
            tmp[7], tmp[6], tmp[5], tmp[4],
            tmp[3], tmp[2], tmp[1], tmp[0]
        );
        vectors.push_back(vec);
    }
}

// __m256 farkın karelerini toplayarak dotProductItSelf yap
float distanceSquare(__m256 a, __m256 b) {
    __m256 diff = _mm256_sub_ps(a, b);      // a - b
    __m256 sq = _mm256_mul_ps(diff, diff);  // (a-b)^2

    // Toplamını hesapla
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, sq);
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) sum += tmp[i];
    return sum;
}

int findNearestVector(const vector<__m256>& vectors, __m256 myVector) {
    float minValue = 99;
    int minIndex = -1;

    for (size_t i = 0; i < vectors.size(); i++) {
        float val = distanceSquare(myVector, vectors[i]);
        if (val < minValue) {
            minValue = val;
            minIndex = static_cast<int>(i);
        }
    }
    return minIndex;
}

#pragma pack(push, 1)
struct Vertex { float x, y, z; };
#pragma pack(pop)

#pragma pack(push, 1)
struct Edge { int v0, v1; };
#pragma pack(pop)

struct Mesh {
    vector<Vertex> vertices;
    vector<Edge> edges;
};

Mesh loadMeshFromJSON(const string& filename) {
    Mesh mesh;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Dosya açılamadı: " << filename << "\n";
        return mesh;
    }

    json j;
    file >> j;

    for (auto& v : j["vertices"]) {
        if (v.size() != 3) continue; // 3 float kontrolü
        mesh.vertices.push_back({ v[0].get<float>(), v[1].get<float>(), v[2].get<float>() });
    }

    for (auto& e : j["edges"]) {
        if (e.size() != 2) continue; // 2 int kontrolü
        mesh.edges.push_back({ e[0].get<int>(), e[1].get<int>() });
    }

    return mesh;
}


vector<vector<__m256>> ConvertToM256_Optimized(const vector<vector<float>>& pxBuffer) {
    size_t h_px = pxBuffer.size();    // Bu height * 4
    if (h_px == 0) return {};
    size_t w_px = pxBuffer[0].size(); // Bu width * 2

    size_t h_char = h_px / 4; // Konsol yüksekliği
    size_t w_char = w_px / 2; // Konsol genişliği

    vector<vector<__m256>> result(h_char, vector<__m256>(w_char));

    for (size_t y_char = 0; y_char < h_char; ++y_char) {
        for (size_t x_char = 0; x_char < w_char; ++x_char) {

            size_t y_px_start = y_char * 4;
            size_t x_px_start = x_char * 2;

            alignas(32) float block[8];

            // 4x2'lik piksel bloğunu 8'lik bir float dizisine doldur
            for (int dy = 0; dy < 4; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    // Sınır kontrolü
                    if (y_px_start + dy < h_px && x_px_start + dx < w_px) {
                        block[dy * 2 + dx] = pxBuffer[y_px_start + dy][x_px_start + dx];
                    }
                    else {
                        block[dy * 2 + dx] = 0.0f; // Sınırların dışı siyah
                    }
                }
            }

            // 8 float'u __m256 vektörüne yükle
            result[y_char][x_char] = _mm256_load_ps(block);
        }
    }
    return result;
}

// --- Bitiş: load.hpp içeriği ---


// --- Başlangıç: mathh.hpp içeriği ---

struct MeshTransform {
    float pos[3];
    float scale[3];
    float rotate[3];
};

MeshTransform generateMeshTransform(const float pos[3], const float scale[3], const float rotate[3]) {
    return {
        {pos[0], pos[1], pos[2]},
        {scale[0], scale[1], scale[2]},
        {rotate[0], rotate[1], rotate[2]}
    };
}

// 3x3 rotation matrisini oluştur (Rz*Ry*Rx)  
array<array<float, 3>, 3> getRotationMatrix(const float r[3]) {
    float cx = cos(r[0]), sx = sin(r[0]);
    float cy = cos(r[1]), sy = sin(r[1]);
    float cz = cos(r[2]), sz = sin(r[2]);

    return { {
        {cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy},
        {cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx},
        {-sy,   cy * sx,             cx * cy}
    } };
}

vector<Vertex> applyTransform(const vector<Vertex>& vertices, const MeshTransform& t) {
    auto R = getRotationMatrix(t.rotate);
    vector<Vertex> newMesh;
    newMesh.reserve(vertices.size());

    for (const auto& v : vertices) {
        // Scale  
        float sx = v.x * t.scale[0];
        float sy = v.y * t.scale[1];
        float sz = v.z * t.scale[2];

        // Rotate (matris çarpımı)  
        Vertex temp;
        temp.x = R[0][0] * sx + R[0][1] * sy + R[0][2] * sz;
        temp.y = R[1][0] * sx + R[1][1] * sy + R[1][2] * sz;
        temp.z = R[2][0] * sx + R[2][1] * sy + R[2][2] * sz;

        // Translate  
        temp.x += t.pos[0];
        temp.y += t.pos[1];
        temp.z += t.pos[2];

        newMesh.push_back(temp);
    }
    return newMesh;
}

inline float removeZforProject2d(float m, float z, float v, float s) {
    if (m == 0) return 0;
    // Kameranın *içinde* veya *arkasında* olan noktaları kırp
    if (z <= 0.1f) return s / 2.0f; // Basitçe merkeze ata

    // Projeksiyonu ekranın merkezine göre yap
    return (s * v / (m * z)) + (s / 2.0f);
}

vector<Vertex> project3dTo2d(const vector<Vertex>& vertices, float mX, float mY, float screenW, float screenH) {
    vector<Vertex> projected;
    projected.reserve(vertices.size());

    for (auto& v : vertices) {
        float x2d = removeZforProject2d(mX, v.z, v.x, screenW);
        float y2d = removeZforProject2d(mY, v.z, v.y, screenH);
        projected.push_back({ x2d, y2d, v.z });
    }
    return projected;
}


// DDA (Digital Differential Analyzer) tabanlı çizgi çizme
vector<vector<float>> PxBufferEdit(
    int tickness,
    int width,
    int height,
    const vector<Vertex>& vertices,
    const vector<Edge>& edges)
{
    vector<vector<float>> buf(height, vector<float>(width, 0.0f));

    for (auto& e : edges) {
        const Vertex& a = vertices[e.v0];
        const Vertex& b = vertices[e.v1];

        float dX = b.x - a.x;
        float dY = b.y - a.y;

        float steps = max(fabs(dX), fabs(dY));

        if (steps < 1e-5) continue;

        float xInc = dX / steps;
        float yInc = dY / steps;
        float tempX = a.x;
        float tempY = a.y;

        for (int k = 0; k <= static_cast<int>(steps); k++)
        {
            int px = static_cast<int>(tempX);
            int py = static_cast<int>(tempY);

            if (px >= 0 && px < width && py >= 0 && py < height)
            {
                float fx = tempX - floor(tempX);
                float fy = tempY - floor(tempY);

                float density = (1.0f - fabs(0.5f - fx) * 2.0f) * (1.0f - fabs(0.5f - fy) * 2.0f)+k;

                density = 1.0f - pow(1.0f - density, (float)tickness);
                density = max(0.0f, density); // Negatif yoğunluk olmasın

                buf[py][px] = min(1.0f, buf[py][px] + density);
            }

            tempX += xInc;
            tempY += yInc;
        }
    }

    return buf;
}

// --- Bitiş: mathh.hpp içeriği ---


// --- Başlangıç: 3dConsole.cpp içeriği ---

size_t width, height;
string meshPath;
Mesh mesh;

vector<wstring> Elements;
vector<__m256> ElementDensities;

// -> geniş karakter tamponu
wstring frameBuffer;
size_t frameBufferSize;

void print_screen(wchar_t c) {
    frameBuffer.clear();

    for (size_t i = 1; i <= height; i++) {
        for (size_t j = 1; j <= width; j++)
            frameBuffer += c;
        frameBuffer += L"\n";
    }
}

void initalize(int param);

float standartZ = 5.0f;

void initalize(int param) {
    if (param == 0) {
        cout << "Distance?";
        cin >> standartZ;
        cout << "Width? ";
        cin >> width;
        cout << "Height? ";
        cin >> height;

        print_screen('/');
        frameBufferSize = (width + 1) * height * sizeof(wchar_t);

        wcout << frameBuffer;
        char temp;
        cout << "\n retry for 0, continue for anykey\n";
        cin >> temp;
        if (temp == '0') {
            system("cls");
            initalize(0);
            return;
        }
    }
    cout << "Mash.json? ";
    cin >> meshPath;

    load_json_to_m256_w(L"ascii_1x8.json", Elements, ElementDensities);
    mesh = loadMeshFromJSON(meshPath);
    return;
}

MeshTransform MyTransform;
float fovY = 3.141592f/2; // 90 derece (PI/2)
float fovX;

float ratioX;
float ratioY;

constexpr int maxHistory = 50;
vector<float> fpsHistory(maxHistory, 0.0f);
int fpsIndex = 0, fpsCount = 0;
float fpsSum = 0.0f, avgFPS = 0.0f;

inline void updateFPS(float currentFPS) {
    fpsSum -= fpsHistory[fpsIndex];
    fpsHistory[fpsIndex] = currentFPS;
    fpsSum += currentFPS;
    fpsIndex = (fpsIndex + 1) % maxHistory;
    if (fpsCount < maxHistory) fpsCount++;
    avgFPS = fpsSum / fpsCount;
}

// ---------- AVX + Paralel frameBuffer ----------

inline float horizontal_sum(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

void DensityMapToFramebufferAVX(
    wstring& frameBuffer,
    const vector<vector<__m256>>& DensityBuffer, // Boyut: (height, width)
    const vector<wstring>& Elements,
    const vector<__m256>& ElementDensities)
{
    frameBuffer.clear();
    frameBuffer.reserve(frameBufferSize);

    size_t h_char = DensityBuffer.size(); // Konsol yüksekliği
    if (h_char == 0) return;
    size_t w_char = DensityBuffer[0].size(); // Konsol genişliği

    vector<wstring> localRows(h_char);

#pragma omp parallel for
    for (size_t i = 0; i < h_char; i++) { // Konsol yüksekliği kadar döner
        wstring line;
        line.reserve(w_char + 1);

        for (size_t j = 0; j < w_char; j++) { // Konsol genişliği kadar döner

            // (i, j)'deki karakter, DensityBuffer[i][j]'deki 4x2'lik bloğa karşılık gelir.
            __m256 dens = DensityBuffer[i][j];

            int bestIdx = -1;
            float bestDiff = 1e10f;

            // En yakın karakteri bul
            for (size_t k = 0; k < ElementDensities.size(); k++) {
                __m256 diff = _mm256_sub_ps(dens, ElementDensities[k]);
                diff = _mm256_mul_ps(diff, diff);
                float totalDiff = horizontal_sum(diff);
                if (totalDiff < bestDiff) {
                    bestDiff = totalDiff;
                    bestIdx = (int)k;
                }
            }

            line += (bestIdx >= 0 && bestIdx < (int)Elements.size()) ? Elements[bestIdx][0] : L' ';
        }

        line += L'\n';
        localRows[i] = line;
    }

    frameBuffer.clear();
    for (const auto& row : localRows)
        frameBuffer += row;
}

// ---------------------------------------------

int main() {
    initalize(0);

    fovX = 2 * atan(tan(fovY / 2) * (float(width) / float(height)));
    ratioX = tan(fovY / 2);
    ratioY = tan(fovX / 2);

    float position[3] = { 0.0f, 0.0f, standartZ };
    float scale[3] = { 1.0f, 1.0f, 1.0f };
    float rotation[3] = { 1.0f, 0.0f, 0.0f };

    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    COORD topLeft = { 0, 0 };

    using clock = chrono::high_resolution_clock;
    auto lastTime = clock::now();

    Mesh dynamicMesh = Mesh(mesh);
    vector<vector<float>> pxBuffer;
    system("cls");

    float rot = 0.0f;

    while (true) {
        auto currentTime = clock::now();
        chrono::duration<float> delta = currentTime - lastTime;
        lastTime = currentTime;
        float fps = 1.0f / delta.count();

        rot += 0.5f * delta.count();
        rotation[1] = rot;

        // ---------------- İşlem ----------------

        MyTransform = generateMeshTransform(position, scale, rotation);
        // Her zaman orijinal mesh'i dönüştür
        dynamicMesh.vertices = applyTransform(mesh.vertices, MyTransform);

        // Piksel tamponunun boyutları (yükseklik*4, genişlik*2)
        vector<Vertex> dynamicMeshV_2d = project3dTo2d(dynamicMesh.vertices, ratioY, ratioX, (float)width * 2, (float)height * 4);

        pxBuffer = PxBufferEdit(1, width * 2, height * 4, dynamicMeshV_2d, dynamicMesh.edges);

        // pxBuffer'ı (height, width) boyutunda __m256 vektörlerine dönüştür
        vector<vector<__m256>> DensityBuffer = ConvertToM256_Optimized(pxBuffer);

        // (height, width) __m256 buffer'ını kullanarak framebuffer'ı oluştur
        DensityMapToFramebufferAVX(frameBuffer, DensityBuffer, Elements, ElementDensities);
        // ---------------------------------------

        SetConsoleCursorPosition(hConsole, topLeft);
        WriteConsoleW(hConsole, frameBuffer.c_str(), (DWORD)frameBuffer.size(), nullptr, nullptr);

        //this_thread::sleep_for(chrono::milliseconds(200));

        updateFPS(fps);
        wcout << L"AvarageFPS(last 50): " << avgFPS;
        cout.flush();
    }
}
// --- Bitiş: 3dConsole.cpp içeriği ---