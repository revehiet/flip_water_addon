#pragma once
#include <vector>
#include <cstring>

namespace flipcore {

// Simple dense 3D array, column-major-ish flat storage: index = i + nx*(j + ny*k)
template <typename T>
class Array3 {
public:
    Array3() = default;
    Array3(int nx, int ny, int nz, T fill = T())
        : nx_(nx), ny_(ny), nz_(nz), data_(size_t(nx) * ny * nz, fill) {}

    void resize(int nx, int ny, int nz, T fill = T()) {
        nx_ = nx; ny_ = ny; nz_ = nz;
        data_.assign(size_t(nx) * ny * nz, fill);
    }

    void fill(T v) { std::fill(data_.begin(), data_.end(), v); }

    inline int nx() const { return nx_; }
    inline int ny() const { return ny_; }
    inline int nz() const { return nz_; }
    inline size_t size() const { return data_.size(); }

    inline bool inBounds(int i, int j, int k) const {
        return i >= 0 && i < nx_ && j >= 0 && j < ny_ && k >= 0 && k < nz_;
    }

    inline size_t idx(int i, int j, int k) const {
        return size_t(i) + size_t(nx_) * (size_t(j) + size_t(ny_) * size_t(k));
    }

    inline T& operator()(int i, int j, int k) { return data_[idx(i, j, k)]; }
    inline const T& operator()(int i, int j, int k) const { return data_[idx(i, j, k)]; }

    inline T at(int i, int j, int k, T outside) const {
        if (!inBounds(i, j, k)) return outside;
        return data_[idx(i, j, k)];
    }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }

private:
    int nx_ = 0, ny_ = 0, nz_ = 0;
    std::vector<T> data_;
};

} // namespace flipcore
