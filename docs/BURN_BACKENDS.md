# Burn 0.20 Backends

This document lists all backends available in Burn 0.20 and how to use them.

## Current Status

⚠️ **Backend selection is partially implemented.**

- ✅ Backend feature flags are configured in Cargo.toml
- ✅ CLI argument `--backend` is accepted
- ⚠️ **Actual device creation is not yet implemented** - all paths currently use default device
- 📋 **TODO**: Implement proper device creation for each backend

### What's Working Now

All training currently runs on CPU (NdArray) regardless of --backend flag.
To use actual GPU backends, additional implementation is needed in train.rs.

## 1. NdArray Backend

- **Name**: NdArray
- **Backend type**: CPU
- **Enable in Cargo.toml**:
  ```toml
  [dependencies]
  burn = { version = "0.20", features = ["ndarray"] }
  ```
- **Example use**:
  ```rust
  use burn::backend::NdArray;

  type Backend = NdArray;
  ```

## 2. Wgpu Backend

- **Name**: Wgpu
- **Backend type**: GPU (cross-platform)
- **Enable in Cargo.toml**:
  ```toml
  [dependencies]
  burn = { version = "0.20", features = ["wgpu"] }
  ```
- **Example use**:
  ```rust
  use burn::backend::Wgpu;

  type Backend = Wgpu;
  ```

## 3. Cuda Backend

- **Name**: Cuda
- **Backend type**: GPU (NVIDIA)
- **Enable in Cargo.toml**:
  ```toml
  [dependencies]
  burn = { version = "0.20", features = ["cuda"] }
  ```
- **Example use**:
  ```rust
  use burn::backend::Cuda;

  type Backend = Cuda;
  ```

## 4. Rocm Backend

- **Name**: Rocm
- **Backend type**: GPU (AMD)
- **Enable in Cargo.toml**:
  ```toml
  [dependencies]
  burn = { version = "0.20", features = ["rocm"] }
  ```
- **Example use**:
  ```rust
  use burn::backend::Rocm;

  type Backend = Rocm;
  ```