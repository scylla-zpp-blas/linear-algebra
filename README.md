
# Scylla BLAS

Implementation of (part of) BLAS specification that uses Scylla database to store matrices/vectors.

Algorithms are parallel - main library divides task into subtasks, and delegates them to workers - all communication is done trough Scylla.



## Building

Clone the project

```bash
  https://github.com/scylla-zpp-blas/linear-algebra
```

Go to the project directory

```bash
  cd linear-algebra
```

Update submodules

```bash
  git submodule update --init --recursive
```

Create build directory and cd into it
```bash
  mkdir build
```

Build project
```bash
  cmake ..
  make
```

This will build in Release mode. If you want to build in Debug mode add `-DCMAKE_BUILD_TYPE=Debug` flag to cmake (before `..`).

To change log level, use `-DSCYLLA_BLAS_LOGLEVEL=<level>` where `<level>` is one of `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`, `CRITICAL`.

Tests are by default built only in Debug mode. You can override that behaviour with `-DBUILD_TESTS=ON/OFF`.
Example programs are always built by default. You can disable that with `-DBUILD_EXAMPLES=ON/OFF`.

Files produced by build:
- `.so` files - the library to link against.
- `scylla_blas_worker`
- (only in Debug) `tests/scylla_blas_tests`
## Usage

To use library, you need to add `include` to your inlude directories, and link against `.so` library that was built.

Before using, database must be initialized: `./scylla_blas_worker --init -H scylla_address`.

To run a worker: `./scylla_blas_worker --worker -H scylla_address`



## Authors

- [Karol Baryła](https://github.com/Lorak-mmk)
- [Robert Michna](https://github.com/Rhantolq)
- [Przemysław Podleśny](https://github.com/jbhayven)
- [Szymon Stolarczyk](https://github.com/s-stol)

## License

[Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0.txt)
  