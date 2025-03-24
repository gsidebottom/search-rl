Needed to explicitly set the `bincode` version in `Cargo.lock` to make
`burn` happy.  

```text
[[package]]
name = "bincode"
version = "2.0.0-rc.3"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "f11ea1a0346b94ef188834a65c068a03aec181c94896d481d7a0a40d85b0ce95"
dependencies = [
 "serde",
]
```