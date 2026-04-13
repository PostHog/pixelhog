/// Errors returned by pixelhog operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// PNG decoding failed.
    #[error("failed to decode PNG: {0}")]
    Decode(#[from] image::ImageError),

    /// PNG encoding failed.
    #[error("failed to encode PNG: {0}")]
    Encode(image::ImageError),

    /// RGBA buffer length does not match the declared dimensions.
    #[error(
        "invalid RGBA buffer length: expected {expected} bytes for {width}x{height}, got {actual}"
    )]
    BufferLength {
        expected: usize,
        actual: usize,
        width: usize,
        height: usize,
    },

    /// Image dimensions overflow when computing buffer sizes.
    #[error("image dimensions overflowed")]
    Overflow,

    /// A parameter is outside its valid range.
    #[error("{0}")]
    InvalidOption(&'static str),

    /// Source image is larger than the target padding size (internal invariant).
    #[error("source image is larger than target size")]
    PadOverflow,

    /// Width or height value does not fit in a `u32` (PNG limit).
    #[error("image {dimension} is too large")]
    DimensionTooLarge { dimension: &'static str },
}
