pub mod patch;
pub mod find;
pub mod rife;
pub mod utils;


pub struct Args {
    /// Minium confidence to consider two frames distinct
    /// Range: 0.0..1.0 (1.0 = max confidence, 0.0 = no confidence)
    pub min_duplicate_confidence: f32,
    /// Minimum number of consecutive duplicate frames required to trigger interpolation
    /// Range: 1..
    pub min_duplicates: usize,
    /// Maximum number of consecutive duplicate frames to interpolate
    /// Range: 1..
    pub max_duplicates: usize,
    /// Alpha value for exponential moving average used in recent motion calculation (Responds fast)
    /// Range: 0.5..1.0 (lower = smoother, but slower to adapt)
    pub recent_motion_mean_alpha: f32,
    /// Alpha value for exponential moving average used in background motion calculation (Responds slow)
    /// Range: 0.0..0.4 (lower = smoother, but slower to adapt)
    pub slow_motion_mean_alpha: f32,
    /// Multiplier applied to recent motion to calculate the required motion threshold for the motion compensation
    /// Range: 0.0..1.0 (lower = less compensation, higher = more compensation)
    pub motion_compensate_threshold: f32,
    /// Multiplier applied to the motion compensation threshold, to require less motion when retrying a failed motion compensation
    /// Range: 0.0..1.0 (lower = less compensation, higher = more compensation)
    pub mul_motion_compensate_threshold_retry: f32,
    /// How many duplicate frames are needed for the motion compensation to be active
    /// Range: 1..
    pub motion_compensate_start: usize,
    /// Maximum multiple of the background average motion allowed to still interpolate
    /// Allows for more interpolation in low motion areas while thwarting troubling high motion areas to be interpolated too much
    /// Range: 1.0.. (higher = more motion allowed to be interpolated)
    pub max_motion_mul: f32,

    /// Factor by which input frames are downscaled for perceptual hashing
    /// Range: 1.. (lower = higher hash resolution, more sensitive to small differences)
    pub diff_hash_resize: u32,
    /// Minimum allowed hash difference to definitely consider two frames distinct.
    /// This skips calculating the more costly distinctness model.
    /// Range: 0.0..1.0 (1.0 = completely different frame, 0.0 = hash identical)
    pub min_hash_diff: f32,
}

pub const VIDEO_SWS_FLAGS: &str = "accurate_rnd+full_chroma_inp+full_chroma_int";

pub const VIDEO_DECODE_ARGS: [&str; 7] = [
    "-sws_flags", VIDEO_SWS_FLAGS, // Important to get same(ish) colors
    "-f", "rawvideo",
    "-pix_fmt", "rgb24",
    "-"
];