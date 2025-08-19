use ffmpeg_sidecar::event::OutputVideoFrame;
use fast_image_resize::images::Image;
use fast_image_resize::{CpuExtensions, FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image_hasher::HashAlg;
use image::DynamicImage;
use crate::Args;
use crate::find::tiny_motion_net;

#[inline]
fn get_luma(r: u8, g: u8, b: u8) -> u32 {
    ((r as u32 * 77) + (g as u32 * 150) + (b as u32 * 29)) >> 8 // Approx Rec. 709 luma
}

fn to_grayscale(frame: &OutputVideoFrame) -> Vec<u8> {
    let src = &frame.data;
    let n = src.len() / 3;
    let mut gray = vec![0u8; n];
    for (i, pixel) in frame.data.chunks_exact(3).enumerate() {
        let y = get_luma(pixel[0], pixel[1], pixel[2]) as u8;
        unsafe {
            // SAFETY: gray is allocated with length n
            *gray.get_unchecked_mut(i) = y;
        }
    }

    gray
}

pub struct FrameDifference {
    pub hash_distance: f32,
    pub motion_estimate: f32
}

impl FrameDifference {
    pub const INFINITY: FrameDifference = FrameDifference {
        hash_distance: f32::INFINITY,
        motion_estimate: f32::INFINITY
    };

    pub fn is_normal(&self) -> bool {
        self.hash_distance.is_normal() && self.motion_estimate.is_normal()
    }
}

pub fn compare_frames(a: &OutputVideoFrame, b: &OutputVideoFrame, args: &Args) -> FrameDifference {
    assert_eq!(a.width, b.width);
    assert_eq!(a.height, b.height);
    // Grayscale
    let gray_a = to_grayscale(a);
    let gray_b = to_grayscale(b);
    let img_a = Image::from_vec_u8(a.width, a.height, gray_a, PixelType::U8).expect("Image buffer should be valid");
    let img_b = Image::from_vec_u8(b.width, b.height, gray_b, PixelType::U8).expect("Image buffer should be valid");
    // Calc motion
    let motion_estimate = tiny_motion_net::predict_motion(&img_a, &img_b);
    // Setup PHash
    let hash_width = a.width / args.diff_hash_resize;
    let hash_height = a.height / args.diff_hash_resize;
    let hasher = image_hasher::HasherConfig::new()
        .hash_size(hash_width, hash_height)
        .resize_filter(image::imageops::FilterType::Nearest) // Actual resizing happened already
        .hash_alg(HashAlg::Gradient)
        .to_hasher();
    // Pre-resize because image-rs resizing in Hasher is slow.
    let mut resizer = Resizer::new();
    #[cfg(target_arch = "x86_64")]
    unsafe {
        resizer.set_cpu_extensions(CpuExtensions::Avx2);
    }
    let resize_options = ResizeOptions::new()
        .resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom));
    let width = hash_width + 1; // According to HashAlg::Gradient (row-major)
    let height = hash_height;
    let mut resized_a = DynamicImage::new(width, height, image::ColorType::L8);
    let mut resized_b = DynamicImage::new(width, height, image::ColorType::L8);
    resizer.resize(&img_a, &mut resized_a, &resize_options).expect("Resize should not fail");
    resizer.resize(&img_b, &mut resized_b, &resize_options).expect("Resize should not fail");
    // Hash
    let hash_a = hasher.hash_image(&resized_a);
    let hash_b = hasher.hash_image(&resized_b);
    let hash_distance = hash_a.dist(&hash_b) as f32 / (hash_width * hash_height) as f32;
    FrameDifference {
        hash_distance,
        motion_estimate,
    }
}